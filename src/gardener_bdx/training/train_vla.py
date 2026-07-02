"""Distill the scripted expert into the end-to-end ``NeuralVLA`` (behavior
cloning). Two losses:

  * **System 1 (action):** flow-matching / rectified-flow loss so the action
    head learns to generate the expert's whole-body command from pixels+proprio.
  * **System 2 (skill):** cross-entropy on the expert's skill choice, so the
    reasoning latent organizes around the task structure.

    python -m gardener_bdx.training.train_vla --data data/expert_demos.npz \
        --epochs 30 --out models/policies/vla_bc.pt

This is the bootstrap. After BC, RL-finetune in Isaac Lab against the task
reward (and, ultimately, swap the backbone for GR00T N1 and fine-tune on the
same data). See docs/SIM2REAL.md. The closed loop — expert teaches pixels-policy
in sim, policy is hardened by RL, then ported — is the whole point."""

from __future__ import annotations

import argparse

import numpy as np

_OPTIONAL_KEYS = ("depths", "bev", "expressions")


def _load_datasets(spec: str) -> dict:
    """Load one or more demo ``.npz`` files (comma-separated) into one dict —
    e.g. BC + DAgger rounds: ``--data data/expert_demos.npz,data/dagger_demos.npz``.
    Episode boundaries are re-offset; optional channels missing from a file are
    zero-filled so mixed-generation datasets still train."""
    paths = [p.strip() for p in spec.split(",") if p.strip()]
    parts = [dict(np.load(p, allow_pickle=True)) for p in paths]
    if len(parts) == 1:
        return parts[0]

    hw = parts[0]["images"].shape[1:3]
    for p, part in zip(paths, parts):
        assert part["images"].shape[1:3] == hw, f"{p}: image size {part['images'].shape[1:3]} != {hw}"

    out: dict = {}
    for key in ("proprio", "tokens", "actions", "skills", "images", "instructions"):
        out[key] = np.concatenate([p[key] for p in parts])
    for key in _OPTIONAL_KEYS:
        if any(key in p for p in parts):
            n_of = {"depths": lambda p: (len(p["images"]), *hw),
                    "bev": lambda p: (len(p["images"]), 64, 64),
                    "expressions": lambda p: (len(p["images"]),)}[key]
            dtype = np.int64 if key == "expressions" else np.float32
            out[key] = np.concatenate([
                np.asarray(p[key]) if key in p else np.zeros(n_of(p), dtype) for p in parts
            ])
    ends, off = [], 0
    for p in parts:
        ends.extend((np.asarray(p["episode_ends"]) + off).tolist())
        off += len(p["actions"])
    out["episode_ends"] = np.array(ends, np.int64)
    print(f"[data] {len(paths)} datasets merged: {[len(p['actions']) for p in parts]} samples")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/expert_demos.npz",
                    help="demo .npz path(s), comma-separated (e.g. BC + DAgger rounds)")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="models/policies/vla_bc.pt")
    ap.add_argument("--skill-weight", type=float, default=0.5,
                    help="weight on the System-2 skill cross-entropy vs. the flow loss")
    ap.add_argument("--expr-weight", type=float, default=0.25,
                    help="weight on the expression (animation-selection) cross-entropy")
    args = ap.parse_args()

    try:
        import torch
        import torch.nn.functional as F
    except ImportError as e:
        raise SystemExit(f"VLA training needs torch: pip install -e '.[vla]' ({e})")

    from ..policy.vla_brain import ACTION_DIM
    from ..policy.vla_net import RGB_HW, WorldModelVLANet

    d = _load_datasets(args.data)
    dev = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    N = d["actions"].shape[0]

    net = WorldModelVLANet(action_dim=ACTION_DIM).to(dev)
    T = net.history_len

    # Standardize the (mixed-scale) action vector and detect which optional
    # vision channels this dataset actually carries — both are saved in the
    # checkpoint so inference feeds the net exactly what it trained on.
    a_np = d["actions"].astype(np.float32)
    net.set_action_stats(a_np.mean(0), a_np.std(0))
    has_depth = "depths" in d and bool(np.any(d["depths"]))
    has_bev = "bev" in d and bool(np.any(d["bev"]))
    net.set_modalities(has_depth, has_bev)
    print(f"loaded {N} samples from {args.data}; training on {dev} "
          f"(history={T}, depth={has_depth}, bev={has_bev})")

    proprio = torch.tensor(d["proprio"], dtype=torch.float32)
    tokens = torch.tensor(d["tokens"], dtype=torch.long)
    actions = torch.tensor(a_np, dtype=torch.float32)
    skills = torch.tensor(d["skills"], dtype=torch.long)
    # Expression supervision (older datasets predate the channel -> all NONE).
    exprs = torch.tensor(
        d["expressions"] if "expressions" in d else np.zeros(N, np.int64),
        dtype=torch.long,
    )
    images = torch.tensor(d["images"], dtype=torch.float32).permute(0, 3, 1, 2) / 255.0  # (N,3,48,64)
    depth = (torch.tensor(d["depths"], dtype=torch.float32).unsqueeze(1) / 5.0
             if has_depth else torch.zeros(N, 1, images.shape[2], images.shape[3]))
    bev_all = torch.tensor(d["bev"], dtype=torch.float32).unsqueeze(1) if has_bev else None

    # Temporal windows: for each sample i, the T indices [i-T+1 .. i] clamped to
    # the start of i's episode (front-padded, never crossing an episode boundary).
    ends = d["episode_ends"].tolist() if "episode_ends" in d else [N]
    ep_start = np.zeros(N, dtype=np.int64)
    s0 = 0
    for e in ends:
        ep_start[s0:e] = s0
        s0 = e
    ar = np.arange(N)
    win = np.stack([np.maximum(ep_start, ar - (T - 1 - k)) for k in range(T)], axis=1)  # (N,T)
    win = torch.tensor(win, dtype=torch.long)

    def make_img_seq(rgb_seq, depth_seq):  # (B,T,3,h,w)+(B,T,1,h,w) -> (B,T,4,96,96)
        b, t = rgb_seq.shape[:2]
        x = torch.cat([rgb_seq.flatten(0, 1), depth_seq.flatten(0, 1)], dim=1)
        x = F.interpolate(x, size=(RGB_HW, RGB_HW), mode="bilinear", align_corners=False)
        return x.view(b, t, 4, RGB_HW, RGB_HW)

    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs))
    n_val = max(1, N // 10)
    train_idx = torch.arange(0, N - n_val)
    val_idx = torch.arange(N - n_val, N)             # last 10% as a quick holdout

    def batch_loss(idx):
        wb = win[idx]                                       # (B,T)
        img = make_img_seq(images[wb], depth[wb]).to(dev)   # (B,T,4,96,96)
        bev = (bev_all[wb].to(dev) if bev_all is not None
               else torch.zeros(len(idx), T, 1, 64, 64, device=dev))
        pr = proprio[wb].to(dev)                            # (B,T,P)
        tok = tokens[idx].to(dev)                           # (B,L) — const within an episode
        act = net.normalize_action(actions[idx].to(dev))    # learn in standardized space
        sk = skills[idx].to(dev)
        ex = exprs[idx].to(dev)
        latent, skill_logits, proprio_embed = net.encode(img, bev, pr, tok)
        cond = net.action_cond(latent, proprio_embed, skill_logits)
        return (net.flow_matching_loss(cond, act)
                + args.skill_weight * F.cross_entropy(skill_logits, sk)
                + args.expr_weight * F.cross_entropy(net.expression_logits(latent), ex))

    for epoch in range(args.epochs):
        net.train()
        perm = train_idx[torch.randperm(len(train_idx))]
        tot = 0.0
        for s in range(0, len(perm), args.batch):
            idx = perm[s : s + args.batch]
            loss = batch_loss(idx)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            tot += loss.item() * len(idx)
        sched.step()
        net.eval()
        with torch.no_grad():
            val = batch_loss(val_idx).item()
        print(f"epoch {epoch+1}/{args.epochs}  train_loss={tot / len(perm):.4f}  val_loss={val:.4f}")

    torch.save({"model": net.state_dict()}, args.out)
    print(f"saved VLA checkpoint -> {args.out}")


if __name__ == "__main__":
    main()
