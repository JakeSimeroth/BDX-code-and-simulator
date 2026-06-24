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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/expert_demos.npz")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="models/policies/vla_bc.pt")
    args = ap.parse_args()

    try:
        import torch
        import torch.nn.functional as F
    except ImportError as e:
        raise SystemExit(f"VLA training needs torch: pip install -e '.[vla]' ({e})")

    from ..policy.vla_brain import ACTION_DIM
    from ..policy.vla_net import RGB_HW, WorldModelVLANet

    d = np.load(args.data, allow_pickle=True)
    dev = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    N = d["actions"].shape[0]

    net = WorldModelVLANet(action_dim=ACTION_DIM).to(dev)
    T = net.history_len
    print(f"loaded {N} samples from {args.data}; training on {dev} (history={T})")

    proprio = torch.tensor(d["proprio"], dtype=torch.float32)
    tokens = torch.tensor(d["tokens"], dtype=torch.long)
    actions = torch.tensor(d["actions"], dtype=torch.float32)
    skills = torch.tensor(d["skills"], dtype=torch.long)
    images = torch.tensor(d["images"], dtype=torch.float32).permute(0, 3, 1, 2) / 255.0  # (N,3,48,64)

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

    def make_img_seq(seq):  # seq: (B,T,3,48,64) -> (B,T,4,96,96)
        b, t = seq.shape[:2]
        x = seq.flatten(0, 1)
        x = torch.cat([x, torch.zeros(x.shape[0], 1, x.shape[2], x.shape[3])], dim=1)
        x = F.interpolate(x, size=(RGB_HW, RGB_HW), mode="bilinear", align_corners=False)
        return x.view(b, t, 4, RGB_HW, RGB_HW)

    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    for epoch in range(args.epochs):
        perm = torch.randperm(N)
        tot = 0.0
        for s in range(0, N, args.batch):
            idx = perm[s : s + args.batch]
            wb = win[idx]                                   # (B,T)
            img = make_img_seq(images[wb]).to(dev)          # (B,T,4,96,96)
            bev = torch.zeros(len(idx), T, 1, 64, 64, device=dev)
            pr = proprio[wb].to(dev)                        # (B,T,P)
            tok = tokens[idx].to(dev)                       # (B,L) — instruction const within episode
            act = actions[idx].to(dev)
            sk = skills[idx].to(dev)

            latent, skill_logits, proprio_embed = net.encode(img, bev, pr, tok)
            cond = torch.cat([latent, proprio_embed], dim=-1)
            loss = net.flow_matching_loss(cond, act) + F.cross_entropy(skill_logits, sk)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        print(f"epoch {epoch+1}/{args.epochs}  loss={tot / N:.4f}")

    torch.save({"model": net.state_dict()}, args.out)
    print(f"saved VLA checkpoint -> {args.out}")


if __name__ == "__main__":
    main()
