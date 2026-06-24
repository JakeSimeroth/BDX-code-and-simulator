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

    d = np.load(args.data)
    dev = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    N = d["actions"].shape[0]
    print(f"loaded {N} samples from {args.data}; training on {dev}")

    proprio = torch.tensor(d["proprio"], dtype=torch.float32)
    tokens = torch.tensor(d["tokens"], dtype=torch.long)
    actions = torch.tensor(d["actions"], dtype=torch.float32)
    skills = torch.tensor(d["skills"], dtype=torch.long)
    images = torch.tensor(d["images"], dtype=torch.float32).permute(0, 3, 1, 2) / 255.0

    def make_img(batch_imgs):
        depth = torch.zeros(batch_imgs.shape[0], 1, batch_imgs.shape[2], batch_imgs.shape[3])
        img = torch.cat([batch_imgs, depth], dim=1)
        return F.interpolate(img, size=(RGB_HW, RGB_HW), mode="bilinear", align_corners=False)

    net = WorldModelVLANet(action_dim=ACTION_DIM).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        perm = torch.randperm(N)
        tot = 0.0
        for s in range(0, N, args.batch):
            idx = perm[s : s + args.batch]
            img = make_img(images[idx]).to(dev)
            bev = torch.zeros(len(idx), 1, 64, 64, device=dev)
            pr = proprio[idx].to(dev)
            tok = tokens[idx].to(dev)
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
