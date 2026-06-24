"""Train the locomotion substrate with PPO, then export it to the numpy ``.npz``
the runtime loads (no torch on the robot).

    python -m gardener_bdx.training.train_locomotion --timesteps 50_000_000 \
        --backend mujoco --num-envs 4096 --out models/policies/locomotion.npz

For serious training use Isaac Lab's vectorized envs (thousands in parallel on
one GPU); this SB3 path is the reference/CPU-friendly version and keeps the
export format identical."""

from __future__ import annotations

import argparse

import numpy as np

from ..common.config import RobotConfig
from ..policy.locomotion import MLP
from .locomotion_env import LocomotionEnv


def export_sb3_policy(model, path: str) -> None:
    """Pull the actor MLP out of a Stable-Baselines3 PPO model into our numpy
    format (Linear layers in order, tanh activations)."""
    import torch  # noqa: WPS433

    layers = []
    modules = list(model.policy.mlp_extractor.policy_net) + [model.policy.action_net]
    for m in modules:
        if isinstance(m, torch.nn.Linear):
            layers.append((m.weight.detach().cpu().numpy(), m.bias.detach().cpu().numpy()))
    MLP.save_npz(path, layers, activation="tanh")
    print(f"exported locomotion policy -> {path} ({len(layers)} layers)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=5_000_000)
    ap.add_argument("--backend", default="mujoco")
    ap.add_argument("--num-envs", type=int, default=8)
    ap.add_argument("--out", default="models/policies/locomotion.npz")
    ap.add_argument("--no-domain-rand", action="store_true")
    args = ap.parse_args()

    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import SubprocVecEnv
    except ImportError as e:
        raise SystemExit(f"Training needs stable-baselines3 + torch: pip install -e '.[train]' ({e})")

    rc = RobotConfig.from_yaml()

    def make(rank: int):
        def _f():
            return LocomotionEnv(rc, backend_name=args.backend, domain_rand=not args.no_domain_rand, seed=rank)
        return _f

    venv = SubprocVecEnv([make(i) for i in range(args.num_envs)])
    model = PPO(
        "MlpPolicy", venv, verbose=1, n_steps=32, batch_size=4096, gae_lambda=0.95,
        gamma=0.99, learning_rate=3e-4, ent_coef=0.0,
        policy_kwargs=dict(net_arch=dict(pi=[256, 128], vf=[256, 128]), activation_fn=__import__("torch").nn.Tanh),
    )
    model.learn(total_timesteps=args.timesteps)
    export_sb3_policy(model, args.out)


if __name__ == "__main__":
    main()
