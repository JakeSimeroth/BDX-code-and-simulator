"""Train the locomotion substrate at scale in Isaac Lab (rsl_rl PPO), then export
the actor to the numpy ``.npz`` the on-robot runtime loads — so the GPU-trained
policy runs torch-free on the Jetson with identical observations.

Run under an Isaac Lab checkout's Python:

    python -m gardener_bdx.training.train_isaaclab --num_envs 4096 --headless \
        --max_iterations 1500 --out models/policies/locomotion.npz

Isaac Lab is installed via its repo (not pip); see docs/ISAACLAB.md."""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Isaac Lab locomotion training for GardenerBDX")
    parser.add_argument("--num_envs", type=int, default=4096)
    parser.add_argument("--max_iterations", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="models/policies/locomotion.npz")
    parser.add_argument("--smoke", action="store_true",
                        help="pipeline check, not a training run: 64 envs / 20 iters "
                             "(~2-3 min) to validate USD, env, PPO, and the .npz export "
                             "BEFORE committing to the long run")

    # AppLauncher injects --headless, --device, etc., and must boot before any
    # omni/isaaclab import.
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.smoke:
        args.num_envs = min(args.num_envs, 64)
        args.max_iterations = min(args.max_iterations, 20)
        args.out = "models/policies/locomotion_smoke.npz"  # don't shadow a real policy
        print("[smoke] 64 envs / 20 iters -> models/policies/locomotion_smoke.npz "
              "(validates the pipeline; the gait will NOT walk yet)")
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    # ---- imports valid only after the app exists --------------------------
    import torch
    from rsl_rl.runners import OnPolicyRunner

    from isaaclab_rl.rsl_rl import (
        RslRlOnPolicyRunnerCfg,
        RslRlPpoActorCriticCfg,
        RslRlPpoAlgorithmCfg,
        RslRlVecEnvWrapper,
    )

    from .isaaclab_locomotion_env import GardenerBdxFlatEnvCfg, GardenerBdxLocomotionEnv

    env_cfg = GardenerBdxFlatEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.sim.device = args.device

    agent_cfg = RslRlOnPolicyRunnerCfg(
        num_steps_per_env=24,
        max_iterations=args.max_iterations,
        save_interval=100,
        experiment_name="gardener_bdx_locomotion",
        empirical_normalization=False,  # keep export simple: no obs normalizer to fold in
        policy=RslRlPpoActorCriticCfg(
            init_noise_std=1.0,
            actor_hidden_dims=[256, 128],
            critic_hidden_dims=[256, 128],
            activation="elu",
        ),
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0, use_clipped_value_loss=True, clip_param=0.2,
            entropy_coef=0.005, num_learning_epochs=5, num_mini_batches=4,
            learning_rate=1e-3, schedule="adaptive", gamma=0.99, lam=0.95,
            desired_kl=0.01, max_grad_norm=1.0,
        ),
    )

    env = GardenerBdxLocomotionEnv(env_cfg)
    env = RslRlVecEnvWrapper(env)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir="logs/gardener_bdx", device=args.device)
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    export_actor_to_npz(runner, args.out)
    simulation_app.close()


def export_actor_to_npz(runner, path: str) -> None:
    """Pull the rsl_rl actor MLP into the runtime's numpy format (ELU MLP)."""
    import torch

    from ..policy.locomotion import MLP

    actor = runner.alg.actor_critic.actor
    layers = [(m.weight.detach().cpu().numpy(), m.bias.detach().cpu().numpy())
              for m in actor.modules() if isinstance(m, torch.nn.Linear)]
    MLP.save_npz(path, layers, activation="elu")
    print(f"exported locomotion policy -> {path} ({len(layers)} layers, ELU)")


if __name__ == "__main__":
    main()
