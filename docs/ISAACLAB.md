# Isaac Lab training

Train the fluid BDX walk at scale (thousands of envs on one GPU), then export a
numpy policy the robot runs torch-free. The Isaac Lab env
(`training/isaaclab_locomotion_env.py`) is a `DirectRLEnv` whose observation and
reward are **identical** to the runtime + the numpy RL env, so the policy
transfers without drift.

## Pipeline

```bash
# 0) (once) convert the robot to USD — see "Robot asset" below
# 1) train at scale under Isaac Lab's python
python -m gardener_bdx.training.train_isaaclab --num_envs 4096 --headless \
    --max_iterations 1500 --out models/policies/locomotion.npz
# 2) run it in any twin / on the robot (numpy inference, no torch needed)
python -m gardener_bdx.runtime.sim_main --backend mujoco --render
```

The trainer uses rsl_rl PPO and `export_actor_to_npz()` writes the actor MLP into
`models/policies/locomotion.npz` (ELU MLP), exactly what `LocomotionPolicy`
loads. `empirical_normalization=False` keeps the export a pure MLP — if you turn
observation normalization on, fold the running mean/var into the export.

## What this env gives you

- **Observation** (`_get_observations`): projected gravity, base angular velocity,
  joint pos − default, joint velocity, last action, the velocity command, and a
  gait clock — the same 47-dim vector (for 12 joints) as the runtime.
- **Reward** (`_get_rewards`): Disney-style imitation to a reference gait +
  velocity tracking + upright + height + energy/action-rate regularizers + alive,
  mirroring `training/rewards.py`.
- **Sim2Real**: per-episode command curriculum + random base pushes; add Isaac
  Lab material/mass/friction events for full domain randomization.
- **Action convention**: clipped to [-1, 1] then scaled by `action_scale`, exactly
  as the numpy env and the runtime — so the exported policy behaves the same.

## What you need to do

1. **Install Isaac Lab** (and Isaac Sim) per the official guide:
   https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html .
   You need an NVIDIA RTX GPU. Run training under Isaac Lab's Python environment.
2. **Robot asset (USD).** Provide `models/robot/gardener_bdx.usd`. Two options:
   - Convert the bundled MJCF/URDF with Isaac Lab's converters
     (`scripts/tools/convert_mjcf.py` / `convert_urdf.py` in the Isaac Lab repo), or
   - Author it in Isaac Sim and keep the joint **names/order** matching
     `configs/robot/gardener_bdx.yaml` (the env maps joints by the config order).
   Tune link masses/inertias and collision meshes to your real robot.
3. **Greenhouse scene (optional, for the full task / VLA).** Author
   `models/scenes/greenhouse.usd` (benches, pots, dock) for `sim/isaac_backend.py`
   and add RTX camera + RTX-LiDAR prims on the head; wire their handles where
   marked in that file.
4. **(Optional) register a Gym id** so the standard `isaaclab.../train.py` runner
   can launch it; the bundled `train_isaaclab.py` already builds it directly.

## Mapping to a registered task (optional)

If you prefer Isaac Lab's task registry + `rsl_rl`/`rl_games` CLIs, register
`GardenerBdxLocomotionEnv`/`GardenerBdxFlatEnvCfg` with `gymnasium.register(...)`
under an id like `Isaac-GardenerBDX-Locomotion-Direct-v0` and point an
`rsl_rl` agent cfg at it; the env is already structured for that.
