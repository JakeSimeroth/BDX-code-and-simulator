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
- **Sim2Real / DR**: built in via Isaac Lab's event manager (`EventCfg`):
  `randomize_rigid_body_material` (friction/restitution), `randomize_rigid_body_mass`
  (torso payload spread), and interval `push_by_setting_velocity` shoves, plus a
  velocity-command curriculum (interval resample) and random initial heading +
  joint jitter on reset. Tune the ranges in `EventCfg`; switch material/mass to
  `mode="reset"` for per-episode re-randomization.
- **Action convention**: clipped to [-1, 1] then scaled by `action_scale`, exactly
  as the numpy env and the runtime — so the exported policy behaves the same.

## What you need to do

1. **Install Isaac Lab** (and Isaac Sim) per the official guide:
   https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html .
   You need an NVIDIA RTX GPU. Run training under Isaac Lab's Python environment.
2. **Robot asset (USD).** One command — the repo ships a URDF + converter:
   ```bash
   python scripts/convert_to_usd.py            # models/robot/gardener_bdx.urdf -> .usd
   ```
   (or `./scripts/setup_omniverse.sh` to install + convert + smoke-train in one go).
   The bundled `models/robot/gardener_bdx.urdf` matches `configs/robot/gardener_bdx.yaml`
   joint names/order exactly. Tune link masses/inertias and collision meshes to
   your real robot before serious Sim2Real (the inertials are placeholders).
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
