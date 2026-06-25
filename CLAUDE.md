# CLAUDE.md — orientation for a Claude Code session in this repo

You (Claude) are continuing **GardenerBDX**: a VLA-driven, bipedal, BDX-inspired
greenhouse-gardener droid — a simulation-first digital twin that ports to a
Jetson later. Read this first; it's the fast path to being useful here.

## What this is

End-to-end, **non-ROS** control stack. A single **Vision-Language-Action world
model** is the brain; it expresses intent through a fast learned locomotion
substrate and is screened by a deterministic, NVIDIA-Halos-aligned safety
Guardian. Everything talks to one hardware-abstraction seam (`RobotIO`) so the
*same* control code runs in the digital twin and on the robot.

Architecture (four tiers, decoupled clocks) — details in `docs/ARCHITECTURE.md`:
- **System 2/1 VLA** (`src/gardener_bdx/policy/vla_brain.py`, `vla_net.py`,
  `groot_vla.py`): GR00T-style reasoning (temporal transformer over a 4-frame
  history + residual vision encoder) + a flow-matching action head. Two impls
  behind one `VLAPolicy`: `NeuralVLA` (ours, torch) and `ScriptedGardenerVLA`
  (the runs-anywhere expert/teacher). GR00T N1 adapter is `GR00TVLA`.
- **System 0 locomotion** (`policy/locomotion.py`): RL gait policy, numpy
  inference (train in torch → export `.npz` → run torch-free), CPG fallback.
- **Safety Guardian** (`safety/guardian.py`): deterministic limits, tip-over,
  human-stop, geofence, water interlocks, e-stop.
- **Perception** (`perception/`): LiDAR+camera → occupancy + plant/human/dock belief.

Backends (all implement `RobotIO`): `sim/kinematic_backend.py` (pure numpy, runs
anywhere), `sim/mujoco_backend.py` (physics), `sim/isaac_backend.py` (Isaac Sim,
interactive GUI + eval), `hardware/jetson_backend.py` (real robot, driver seams).

## Current state (read before acting)

- **Implemented + tested**: 29 passing tests (`pytest`). Kinematic twin runs the
  full gardener loop; MuJoCo backend + MJCF/URDF verified to load and step.
- **NOT yet trained**: there is no trained gait or VLA checkpoint yet. Without a
  trained policy the robot uses the CPG fallback and **falls in physics** (the
  Guardian e-stops — expected). Training is the immediate GPU work.
- **Isaac code is a GPU-only template**: `sim/isaac_backend.py` and
  `training/isaaclab_locomotion_env.py` are written against the Isaac Lab 2.x /
  Isaac Sim API but were **never executed** (prior dev box had no GPU). Expect to
  fix small version mismatches (esp. `omni.isaac.core` vs `isaacsim.core`
  namespaces) on first run. RTX camera/LiDAR are marked seams (`_read_camera`/
  `_read_lidar`), so the Isaac twin currently runs on ground-truth `semantics()`.
- Branch: `claude/magical-hamilton-q8uu91`. The single source of truth for the
  robot is `configs/robot/gardener_bdx.yaml` (joint names/order, limits, gains).

## Immediate next steps on the GPU (the plan)

The runbook is `docs/GETTING_STARTED_GPU.md`; the interactive-Isaac guide is
`docs/ISAACSIM.md`. In order:

1. `python scripts/preflight.py` — confirm CUDA/GPU, MuJoCo, Isaac, deps.
2. `python scripts/convert_to_usd.py` — URDF → USD for Isaac.
3. Train the walk: `python -m gardener_bdx.training.train_isaaclab --num_envs 4096 --headless`
   → exports `models/policies/locomotion.npz`. Watch curves: `tensorboard --logdir logs/`.
4. See it: `python scripts/run_isaac.py --policy models/policies/locomotion.npz`
   (interactive GUI) or `python scripts/view_mujoco.py --view`.
5. VLA brain: `collect_demos` → `train_vla` → `evaluate --compare`.
6. Expect to debug Isaac version/namespace issues at step 2–4; fix in
   `sim/isaac_backend.py` / `training/isaaclab_locomotion_env.py`.

## Conventions & gotchas

- **Run on the GPU box.** A cloud Claude Code session has no GPU; Isaac needs an
  RTX GPU. This repo is being run locally on an **RTX 4070, Windows 11 native**.
- **Windows**: the `.sh` scripts won't run natively — use the cross-platform
  `python -m ...` / `python scripts/*.py` commands, or `scripts/setup_omniverse.ps1`.
  Isaac Lab's wrapper is `isaaclab.bat` (not `.sh`). venv activate is
  `.\<venv>\Scripts\activate`. Python **3.11** (Isaac Sim requirement).
- **Editing the robot**: change `configs/robot/gardener_bdx.yaml`, then mirror in
  `models/robot/gardener_bdx.xml` (MJCF) and `.urdf`; obs/action dims auto-update.
- **Policy export contract**: actions are clipped to [-1,1] then scaled by
  `action_scale` in the numpy env, the Isaac env, AND the runtime — keep them in sync.
- **Tests**: `pytest` (root `conftest.py` puts `src/` on path; torch paths are
  skipped if torch absent). Add safety cases to `tests/test_safety.py`; use
  `training/evaluate.py` as the task-success gate.
- Commit/push to the feature branch when work is complete; don't open a PR unless asked.

## Docs index
`README.md` · `docs/GETTING_STARTED_GPU.md` (start here on GPU) · `ISAACSIM.md` ·
`ISAACLAB.md` · `GROOT.md` · `ARCHITECTURE.md` · `SIMULATION.md` · `SIM2REAL.md` ·
`SAFETY.md` · `HARDWARE.md`.
