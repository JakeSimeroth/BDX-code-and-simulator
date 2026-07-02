# Getting started on your GPU box (RTX 4070)

A practical runbook: boot the project, see it move, train it, and then refine the
robot, the greenhouse, and the tests. Written for a single workstation with an
**RTX 4070 (12 GB)** — plenty for the locomotion RL and the NeuralVLA brain;
GR00T fine-tuning is the one memory-constrained piece (see the last section).

> **Windows 11 native + driving with Claude Code on the 4070?**
> (1) The `.sh` scripts don't run natively — use the cross-platform
> `python -m ...` / `python scripts\*.py` commands below, or
> `scripts\setup_omniverse.ps1`. Isaac Lab's wrapper is **`isaaclab.bat`**.
> (2) Do everything inside your **Python 3.11** Isaac venv
> (`.\<venv>\Scripts\activate`). (3) A fresh Claude Code session opened in this
> repo auto-reads **`CLAUDE.md`**, so it picks up the project state immediately —
> just say "read CLAUDE.md and run preflight."

---

## 0. First boot

**Prerequisites (RTX 4070 ✓):** Ubuntu 22.04/24.04 (or Windows; Isaac needs GLIBC
2.35+, so *not* Ubuntu 20.04), **Python 3.11**, NVIDIA driver **580.65+**, ≥32 GB
RAM, ~50 GB free disk. The 4070's 12 GB VRAM + RT cores meet the Isaac minimum.

The clean, current path is **pip Isaac Sim + Isaac Lab from source** — you do
*not* need the legacy Omniverse Launcher (it's deprecated). Use one Python 3.11
virtualenv for everything so Isaac Lab can import this package:

```bash
# 1) a Python 3.11 venv (Isaac Sim requires 3.11; the venv must match)
python3.11 -m venv ~/isaac && source ~/isaac/bin/activate
pip install --upgrade pip

# 2) Isaac Sim via pip, then Isaac Lab from source
pip install 'isaacsim[all,extscache]' --extra-index-url https://pypi.nvidia.com
git clone https://github.com/isaac-sim/IsaacLab.git ~/IsaacLab
cd ~/IsaacLab && ./isaaclab.sh --install        # installs Isaac Lab + rsl_rl into this venv
cd -

# 3) this project (into the SAME venv) + a quick check
git clone <your-fork> && cd BDX-code-and-simulator
git checkout claude/magical-hamilton-q8uu91
pip install -e '.[sim,vla,train,viz]'
python scripts/preflight.py
```

`preflight.py` prints a checklist (Python 3.11?, CUDA on?, which GPU, MuJoCo,
Isaac, GR00T, optional libs), runs a 200-tick gardener smoke test, and builds
the NeuralVLA on the GPU. Green ✓ = ready; yellow • = optional/missing with the
install hint.
Official install reference: <https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html>.

> **First-run gotchas (esp. Windows)**
> 1. **Python 3.11 exactly** — `pip install isaacsim` has wheels for 3.11 only
>    (preflight now checks). Build the venv with `py -3.11 -m venv ...`.
> 2. **EULA**: the first Isaac boot asks you to accept the NVIDIA EULA. For
>    scripted/headless runs: `$env:OMNI_KIT_ACCEPT_EULA='YES'`.
> 3. **The first boot "hangs"** — it's compiling RTX shaders; several minutes
>    is normal. Subsequent boots are fast.
> 4. **One venv = no wrapper needed**: because Isaac Sim/Lab are pip-installed
>    into the same venv as this repo, plain `python -m gardener_bdx...` works —
>    `isaaclab.bat -p` is only needed for source-checkout installs.
> 5. **Validate cheap before training long**: `python scripts/dev.py walk-smoke`
>    (64 envs / 20 iters, ~2-3 min) proves USD → env → PPO → export end-to-end
>    before you commit to the 4096-env run.
> 6. **If Isaac Lab RL fights your setup**, you're not blocked: the MuJoCo path
>    (`python -m gardener_bdx.training.train_locomotion --backend mujoco`)
>    trains a first walk on CPU/GPU without Isaac, exporting the same `.npz`.

**The task runner.** Every step below has a canonical verb in
`scripts/dev.py` (same on Windows/Linux; prints the underlying command it
runs): `python scripts/dev.py list`. A Claude Code session in this repo uses
the same verbs, so you and the agent share one vocabulary:

```powershell
python scripts\dev.py preflight     # readiness
python scripts\dev.py gif           # see the task (no GPU)
python scripts\dev.py usd           # robot -> USD
python scripts\dev.py walk-smoke    # 2-min Isaac pipeline check
python scripts\dev.py walk          # the real gait training
python scripts\dev.py isaac         # interactive photoreal twin
python scripts\dev.py demos         # expert demonstrations
python scripts\dev.py vla           # train the brain
python scripts\dev.py eval          # scoreboard -> out/eval_report.md
python scripts\dev.py dagger        # DAgger round (fixes BC drift), then vla again
```

Then convert the robot to USD and smoke-train (one command):

```bash
ISAACLAB_PATH=~/IsaacLab ./scripts/setup_omniverse.sh   # install + URDF→USD + smoke-train
```

> Note: if you're driving this from a cloud Claude Code session, that session has
> no GPU — run these on the 4070 itself (or run Claude Code locally on the 4070).

---

## 1. See it move (do this first — it builds intuition)

| You want to see... | Command | Needs |
|---|---|---|
| the **gardening behavior** (path, watering, dock, human), fast | `python scripts/view_kinematic.py --gif out/gardener.gif` | matplotlib |
| the **robot in physics** (gait, balance, contact) | `python scripts/view_mujoco.py --view` | mujoco + display |
| the **photoreal twin — and interact with it** (orbit, drag plants, lighting) | `python scripts/run_isaac.py` | Isaac Sim |

The kinematic GIF is the quickest sanity check of *task* logic. The MuJoCo
viewer is for *gait* debugging — with no trained policy the droid falls and the
Guardian e-stops (expected); after step 2 it walks. **Isaac Sim is the
interactive twin** you asked about — see §6 and [ISAACSIM.md](ISAACSIM.md).

---

## 2. Train the fluid walk (Isaac Lab, the 4070's sweet spot)

Locomotion RL is light on VRAM (no cameras), so the 4070 runs thousands of envs:

```bash
ISAACLAB=~/IsaacLab/isaaclab.sh
$ISAACLAB -p -m gardener_bdx.training.train_isaaclab \
    --num_envs 4096 --headless --max_iterations 1500
# OOM? drop to --num_envs 2048. Watching live? drop --headless and --num_envs 256.
```

Expect ~15-40 min for a decent walk on a 4070. It **exports**
`models/policies/locomotion.npz` — the numpy policy the runtime loads with no
torch. Deploy + watch it:

```bash
python scripts/view_mujoco.py --view --policy models/policies/locomotion.npz
python -m gardener_bdx.runtime.sim_main --backend mujoco   # full gardener loop, trained gait
```

Watch training curves with TensorBoard: `tensorboard --logdir logs/`.

Tuning lives in `training/isaaclab_locomotion_env.py`: reward weights
(`GardenerBdxFlatEnvCfg.w_*`), and domain randomization in `EventCfg` (friction,
mass, pushes). Widen DR for a more robust (but slower-to-learn) policy.

---

## 3. Train & score the VLA brain

```bash
# 1) demonstrations from the scripted expert (use mujoco --render for real pixels)
python -m gardener_bdx.training.collect_demos --backend mujoco --render --episodes 200
# 2) distill the end-to-end, temporal NeuralVLA from pixels (fits the 4070 easily)
python -m gardener_bdx.training.train_vla --epochs 30 --device cuda
# 3) run it, and score it against the teacher
python -m gardener_bdx.runtime.sim_main --backend mujoco --vla neural \
    --vla-ckpt models/policies/vla_bc.pt --render
python -m gardener_bdx.training.evaluate --compare   # success / collisions / time-to-complete
```

`evaluate --compare` is your **regression gate**: it prints plants serviced,
water delivered, collisions, completion time and safety interventions for the
scripted expert vs. your trained VLA — and writes `out/eval_report.{md,json}`
(stamped with the git commit) so runs are comparable across iterations. The
goal is for the learned policy to match then beat the teacher.

**When BC plateaus, run DAgger.** Behavior cloning drifts off the expert's
states (compounding small errors with no supervision on how to recover). One or
two DAgger rounds fix it — the *learned* policy drives, the *expert* labels the
states it actually visits:

```bash
python -m gardener_bdx.training.collect_demos --backend mujoco --render \
    --driver neural --vla-ckpt models/policies/vla_bc.pt \
    --episodes 100 --out data/dagger_demos.npz
python -m gardener_bdx.training.train_vla \
    --data data/expert_demos.npz,data/dagger_demos.npz --epochs 30 --device cuda
python -m gardener_bdx.training.evaluate --compare   # success rate should jump
```

---

## 4. Hardware selection — how to change the robot

Everything flows from one file: **`configs/robot/gardener_bdx.yaml`** (joint
names/order, limits, PD gains, torque/velocity limits, default stance, masses,
water capacity). It is the single source of truth shared by the policy, every
simulator, and the Jetson driver.

To pick real actuators/sensors, edit that file so sim matches your BOM:

- **Servo choice** → set per-joint `torque_limit` (N·m) and `velocity_limit`
  (rad/s) to the datasheet, and `kp/kd` to your position-control gains. These feed
  the MJCF/URDF actuators and the Isaac `ImplicitActuatorCfg`.
- **Adding/removing a DOF** (e.g., a waist or a 3-DOF neck): add the joint to
  `joints.names` (keep left/right legs first if you want the CPG/gait helpers to
  auto-detect them by name), extend every per-joint array, then mirror the joint
  in `models/robot/gardener_bdx.xml` (MJCF) **and** `.urdf`. The observation/action
  dims update automatically from `RobotConfig.n_joints`.
- **Water payload / battery** → `water_capacity_liters`, `mass_kg`, and the
  base-link mass in the MJCF/URDF (and the `base_mass` DR range in `EventCfg`).
- **Sensors** (camera, LiDAR, IMU, battery gauge, pump): these are *drivers*, not
  config — implement the seams in `hardware/jetson_backend.py` (see
  [HARDWARE.md](HARDWARE.md)). The robot config doesn't change for sensor swaps.

After editing joints, re-validate and regenerate USD:

```bash
python -m pytest tests/ -q                          # confirms config/MJCF/obs still consistent
python scripts/convert_to_usd.py                    # URDF -> USD for Isaac
```

---

## 5. Refining the 3D model and the greenhouse

**The robot mesh.** The shipped `models/robot/gardener_bdx.xml` (MuJoCo) and
`.urdf` (Isaac) use primitive shapes with **placeholder inertials** — fine to
train against, but tune to your CAD before serious Sim2Real:

- Replace `<geom>`/`<visual>`/`<collision>` primitives with your meshes (`.obj`/
  `.stl`); set real `mass`/`inertia` from CAD. Keep joint **names/origins/axes**
  consistent across MJCF, URDF, and the config.
- Regenerate the Isaac asset: `python scripts/convert_to_usd.py`. Open the result
  in Isaac Sim to check joints/collisions, then save the curated USD.

**The greenhouse scene.** Today the scene is generated procedurally:

- *Kinematic & MuJoCo*: plants/dock/human are laid out in code
  (`sim/kinematic_backend.py` / `sim/mujoco_backend.py`) from
  `configs/greenhouse/scene.yaml` (arena size, plant count, dryness ranges, dock
  position, dry-out rate, human speed). Edit that YAML to change the layout/task
  difficulty; edit the backends to change geometry.
- *Isaac (photoreal)*: author `models/scenes/greenhouse.usd` — real benches, pots,
  foliage, lighting — and it's referenced automatically. An RTX camera +
  RTX-LiDAR attach to the head by default (`_attach_sensors`; check the console
  on first run — see docs/ISAACSIM.md). For perception training, tag
  plants/people with semantic labels (Replicator) so the privileged
  `semantics()` channel and the real `PlantDetector` produce the same
  `Detection`s.

**Where "dryness" comes from.** It's a per-plant soil-moisture state the sim
evolves (waters down on dispense, dries over time). On hardware it's regressed
from the camera by the dryness head in `perception/plant_detector.py` — train
that head on labeled greenhouse images.

---

## 6. Visualizing & working with it — which tool when

- **Isaac Sim GUI** (`python scripts/run_isaac.py`) — the **interactive** photoreal
  twin. A window opens with the gardener running live; orbit/pause, drag plants or
  the person, change lighting — the loop reads your edits each tick (interactive
  domain randomization + safety testing). Full guide: [ISAACSIM.md](ISAACSIM.md).
  Use it for anything vision/VLA and for inspecting the imported robot. Heaviest,
  most realistic.
- **MuJoCo viewer** (`scripts/view_mujoco.py --view`) — the gait lab. Fast,
  contact-accurate, perfect for debugging balance/locomotion and the exported
  policy. No photorealism.
- **Kinematic top-down** (`scripts/view_kinematic.py`) — the behavior lab.
  Instant, no GPU, shows navigation/watering/docking/safety logic. Use it to
  iterate task logic and to make shareable GIFs.

Rule of thumb: **prototype task logic kinematically → train/​debug the gait in
MuJoCo → validate vision + the full loop in Isaac → port to the Jetson.**

---

## 7. Editing & adding tests

Tests live in `tests/` and run with plain `pytest` (a root `conftest.py` puts
`src/` on the path; no install needed). They're fast and dependency-light (torch
paths are skipped when torch is absent).

- `test_safety.py` — the Guardian's guarantees. **Add a case here whenever you add
  a safety rule** (e.g., a new interlock): construct a `RobotState`/`Action`, call
  `guardian.check(...)`, assert the verdict. This is the highest-value place to
  test.
- `test_rollout.py` — end-to-end kinematic episodes (waters plants, docks on low
  battery, stays within joint limits). Add an assertion here for any new
  end-to-end behavior.
- `test_vla.py` / `test_training.py` — VLA behavior, the LeRobot export layout,
  and the eval harness.

Two patterns you'll use a lot:

```python
# 1) a new behavioral guarantee — assert over a rollout
for _ in range(2000):
    info = ctrl.step(io)
assert info.verdict.level.name != "ESTOP"            # e.g. "never tips on flat ground"

# 2) task-success as a numeric gate (great for CI once the gait is trained)
from gardener_bdx.training.evaluate import run_episode, aggregate
res = [run_episode("kinematic", "neural", goal, 4000, s, {"checkpoint": ck, "device": "cuda"}, dt, rc, h)
       for s in range(10)]
assert aggregate(res, dt)["success_rate"] >= 0.8      # learned VLA must clear the bar
```

Run a subset while iterating: `pytest tests/test_safety.py -q`.

---

## 8. RTX 4070 (12 GB) specifics & limits

- **Locomotion RL**: comfortable at `--num_envs 4096` (no cameras). Drop to 2048
  if you OOM; raise toward 8192 if you have headroom.
- **NeuralVLA**: the 6.2M-param temporal model trains/infers easily on the 4070;
  use `--device cuda`. Increase `train_vla --batch` until ~10 GB used.
- **Photoreal Isaac envs** (with cameras) are the VRAM hog — use *tens*, not
  thousands, of camera envs for VLA-in-the-loop RL.
- **GR00T N1 (2-3B)**: full fine-tuning won't fit in 12 GB. Options: LoRA
  fine-tune (fits), inference-only in fp16/8-bit (tight but doable), or fine-tune
  in the cloud and run inference locally. For this GPU, treat **NeuralVLA as the
  primary brain** and GR00T as a stretch/offload path (see [GROOT.md](GROOT.md)).

---

### TL;DR first session

```bash
python scripts/dev.py preflight     # ready?
python scripts/dev.py gif           # see the task logic (no GPU needed)
python scripts/dev.py usd           # robot -> USD
python scripts/dev.py walk-smoke    # 2-min pipeline validation
python scripts/dev.py walk          # train the gait (~15-40 min on a 4070)
python scripts/dev.py watch         # watch it walk in MuJoCo
python scripts/dev.py isaac         # the interactive photoreal twin
python scripts/dev.py demos && python scripts/dev.py vla && python scripts/dev.py eval
```
