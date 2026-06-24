# GardenerBDX — a VLA-driven bipedal greenhouse gardener

An end-to-end, learning-first control stack for a **BDX-inspired bipedal droid**
that tends a greenhouse: it walks fluidly, maps its surroundings with LiDAR +
camera, finds plants that need water, carries and delivers water, works safely
around people, and docks itself to recharge and refill.

This is **not** a classical ROS behavior tree. The brain is a single
**Vision-Language-Action (VLA) world model** — raw sensors + a natural-language
goal in, whole-body intent out — trained and proven in a **digital twin** and
then ported to hardware (Sim2Real). NVIDIA is the base platform end-to-end:
**Isaac Sim / Isaac Lab** for the twin, **GR00T N1**-class VLA for the brain,
**Jetson** for deployment, and a **Halos**-aligned safety layer.

> **Status:** the full architecture is implemented and runs end-to-end *today*
> in a bundled pure-numpy twin (no GPU/ML deps needed). The MuJoCo and Isaac
> backends, the RL/VLA trainers, and the Jetson driver layer are wired against
> the same interfaces, ready to scale up.

```
                       "water the thirsty plants"   (language goal)
                                   │
   RGB-D ─┐                ┌───────▼─────────────────────────┐
   LiDAR ─┼──────────────► │  SYSTEM 2  — VLA reasoning (VLM) │  ~1–10 Hz
   proprio┘                │  scene + goal → latent + skill   │
                           └───────┬─────────────────────────┘
                                   │ latent
                           ┌───────▼─────────────────────────┐
                           │  SYSTEM 1  — flow-matching action│  every tick
                           │  head → whole-body velocity cmd  │
                           └───────┬─────────────────────────┘
                                   │ velocity command (vx,vy,wz,…)
                           ┌───────▼─────────────────────────┐
                           │  SYSTEM 0  — locomotion substrate│  50–200 Hz
                           │  RL gait policy → joint targets  │  (the BDX walk)
                           └───────┬─────────────────────────┘
                                   │ joint targets + water valve
                           ┌───────▼─────────────────────────┐
                           │  SAFETY GUARDIAN (deterministic) │  motor rate
                           │  limits · tip-over · human-stop  │  (Halos-aligned)
                           └───────┬─────────────────────────┘
                                   ▼
                      RobotIO  →  { Isaac | MuJoCo | kinematic | Jetson }
```

The four tiers run on **decoupled clocks**, which is what keeps motion fluid even
when the VLA reasons slowly — and is why the hardware port is a question of
*compute budget*, not architecture. Everything talks to one seam, `RobotIO`, so
the **same control code** drives the simulator and the real robot.

---

## The simulator recommendation

**Primary digital twin & training: [NVIDIA Isaac Sim + Isaac Lab].**
**Companion for fast gait RL: [MuJoCo / MJX].** Full reasoning in
[`docs/SIMULATION.md`](docs/SIMULATION.md); the short version:

| Need | Why Isaac Sim / Isaac Lab | Why also MuJoCo |
|---|---|---|
| **Vision-first VLA** | Photoreal RTX camera + RTX-LiDAR with domain randomization — the VLA learns from images that look like the real greenhouse. | — |
| **Massive RL throughput** | Thousands of cloned greenhouses on one GPU (Isaac Lab) to train locomotion and RL-finetune the VLA. | MJX is the fastest path to a first *walking* policy; the Disney BDX / Open Duck Mini lineage. |
| **Same stack as the robot** | USD assets, Jetson deploy target, and **GR00T N1 / Cosmos** tooling live in one ecosystem. | Lightweight, CPU-friendly, great for gait iteration & CI. |
| **Sim2Real** | Built-in randomization, sensor models, and a documented GR00T sim-to-real workflow. | Trivial to reproduce dynamics for quick transfer checks. |

We support **both** behind `RobotIO`, plus a third **pure-numpy kinematic twin**
that needs zero heavy dependencies and is what runs the quickstart below and the
test suite.

---

> **Deploying on a GPU box?** Start with **[`docs/GETTING_STARTED_GPU.md`](docs/GETTING_STARTED_GPU.md)**
> (RTX 4070 runbook): first boot, `scripts/preflight.py`, training, and how to
> refine the robot/greenhouse and edit the tests.

## Quickstart (runs right now, no GPU)

```bash
pip install -e .          # just numpy + pyyaml for the core
python -m gardener_bdx.runtime.sim_main --backend kinematic --steps 3000 \
    --goal "tend the garden and water the thirsty plants"
```

You'll watch the droid sweep the greenhouse, discover all the plants, navigate →
approach → dispense water on the thirsty ones, slow/stop for the person pacing
the aisle, and manage its battery/water — with the Guardian screening every
command. Run the suite with `pytest`.

Scale up (needs extras, see below):

```bash
# 0) on a GPU box: one-command Omniverse bring-up (install into Isaac,
#    URDF->USD, smoke-train). See docs/ISAACLAB.md.
ISAACLAB_PATH=/opt/IsaacLab ./scripts/setup_omniverse.sh

# 1a) train the fluid walk — quick MuJoCo/SB3 path (CPU-friendly)
python -m gardener_bdx.training.train_locomotion --backend mujoco --num-envs 8
# 1b) ...or at scale in Isaac Lab (thousands of envs on GPU, full DR), same export
python -m gardener_bdx.training.train_isaaclab --num_envs 4096 --headless

# 2) generate expert demos, then distill the end-to-end (temporal) VLA from pixels
python -m gardener_bdx.training.collect_demos --backend mujoco --render --episodes 200
python -m gardener_bdx.training.train_vla --epochs 30

# 3) run the full physics twin with the neural VLA, and score it
python -m gardener_bdx.runtime.sim_main --backend mujoco --vla neural \
    --vla-ckpt models/policies/vla_bc.pt --render
python -m gardener_bdx.training.evaluate --compare   # scripted vs neural: success/collisions/time

# 4) ...or drive it with a fine-tuned GR00T N1 (see docs/GROOT.md)
python -m gardener_bdx.training.export_lerobot --data data/expert_demos.npz --out data/gardener_lerobot
python -m gardener_bdx.runtime.sim_main --backend mujoco --vla groot --vla-ckpt <ckpt> --render
```

Install extras as needed: `pip install -e '.[sim]'` (MuJoCo), `'.[train]'`
(PPO), `'.[vla]'` (torch), `'.[data]'` (LeRobot export). Isaac Sim/Isaac Lab
install via NVIDIA Omniverse (docs/ISAACLAB.md); Isaac-GR00T from its NVIDIA repo
(docs/GROOT.md).

---

## Repository map

```
configs/                 robot, control rates, safety, domain-randomization, scene, groot modality
models/robot/            MuJoCo MJCF + URDF of the BDX biped (URDF→USD for Isaac)
scripts/                 preflight · view_kinematic · view_mujoco · convert_to_usd · setup_omniverse · train
src/gardener_bdx/
  common/                types (the sim↔real contract), quaternion math, config
  interfaces/robot_io.py THE hardware-abstraction seam
  perception/            LiDAR+camera → occupancy + plant/human/dock world belief
  policy/
    vla_brain.py         ★ the VLA: NeuralVLA (end-to-end) + ScriptedGardenerVLA (expert/teacher)
    vla_net.py           dual-system torch net: VLM reasoning + flow-matching action head
    groot_vla.py         GR00T N1 backbone adapter (new-embodiment, action chunking)
    locomotion.py        System 0 RL gait policy (numpy inference + CPG fallback)
    runner.py            the multi-rate control graph
  safety/guardian.py     deterministic, Halos-aligned safety monitor
  sim/                   kinematic (runs now) · mujoco · isaac backends
  hardware/jetson_backend.py   real-robot RobotIO + driver seams
  training/              numpy + Isaac Lab RL envs (event-based DR); rewards; demo
                         collection; VLA distillation; LeRobot export; evaluate
  runtime/               sim_main · robot_main · the loop
tests/                   math · safety · locomotion · rollout · VLA · training glue
docs/                    ARCHITECTURE · SIMULATION · SIM2REAL · SAFETY · HARDWARE · ISAACLAB · GROOT
```

## Documentation

- [`docs/GETTING_STARTED_GPU.md`](docs/GETTING_STARTED_GPU.md) — **RTX 4070 runbook**: deploy, train, refine, test.
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — the four tiers in depth, data flow, why hierarchy.
- [`docs/SIMULATION.md`](docs/SIMULATION.md) — the digital-twin recommendation and setup.
- [`docs/ISAACLAB.md`](docs/ISAACLAB.md) — parallel locomotion RL training + numpy export.
- [`docs/GROOT.md`](docs/GROOT.md) — GR00T N1 fine-tuning (LeRobot export) + deploy.
- [`docs/SIM2REAL.md`](docs/SIM2REAL.md) — domain randomization and the transfer plan.
- [`docs/SAFETY.md`](docs/SAFETY.md) — the Guardian and Halos alignment.
- [`docs/HARDWARE.md`](docs/HARDWARE.md) — BOM, sensors, actuators, the Jetson bring-up.

## Grounding / prior art

- NVIDIA **Isaac GR00T N1** — open dual-system VLA foundation model for humanoids.
- NVIDIA **Halos for Robotics** — full-stack safety system for physical AI near people.
- **Open Duck Mini** (apirrone) — open-source mini BDX, MuJoCo/Isaac, Disney imitation-reward Sim2Real.
- **Jetson Orin Nano Super** — the on-robot compute target (hardware-porting phase).

[NVIDIA Isaac Sim + Isaac Lab]: https://developer.nvidia.com/isaac/sim
[MuJoCo / MJX]: https://mujoco.org
