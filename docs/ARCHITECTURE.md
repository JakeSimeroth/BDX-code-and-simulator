# Architecture

One **VLA world model** is the brain. It expresses itself through a fast
locomotion substrate, and a deterministic Guardian screens everything. Four
tiers on decoupled clocks, one hardware seam.

## The seam: `RobotIO`

`interfaces/robot_io.py` is the only thing that differs between sim and robot.
It exposes `reset / read / write / step / now`, an optional privileged
`semantics()` (ground-truth detections in sim; `None` on hardware), and an
optional `set_base_command_hint()` used only by the reduced-order kinematic twin.
Perception, policy, and safety touch *only* this interface — so Sim2Real is a
backend swap, not a rewrite.

## Tier 2 — VLA reasoning (System 2)  ·  ~1–10 Hz
`policy/vla_brain.py` + `policy/vla_net.py`. A vision-language module encodes
RGB-D + LiDAR-BEV + proprioception + the language instruction and emits a
reasoning **latent** and a discrete **skill**. This is the deliberate step:
*which plant, is that a person, am I low on water, should I dock?* On the full
stack this is GR00T N1-class; the bundled `NeuralVLA` implements the same
dual-system shape and the `ScriptedGardenerVLA` provides a transparent expert.

## Tier 1 — Action head (System 1)  ·  every tick
A **flow-matching** action head (rectified flow) conditioned on the System-2
latent + fresh proprioception generates a continuous **whole-body velocity
command** `(vx, vy, wz, body_height, look_yaw, look_pitch, dispense)`. Fast and
reflexive, it keeps motion fluid even though System 2 thinks slowly. Mirrors
GR00T N1's diffusion/flow action expert.

## Tier 0 — Locomotion substrate  ·  50–200 Hz
`policy/locomotion.py`. A small RL policy maps proprioception + the velocity
command to **joint position targets** — the fluid BDX walk. Trained in physics
with a Disney-style imitation reward + domain randomization, then **exported to
numpy** so it runs on the robot with no torch. A CPG fallback makes the whole
stack animate before any policy is trained.

Why split Tier 0 from the VLA? Balance needs a fast, robust, narrow controller;
task reasoning needs a slow, broad one. Decoupling them is what makes the gait
transferable and the brain swappable — and it's how real humanoid stacks (GR00T,
Figure, Physical Intelligence) are organized.

## Safety Guardian  ·  motor rate
`safety/guardian.py`. Deterministic, non-learned. It screens every actuator
command: joint/torque/slew limits, tip-over arrest, human-proximity freeze,
geofence, water tilt/empty interlocks, latched e-stop, safe-posture fallback.
The learned stack only ever *proposes*; the Guardian *disposes*. See
[SAFETY.md](SAFETY.md).

## Perception → world belief
`perception/`. LiDAR + camera (+ privileged sim semantics or the real neural
detector) become an **occupancy map** plus tracked **plants** (with a dryness
estimate), **humans**, and the **dock** — the `WorldBelief`. The end-to-end VLA
still reads raw sensors; the belief grounds language goals, drives the scripted
expert, and backs the Guardian's geofence/human logic. On hardware the base pose
slot is filled by SLAM; the contract is identical.

## The control graph
`policy/runner.py::GardenerController.step()` is one locomotion-rate tick:

```
read() → world belief → state estimate
   ├─ (slow) VLA.act(obs, goal, world) → Intent     [re-planned every N ticks]
   ├─ (fast) locomotion.act(proprio, Intent.cmd) → Action
   ├─ Guardian.check(Action, state, world) → safe Action
   └─ write() → set_base_command_hint() → step()
```

Decoupled rates come from re-planning the VLA every `locomotion_hz / vla_hz`
ticks while the substrate and Guardian run every tick.

## Data types (`common/types.py`)
The vocabulary shared by everything: `Observation` (the multimodal input),
`Intent` (skill + velocity command + dispense + rationale), `Action` (joint
targets + water valve), `RobotState`, `SafetyVerdict`, and the sensor/`Detection`
structs. Plain dataclasses + numpy, identical in sim and on hardware.
