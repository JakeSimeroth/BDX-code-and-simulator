# Simulation & the digital twin — recommendation

**Recommendation: NVIDIA Isaac Sim + Isaac Lab as the primary digital twin and
training platform, with MuJoCo/MJX as the fast companion for locomotion RL.**
A bundled pure-numpy kinematic twin covers CI and quick high-level iteration.

## Why this split

The robot is **vision-first** (a VLA reasoning over camera + LiDAR) and
**bipedal** (a contact-rich gait that must transfer to hardware). Those two
demands pull on a simulator differently, so we use the best tool for each and
hide both behind `RobotIO`.

### Isaac Sim / Isaac Lab — the twin where the VLA is trained
- **Photorealistic RTX sensors.** The VLA consumes pixels; to cross the reality
  gap it must train on images that look like the real greenhouse — correct
  lighting, materials, translucent leaves, wet soil. Isaac's RTX camera and
  RTX-LiDAR + Replicator domain randomization deliver that. A geometric sim
  (MuJoCo) cannot.
- **Massive parallelism (Isaac Lab).** Thousands of cloned greenhouses on a
  single GPU. This is what makes RL training of the locomotion substrate — and
  RL-finetuning of the VLA against the gardening task reward — tractable in days
  instead of months.
- **One ecosystem, robot included.** USD scene/robot assets, the Jetson deploy
  target, **GR00T N1** (our VLA backbone) and **Cosmos** world-model tooling all
  live in the NVIDIA stack. The twin and the robot share a pipeline; nothing is
  re-implemented at the boundary.
- **Documented Sim2Real workflow.** Isaac Lab ships sensor models, actuator
  models, and randomization specifically for sim-to-real, including the GR00T
  N1 sim-to-real recipe.

### MuJoCo / MJX — the fastest path to a *walking* policy
- The Disney BDX droid and the open-source **Open Duck Mini** both learned to
  walk with MuJoCo-class RL and a Disney **imitation reward**. We reuse that
  lineage: `training/locomotion_env.py` + `training/rewards.py` implement exactly
  that recipe, and `models/robot/gardener_bdx.xml` is a ready MJCF.
- MJX (JAX) gives enormous step throughput for the *low-dimensional* locomotion
  problem, where photorealism is irrelevant. It is also CPU-friendly for CI.

### Bundled kinematic twin — runs anywhere
`sim/kinematic_backend.py` is pure numpy: no contact physics, but a complete
greenhouse (plants with soil-moisture, a dock, a moving human, ray-cast LiDAR).
It closes the **gardener** loop today and runs the test suite and the
quickstart. Use it to shape high-level task logic before paying for GPU physics.

## How a backend swap works

Every backend implements the same four-method `RobotIO` (`reset/read/write/step`)
plus an optional privileged `semantics()` channel. The runner and policies never
know which one they're driving:

```python
io = make_backend("isaac",     rc, dt)   # photoreal twin / VLA training & eval
io = make_backend("mujoco",    rc, dt)   # full-physics gait RL
io = make_backend("kinematic", rc, dt)   # runs now, no deps
# ...later...
io = JetsonBackend(rc, dt, *drivers)     # the real robot — same control graph
```

## Setup

- **Kinematic:** `pip install -e .` — done.
- **MuJoCo:** `pip install -e '.[sim]'` then `--backend mujoco`.
- **Isaac Sim / Isaac Lab:** install via NVIDIA Omniverse + the Isaac Lab guide,
  then run under Isaac's Python. Author `models/robot/gardener_bdx.usd` (convert
  from the MJCF/URDF) and `models/scenes/greenhouse.usd`; wire the RTX
  camera/LiDAR prim handles in `sim/isaac_backend.py` where marked. Parallel RL
  envs use Isaac Lab's `DirectRLEnv` mirroring `training/locomotion_env.py`.

## Alternatives considered

- **Gazebo / classic ROS** — rejected: this is an end-to-end learning system, not
  a behavior-tree stack, and Gazebo's rendering/throughput don't serve VLA
  training. (We can still expose a ROS 2 bridge for tooling/telemetry if desired.)
- **Genesis / PyBullet** — fine for quick experiments; they don't match Isaac's
  photoreal sensors + GR00T integration or MuJoCo's gait-RL maturity.
