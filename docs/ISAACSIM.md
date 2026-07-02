# Interactive Isaac Sim — driving the twin as a user

Isaac Sim is the 3D app where you **watch and interact with** the gardener in a
photoreal greenhouse. It's the same USD assets and the same `RobotIO`/controller
as everything else — only now there's a window, and you're in it.

```
Isaac Lab   →  headless, thousands of envs   →  TRAIN  (training/isaaclab_*)
Isaac Sim   →  one env, GUI, you interacting  →  TEST   (scripts/run_isaac.py)
                         ▲ same robot USD, same greenhouse USD, same control loop
```

## Run it

```bash
# once: turn the robot into a USD Isaac can spawn
python scripts/convert_to_usd.py                 # models/robot/gardener_bdx.urdf -> .usd

# open the interactive window and run the gardener live
python scripts/run_isaac.py
# with a trained gait + brain:
python scripts/run_isaac.py --policy models/policies/locomotion.npz \
    --vla neural --vla-ckpt models/policies/vla_bc.pt
# no window (eval/headless):
python scripts/run_isaac.py --headless
```

A window opens with the droid in the greenhouse, the control loop stepping at
50 Hz. If you haven't authored a greenhouse yet, a **procedural** one (two plant
benches, a charging dock, a roaming person) is spawned automatically so there's
something to do on day one.

## What you can do in the window

Because the loop reads live stage state **every tick**, your interactions feed
straight into perception → the VLA → the safety Guardian:

- **Orbit / zoom / pan** the camera; switch to the robot's head camera viewport.
- **Pause and single-step** (spacebar / step button) to inspect a moment.
- **Drag a plant** somewhere new → it shows up in the next `semantics()` /
  perception update and the gardener re-plans toward it.
- **Drag the person** toward the robot → watch the Guardian's human-proximity
  freeze trigger (the droid stops and won't dispense).
- **Change lighting / materials / add props** → exactly the visual variation the
  vision VLA must be robust to (this is interactive domain randomization).
- **Inspect** joints, contacts, and physics in the property panels.

## Photoreal vision (RTX camera + LiDAR)

An RGB-D camera and an RTX-LiDAR are **attached to `head_link` by default**
(`IsaacGreenhouse._attach_sensors`) — the head is the BDX sensor gimbal, so
`look_yaw`/`look_pitch` (including the expression layer's glances) literally aim
them. When active, `Observation.camera/lidar` carry real renders: the VLA runs
on true pixels, and `collect_demos --backend isaac` records RGB + depth + a
LiDAR-BEV that `train_vla` consumes directly.

Attachment is **best-effort and version-sensitive** (written for Isaac Sim 4.5+
`isaacsim.sensors.*`, falling back to `omni.isaac.sensor`). On your first run
watch the console: a `[isaac] ... not attached` line means your build's sensor
API differs — finish the seam in `_attach_sensors`/`_read_camera`/`_read_lidar`
(the accessors to check are noted inline). Until then the twin runs on the
privileged `semantics()` channel, so the task still closes either way. To skip
sensors entirely: `python scripts/run_isaac.py --no-rtx-sensors`.

## Authoring a real greenhouse

The procedural scene is a placeholder. For a real twin, build
`models/scenes/greenhouse.usd` (benches, pots, foliage, soil, lighting) in Isaac
Sim and it's referenced automatically (the backend prefers a USD scene over the
procedural fallback). Tag plants/people with **semantic labels** (Replicator) so
the ground-truth `semantics()` and the trained `PlantDetector` emit identical
`Detection`s. Keep plant/dock world positions in sync with
`configs/greenhouse/scene.yaml` if you want the metrics/eval to line up.

## Notes & troubleshooting

- **GPU only.** Needs the Isaac Sim runtime; run under Isaac's python
  (`~/IsaacLab/isaaclab.sh -p scripts/run_isaac.py ...` works too).
- **`gardener_bdx.usd not found`** → run `scripts/convert_to_usd.py` first.
- **Import errors on `omni.isaac.core`** → Isaac Sim 4.5+ renamed it to
  `isaacsim.core`; `sim/isaac_backend.py` tries the new namespace then falls back.
  If your version differs, adjust the imports at the top of `_build_stage`.
- **It falls over immediately** → you're on the CPG fallback gait. Train a policy
  (`training/train_isaaclab.py`) and pass `--policy models/policies/locomotion.npz`.
