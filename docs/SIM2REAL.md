# Sim2Real

The plan is "train big in the twin, ship small to the robot," with the reality
gap closed by randomization and a shared interface — never by rewriting code.

## What makes transfer work here

1. **One interface.** Policies touch only `RobotIO`. The Jetson backend
   implements it with real drivers; nothing in perception/policy/safety changes.
2. **Identical observation/action layout.** `locomotion_observation()` and the
   VLA tensor packer are the single source of truth, used in training and at
   deploy — so the policy sees bit-identical inputs in both.
3. **Numpy policy export.** The RL gait trains in torch/Isaac Lab and exports to
   a plain `.npz` (`MLP.save_npz`) the robot runs without torch. Train big, ship
   small.
4. **Domain randomization** (`training/domain_randomization.py`,
   `configs/sim/domain_randomization.yaml`): per-episode randomization of link
   masses, friction, actuator gains, actuation/observation latency, IMU
   bias/noise, and random base shoves. The policy learns robustness instead of
   memorizing one perfect simulator.
5. **Sensor & actuator models.** In Isaac, render the head camera and RTX-LiDAR
   with realistic noise; model the servo as a position source with the same
   `kp/kd`, torque limits, and latency as the hardware. The MJCF/USD actuators
   already mirror `configs/robot/gardener_bdx.yaml`.

## The training pipeline

```
                 ┌─────────────────────────────────────────────┐
   Stage A       │ Locomotion RL (MuJoCo/MJX → Isaac Lab)       │
   the walk      │  imitation reward + velocity tracking + DR   │
                 │  → export models/policies/locomotion.npz     │
                 └───────────────────────┬─────────────────────┘
                                         │ fluid gait substrate
                 ┌───────────────────────▼─────────────────────┐
   Stage B       │ Expert demos: ScriptedGardenerVLA in the twin│
   teach pixels  │  log (RGB-D, LiDAR, proprio, text) → (action,│
                 │  skill)  [collect_demos.py]                   │
                 └───────────────────────┬─────────────────────┘
                                         │ dataset
                 ┌───────────────────────▼─────────────────────┐
   Stage C       │ Distill NeuralVLA (BC + flow matching)       │
   end-to-end    │  [train_vla.py]  → models/policies/vla_bc.pt │
                 └───────────────────────┬─────────────────────┘
                                         │ pixels-policy
                 ┌───────────────────────▼─────────────────────┐
   Stage D       │ RL-finetune the VLA in Isaac Lab against the │
   harden        │  task reward (water delivered, safety, time) │
                 └───────────────────────┬─────────────────────┘
                                         │
   Stage E       │ Port to Jetson: swap backend → JetsonBackend │
   deploy        │  TensorRT-optimize; keep the Guardian inline │
```

The **scripted expert is the teacher**: it solves the task from privileged map
state, the neural VLA learns to match it **from pixels**, and RL then pushes the
neural policy past the teacher. This bootstraps data without teleoperation.

## GR00T N1 as the VLA backbone

`NeuralVLA(backbone="groot")` is the integration seam. GR00T N1 is an open
dual-system VLA foundation model (a VLM "System 2" + a flow/diffusion action
"System 1") with exactly the shape implemented in `vla_net.py`. To use it:

1. Load `Gr00tPolicy.from_pretrained(checkpoint)` in `WorldModelVLANet.load()`.
2. Map its action chunk onto our `ACTION_KEYS` (the embodiment adapter).
3. Fine-tune on the Stage-B/C greenhouse dataset; RL-finetune in Stage D.

Our own `WorldModelVLANet` is the lightweight, dependency-free stand-in that lets
the whole pipeline run and be tested before pulling in the full foundation model.

## Hardware-porting phase (explicitly deferred)

Per project direction, simulation performance of the **full** VLA comes first;
on-robot compute is a later concern. When we get there: quantize/distill the
System-2 VLM (INT8/FP8 via TensorRT-LLM), keep System-1 + the locomotion policy
at full rate, and run the Guardian at the motor rate. The architecture already
assumes a slow brain, so this is a budgeting exercise, not a redesign.
