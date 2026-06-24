# Hardware — the physical BDX gardener

A reference bill of materials and the bring-up path. The software is already
written against these via `hardware/jetson_backend.py`; building the robot is a
matter of implementing the driver seams.

## Reference BOM

| Subsystem | Reference part | Notes |
|---|---|---|
| **Compute** | NVIDIA **Jetson Orin Nano Super** (8 GB, 67 TOPS) | deploy target; hardware-porting phase quantizes the VLA to fit |
| **Actuators** | 12× smart servos (Feetech STS3215 / Dynamixel-class) | 5/leg (hip yaw·roll·pitch, knee, ankle) + neck yaw + head pitch; TTL/RS-485 or CAN bus |
| **IMU** | BNO085 (or ICM-45686) | ≥200 Hz orientation/gyro/accel into `ImuDriver` |
| **Depth camera** | Intel RealSense D435i (head) | RGB-D for the VLA + plant dryness head |
| **LiDAR** | Unitree L1 / RPLiDAR / Livox Mid-360 | mapping + obstacle/human geofence |
| **Battery** | 4S Li-ion + smart fuel gauge (INA226/BQ) | `BatteryMonitor`; sized for the duty cycle |
| **Water** | ~1.5 L tank + metering pump/solenoid + level sensor | `WaterSystem`, PWM/GPIO; nozzle on the head/snout |
| **Dock** | charging contacts + IR/AprilTag beacon | self-align + refill; a fixed, known map fixture |

Lineage: the **Open Duck Mini** (open-source mini BDX, ~$400 BOM, MuJoCo/Isaac
Sim2Real with the Disney imitation reward) is the closest reference build and a
good starting chassis to scale from.

## Joint map (canonical order)

Defined once in `configs/robot/gardener_bdx.yaml` and mirrored by the MJCF, the
URDF/USD, and every backend:

```
0 left_hip_yaw   1 left_hip_roll   2 left_hip_pitch   3 left_knee   4 left_ankle_pitch
5 right_hip_yaw  6 right_hip_roll  7 right_hip_pitch  8 right_knee  9 right_ankle_pitch
10 neck_yaw      11 head_pitch
```

## Bring-up checklist

1. **Driver layer.** Implement the seams in `hardware/jetson_backend.py`:
   `ActuatorBus`, `ImuDriver`, `DepthCamera`, `LidarDriver`, `BatteryMonitor`,
   `WaterSystem`, `SlamEstimator`. Each defaults to raising until wired, so a
   missing driver fails loudly rather than feeding the policy zeros.
2. **Joint sign/zero calibration.** Match real encoder directions and zeros to
   the config's `default` stance; verify `kp/kd`, torque, and velocity limits.
3. **State estimation.** Stand up SLAM (LiDAR+camera+IMU) to fill the base-pose
   slot the simulators got from ground truth.
4. **Static safety test.** With legs off the ground, confirm every Guardian
   check fires on hardware (limits, e-stop on SIGINT, water interlocks).
5. **Gait transfer.** Deploy `locomotion.npz`; tune DR ranges until the twin gait
   transfers; iterate.
6. **VLA deploy.** Load the VLA checkpoint; TensorRT-optimize System 2; validate
   the full gardener loop at low speed near a person before autonomous operation.

## Run it

```bash
python -m gardener_bdx.runtime.robot_main --goal "tend the garden" \
    --vla neural --vla-ckpt models/policies/vla_bc.pt
# Ctrl-C trips the Guardian e-stop (cuts torque + water) before exit.
```

Same control graph as the twin — only the backend differs.
