"""Jetson on-robot backend.

This is the Sim2Real payoff: because perception, the VLA, the locomotion
substrate and the Guardian only ever touch :class:`RobotIO`, deploying to the
physical BDX is a matter of implementing the *drivers* below — not touching a
line of policy code. Swap ``KinematicGreenhouse``/``MujocoGreenhouse`` for
``JetsonBackend`` and the same control graph runs on the robot.

The driver seams (abstract) correspond to the BDX hardware (see docs/HARDWARE.md):
  * ``ActuatorBus``  — smart servos (e.g. Feetech STS / Dynamixel) over TTL/CAN.
  * ``ImuDriver``    — 9-DOF IMU (e.g. BNO085) at >=200 Hz.
  * ``DepthCamera``  — RGB-D (e.g. RealSense D435i) on the head.
  * ``LidarDriver``  — 3D/2D LiDAR (e.g. Unitree L1 / RPLiDAR) on the head.
  * ``BatteryMonitor``— smart battery / INA226 fuel gauge.
  * ``WaterSystem``  — tank level sensor + metering pump/valve (PWM/GPIO).
  * ``SlamEstimator``— LiDAR+camera+IMU fusion -> base pose at the control rate.

Each defaults to raising until wired, so a missing driver fails loudly rather
than silently feeding the policy zeros."""

from __future__ import annotations

import abc
import time
from typing import Optional

import numpy as np

from ..common.config import RobotConfig
from ..common.types import (
    Action,
    BatteryState,
    CameraFrame,
    ImuReading,
    JointState,
    LidarScan,
    Observation,
    Pose,
    Twist,
    WaterTankState,
)
from ..interfaces.robot_io import RobotIO


# --------------------------------------------------------------------------- #
# Driver interfaces (implement these for your hardware)
# --------------------------------------------------------------------------- #
class ActuatorBus(abc.ABC):
    @abc.abstractmethod
    def write_position_targets(self, q: np.ndarray, qd: Optional[np.ndarray], tau_ff: Optional[np.ndarray]) -> None: ...
    @abc.abstractmethod
    def read_joint_state(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:  # q, qd, tau
        ...
    def set_gains(self, kp: np.ndarray, kd: np.ndarray) -> None: ...
    def torque_off(self) -> None: ...


class ImuDriver(abc.ABC):
    @abc.abstractmethod
    def read(self) -> ImuReading: ...


class DepthCamera(abc.ABC):
    @abc.abstractmethod
    def read(self) -> Optional[CameraFrame]: ...


class LidarDriver(abc.ABC):
    @abc.abstractmethod
    def read(self) -> Optional[LidarScan]: ...


class BatteryMonitor(abc.ABC):
    @abc.abstractmethod
    def read(self) -> BatteryState: ...


class WaterSystem(abc.ABC):
    @abc.abstractmethod
    def set_flow(self, lps: float) -> None: ...
    @abc.abstractmethod
    def read(self) -> WaterTankState: ...


class SlamEstimator(abc.ABC):
    """Fuses LiDAR + camera + IMU into a base pose. On hardware this fills the
    ``base_pose`` slot that simulators get from ground truth."""

    @abc.abstractmethod
    def update(self, lidar: Optional[LidarScan], camera: Optional[CameraFrame], imu: ImuReading) -> tuple[Pose, Twist]: ...


# --------------------------------------------------------------------------- #
# The backend
# --------------------------------------------------------------------------- #
class JetsonBackend(RobotIO):
    def __init__(
        self,
        robot_config: RobotConfig,
        control_dt: float,
        actuators: ActuatorBus,
        imu: ImuDriver,
        battery: BatteryMonitor,
        water: WaterSystem,
        camera: Optional[DepthCamera] = None,
        lidar: Optional[LidarDriver] = None,
        slam: Optional[SlamEstimator] = None,
    ):
        super().__init__(robot_config, control_dt)
        self.rc = robot_config
        self.actuators = actuators
        self.imu = imu
        self.battery = battery
        self.water = water
        self.camera = camera
        self.lidar = lidar
        self.slam = slam
        self.actuators.set_gains(robot_config.kp, robot_config.kd)
        self._t0 = time.monotonic()
        self._next_tick = self._t0

    def is_simulation(self) -> bool:
        return False

    def reset(self) -> Observation:
        # Physical reset = ease to the default stance under reduced gains, then
        # restore. Kept explicit and slow for safety; tune per hardware.
        self.actuators.write_position_targets(self.rc.default_joint_positions, None, None)
        time.sleep(0.5)
        self._t0 = time.monotonic()
        self._next_tick = self._t0
        return self.read()

    def read(self) -> Observation:
        t = self.now()
        q, qd, tau = self.actuators.read_joint_state()
        imu = self.imu.read()
        cam = self.camera.read() if self.camera else None
        lid = self.lidar.read() if self.lidar else None
        base_pose = base_twist = None
        if self.slam is not None:
            base_pose, base_twist = self.slam.update(lid, cam, imu)
        return Observation(
            stamp=t,
            imu=imu,
            joints=JointState(q, qd, tau, names=self.rc.joint_names, stamp=t),
            camera=cam,
            lidar=lid,
            battery=self.battery.read(),
            water=self.water.read(),
            base_pose=base_pose,
            base_twist=base_twist,
        )

    def write(self, action: Action) -> None:
        self.actuators.write_position_targets(
            np.clip(action.joint_position_targets, self.rc.joint_lower, self.rc.joint_upper),
            action.joint_velocity_targets,
            action.joint_torque_ff,
        )
        self.water.set_flow(max(0.0, action.water_valve_lps))

    def step(self) -> None:
        # Pace the loop to the control period (real time advances on its own).
        self._next_tick += self.dt
        sleep = self._next_tick - time.monotonic()
        if sleep > 0:
            time.sleep(sleep)
        else:
            # Overran the budget — resync so we don't spiral.
            self._next_tick = time.monotonic()

    def now(self) -> float:
        return time.monotonic() - self._t0

    def emergency_stop(self) -> None:
        """Cut motor torque immediately (hardware e-stop path)."""
        self.water.set_flow(0.0)
        self.actuators.torque_off()

    def close(self) -> None:
        self.emergency_stop()
