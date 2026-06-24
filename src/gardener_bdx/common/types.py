"""Core data types — the shared contract between perception, policy, safety,
the simulator backends, and the real-robot backend.

Design rules:
  * Plain dataclasses + numpy arrays. No framework lock-in, trivially
    serializable, identical in sim and on hardware.
  * Quaternions are ``(w, x, y, z)`` to match Isaac/USD and MuJoCo conventions.
  * SI units everywhere: metres, radians, seconds, volts, amps, litres.
  * Timestamps are float seconds (monotonic clock domain).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np

# --------------------------------------------------------------------------- #
# Kinematics primitives
# --------------------------------------------------------------------------- #


@dataclass
class Pose:
    """Rigid-body pose in a named frame."""

    position: np.ndarray  # (3,) metres, [x, y, z]
    orientation: np.ndarray  # (4,) quaternion [w, x, y, z]
    frame: str = "world"

    @staticmethod
    def identity(frame: str = "world") -> "Pose":
        return Pose(np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]), frame)


@dataclass
class Twist:
    """Spatial velocity."""

    linear: np.ndarray  # (3,) m/s
    angular: np.ndarray  # (3,) rad/s
    frame: str = "base"


# --------------------------------------------------------------------------- #
# Sensors
# --------------------------------------------------------------------------- #


@dataclass
class ImuReading:
    orientation: np.ndarray  # (4,) quaternion [w, x, y, z]
    angular_velocity: np.ndarray  # (3,) rad/s, body frame
    linear_acceleration: np.ndarray  # (3,) m/s^2, body frame (includes gravity)
    stamp: float = 0.0


@dataclass
class JointState:
    """Proprioception for the actuated joints, ordered per the robot config."""

    positions: np.ndarray  # (n_joints,) rad
    velocities: np.ndarray  # (n_joints,) rad/s
    torques: np.ndarray  # (n_joints,) N·m (estimated/commanded)
    names: tuple[str, ...] = ()
    stamp: float = 0.0


@dataclass
class CameraFrame:
    """A single RGB-D frame. ``depth`` may be None for RGB-only cameras."""

    rgb: np.ndarray  # (H, W, 3) uint8
    depth: Optional[np.ndarray] = None  # (H, W) float32 metres
    intrinsics: Optional[np.ndarray] = None  # (3, 3) pinhole K
    frame: str = "head_camera"
    stamp: float = 0.0


@dataclass
class LidarScan:
    """3D LiDAR return as a point cloud in the sensor frame."""

    points: np.ndarray  # (N, 3) metres, sensor frame
    intensities: Optional[np.ndarray] = None  # (N,) optional
    frame: str = "lidar"
    stamp: float = 0.0


@dataclass
class BatteryState:
    voltage: float = 0.0  # V
    current: float = 0.0  # A (positive = discharge)
    state_of_charge: float = 1.0  # [0, 1]
    charging: bool = False
    stamp: float = 0.0


@dataclass
class WaterTankState:
    """The payload the gardener carries and delivers."""

    level_liters: float = 0.0
    capacity_liters: float = 1.5
    dispensing: bool = False
    flow_rate_lps: float = 0.0  # litres/second while dispensing
    stamp: float = 0.0

    @property
    def fraction(self) -> float:
        return 0.0 if self.capacity_liters <= 0 else self.level_liters / self.capacity_liters


# --------------------------------------------------------------------------- #
# Observation bundle — the full multimodal input to perception + VLA
# --------------------------------------------------------------------------- #


@dataclass
class Observation:
    """Everything the robot senses at one tick. This is the raw input that the
    VLA world model consumes (end-to-end), and that perception refines."""

    stamp: float
    imu: ImuReading
    joints: JointState
    camera: Optional[CameraFrame] = None
    lidar: Optional[LidarScan] = None
    battery: BatteryState = field(default_factory=BatteryState)
    water: WaterTankState = field(default_factory=WaterTankState)
    audio: Optional[np.ndarray] = None  # (samples,) mono waveform, optional
    # Optional ground-truth base state — available in sim, used for training /
    # privileged critics and for evaluation. Never required at deploy time.
    base_pose: Optional[Pose] = None
    base_twist: Optional[Twist] = None


# --------------------------------------------------------------------------- #
# Commands & actions — the hierarchy's internal currency
# --------------------------------------------------------------------------- #


class Skill(Enum):
    """Discrete behavioral modes the VLA can select. The motor substrate and
    skill heads specialize on each; the Guardian gates transitions."""

    IDLE = "idle"
    NAVIGATE = "navigate"  # locomote toward a map goal
    APPROACH_PLANT = "approach_plant"  # fine positioning at a plant
    DISPENSE_WATER = "dispense_water"  # aim nozzle + meter water
    DOCK_CHARGE = "dock_charge"  # align to charging station
    RECOVER = "recover"  # get-up / stabilize after a disturbance


@dataclass
class LocomotionCommand:
    """Velocity-space command consumed by the locomotion substrate (System 0).
    This is the narrow, robust interface between high-level intent and the gait
    policy — the same convention used by the Disney BDX / Open Duck Mini work."""

    vx: float = 0.0  # m/s, forward
    vy: float = 0.0  # m/s, lateral
    wz: float = 0.0  # rad/s, yaw rate
    # Stylistic / posture modifiers the gait policy is conditioned on.
    body_height: float = 0.0  # delta from nominal stance height (m)
    look_yaw: float = 0.0  # decouple head/look direction (rad)
    look_pitch: float = 0.0  # rad


@dataclass
class Intent:
    """High-level intent emitted by the VLA brain (System 2/1). It carries both
    a *reactive* velocity command and *symbolic* context for the skill heads,
    safety, and logging."""

    skill: Skill
    locomotion: LocomotionCommand
    dispense_rate_lps: float = 0.0  # commanded water flow
    target_pose: Optional[Pose] = None  # world-frame goal if known
    confidence: float = 1.0
    rationale: str = ""  # human-readable explanation (great for debugging VLAs)


@dataclass
class Action:
    """The low-level command actually sent to the actuators every control tick.
    Position-control with feed-forward, matching typical smart servos / the
    Jetson actuator bus and the MuJoCo/Isaac actuators in sim."""

    joint_position_targets: np.ndarray  # (n_joints,) rad
    joint_velocity_targets: Optional[np.ndarray] = None  # (n_joints,) rad/s
    joint_torque_ff: Optional[np.ndarray] = None  # (n_joints,) N·m feed-forward
    water_valve_lps: float = 0.0  # commanded dispense flow

    def copy(self) -> "Action":
        return Action(
            joint_position_targets=np.array(self.joint_position_targets, copy=True),
            joint_velocity_targets=None
            if self.joint_velocity_targets is None
            else np.array(self.joint_velocity_targets, copy=True),
            joint_torque_ff=None
            if self.joint_torque_ff is None
            else np.array(self.joint_torque_ff, copy=True),
            water_valve_lps=self.water_valve_lps,
        )


# --------------------------------------------------------------------------- #
# State estimate & safety
# --------------------------------------------------------------------------- #


@dataclass
class RobotState:
    """Fused state estimate produced by perception/SLAM, consumed by the
    runner, skills, and Guardian."""

    stamp: float
    base_pose: Pose
    base_twist: Twist
    joints: JointState
    battery: BatteryState
    water: WaterTankState
    # Convenience signals derived once and reused.
    upright_cos: float = 1.0  # cos(angle) between body-up and world-up; 1 = upright
    contact_feet: tuple[bool, ...] = (False, False)


@dataclass
class Detection:
    """A single semantic detection in the world frame.

    In the digital twin these come from ground truth (privileged channel) so we
    can train and evaluate the gardener task without first solving perception.
    On hardware the same structure is produced by the neural detectors running
    on real camera/LiDAR data — so downstream code is identical."""

    kind: str  # "plant" | "human" | "dock" | "obstacle"
    id: int
    position: np.ndarray  # (3,) world frame, metres
    attributes: dict = field(default_factory=dict)  # e.g. {"dryness": 0.7}


@dataclass
class SceneSemantics:
    """Optional privileged semantics a *simulator* backend may expose. Returns
    None on hardware — perception must then earn these from raw sensors."""

    stamp: float
    detections: tuple[Detection, ...] = ()


class SafetyLevel(Enum):
    OK = 0
    CAUTION = 1  # soft-limit applied (e.g. slow near humans)
    OVERRIDE = 2  # learned action replaced by safe fallback
    ESTOP = 3  # motors damped/cut; requires reset


@dataclass
class SafetyVerdict:
    level: SafetyLevel
    action: Action  # the (possibly modified) action to actually execute
    reasons: tuple[str, ...] = ()
