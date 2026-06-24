"""The safety Guardian — a deterministic monitor that screens every actuator
command before it reaches the body.

Why it exists: an end-to-end neural policy is wonderful at being *capable* and
terrible at *guaranteeing* anything. Around people, water, and living plants we
need guarantees that do not depend on a network's good mood. So the learned
stack only ever *proposes* an :class:`Action`; this layer — plain, auditable,
no learning — decides what actually executes.

It is modeled on the layering NVIDIA describes for **Halos for Robotics**: a
standardized safety stack that sits between the AI compute and the actuators and
holds regardless of what the policy does. The checks here (joint/torque/rate
limits, tip-over arrest, human-proximity stop, geofence, tilt/empty water
interlocks, e-stop, safe-posture fallback) are the robot-specific instantiation
of that idea. It runs at the motor rate, every tick, in both sim and hardware.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..common.config import RobotConfig
from ..common.types import (
    Action,
    Observation,
    RobotState,
    SafetyLevel,
    SafetyVerdict,
)


@dataclass
class GuardianConfig:
    tip_caution_cos: float = 0.80  # start arresting below this uprightness
    tip_estop_cos: float = 0.35  # cut power / full protective stop below this
    human_stop_radius: float = 0.7  # m — freeze gait inside this
    water_min_upright_cos: float = 0.90  # don't pour while tilted (spill risk)
    geofence_min: tuple[float, float] = (-5.8, -5.8)  # world XY box
    geofence_max: tuple[float, float] = (5.8, 5.8)
    rate_limit_scale: float = 1.0  # multiplies per-joint velocity limit

    @staticmethod
    def from_yaml(path: str = "control/safety.yaml") -> "GuardianConfig":
        from ..common.config import load_yaml

        try:
            d = load_yaml(path)
        except FileNotFoundError:
            return GuardianConfig()
        return GuardianConfig(
            tip_caution_cos=float(d.get("tip_caution_cos", 0.80)),
            tip_estop_cos=float(d.get("tip_estop_cos", 0.35)),
            human_stop_radius=float(d.get("human_stop_radius", 0.7)),
            water_min_upright_cos=float(d.get("water_min_upright_cos", 0.90)),
            geofence_min=tuple(d.get("geofence_min", (-5.8, -5.8))),
            geofence_max=tuple(d.get("geofence_max", (5.8, 5.8))),
            rate_limit_scale=float(d.get("rate_limit_scale", 1.0)),
        )


class SafetyGuardian:
    def __init__(self, robot_config: RobotConfig, control_dt: float, cfg: Optional[GuardianConfig] = None):
        self.rc = robot_config
        self.dt = float(control_dt)
        self.cfg = cfg or GuardianConfig()
        self._estop_latched = False
        self._last_targets: Optional[np.ndarray] = None
        # The deterministic fallback: nominal stance (legs under the body).
        self._safe_posture = robot_config.default_joint_positions.copy()

    # -- external e-stop control ------------------------------------------ #
    def trip_estop(self) -> None:
        self._estop_latched = True

    def reset_estop(self) -> None:
        self._estop_latched = False

    def reset(self) -> None:
        self._last_targets = None

    # -- the check -------------------------------------------------------- #
    def check(
        self,
        action: Action,
        state: RobotState,
        world=None,
        obs: Optional[Observation] = None,
    ) -> SafetyVerdict:
        a = action.copy()
        q = state.joints.positions
        reasons: list[str] = []
        level = SafetyLevel.OK

        def escalate(new: SafetyLevel, why: str):
            nonlocal level
            level = max(level, new, key=lambda s: s.value)
            reasons.append(why)

        # 1) Latched e-stop: damp to a hold, kill water. Requires manual reset.
        if self._estop_latched:
            a.joint_position_targets = q.copy()
            a.joint_torque_ff = None
            a.water_valve_lps = 0.0
            return SafetyVerdict(SafetyLevel.ESTOP, self._rate_limit(a, q), ("e-stop latched",))

        # 2) Tip-over arrest. Severe tilt => protective posture + cut water.
        if state.upright_cos < self.cfg.tip_estop_cos:
            a.joint_position_targets = self._safe_posture.copy()
            a.water_valve_lps = 0.0
            escalate(SafetyLevel.ESTOP, f"tip-over (cos={state.upright_cos:.2f})")
            return SafetyVerdict(level, self._rate_limit(a, q), tuple(reasons))
        elif state.upright_cos < self.cfg.tip_caution_cos:
            # Blend toward the safe stance proportionally to how far we've tilted.
            t = (self.cfg.tip_caution_cos - state.upright_cos) / (
                self.cfg.tip_caution_cos - self.cfg.tip_estop_cos
            )
            a.joint_position_targets = (1 - t) * a.joint_position_targets + t * self._safe_posture
            a.water_valve_lps = 0.0
            escalate(SafetyLevel.OVERRIDE, "tilt — blending to safe stance")

        # 3) Human proximity: freeze the gait (hold current posture), cut water.
        if world is not None and hasattr(world, "distance_to_nearest_human"):
            if world.distance_to_nearest_human() < self.cfg.human_stop_radius:
                a.joint_position_targets = q.copy()
                a.water_valve_lps = 0.0
                escalate(SafetyLevel.OVERRIDE, "human within stop radius — frozen")

        # 4) Geofence: if the base leaves the allowed box, hold and cut water.
        if world is not None and hasattr(world, "robot_xy"):
            xy = world.robot_xy()
            lo, hi = self.cfg.geofence_min, self.cfg.geofence_max
            if not (lo[0] <= xy[0] <= hi[0] and lo[1] <= xy[1] <= hi[1]):
                a.joint_position_targets = q.copy()
                a.water_valve_lps = 0.0
                escalate(SafetyLevel.OVERRIDE, "geofence breach — frozen")

        # 5) Water interlocks: never pour while tilted or with an empty tank.
        if a.water_valve_lps > 0.0:
            if state.water.level_liters <= 0.0:
                a.water_valve_lps = 0.0
                escalate(SafetyLevel.CAUTION, "tank empty — valve closed")
            elif state.upright_cos < self.cfg.water_min_upright_cos:
                a.water_valve_lps = 0.0
                escalate(SafetyLevel.CAUTION, "too tilted to dispense — valve closed")

        # 6) Hard joint position limits.
        clamped = np.clip(a.joint_position_targets, self.rc.joint_lower, self.rc.joint_upper)
        if not np.allclose(clamped, a.joint_position_targets):
            escalate(SafetyLevel.CAUTION, "joint limit clamp")
        a.joint_position_targets = clamped

        # 7) Torque feed-forward clamp.
        if a.joint_torque_ff is not None:
            tff = np.clip(a.joint_torque_ff, -self.rc.joint_torque_limit, self.rc.joint_torque_limit)
            if not np.allclose(tff, a.joint_torque_ff):
                escalate(SafetyLevel.CAUTION, "torque clamp")
            a.joint_torque_ff = tff

        # 8) Slew-rate limit (respect actuator velocity limits).
        a = self._rate_limit(a, q)
        return SafetyVerdict(level, a, tuple(reasons))

    def _rate_limit(self, action: Action, q: np.ndarray) -> Action:
        max_step = self.rc.joint_velocity_limit * self.dt * self.cfg.rate_limit_scale
        ref = self._last_targets if self._last_targets is not None else q
        delta = np.clip(action.joint_position_targets - ref, -max_step, max_step)
        action.joint_position_targets = ref + delta
        self._last_targets = action.joint_position_targets.copy()
        return action
