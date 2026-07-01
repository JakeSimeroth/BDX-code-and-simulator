"""The control graph. One :meth:`GardenerController.step` is a single
locomotion-rate tick that ties the whole stack together:

    read sensors ─▶ update world belief ─▶ estimate state
        │
        ├─ (slow, vla_hz)   VLA brain        : obs+goal+world ─▶ Intent
        ├─ (fast, every tick) locomotion sub : proprio+Intent ─▶ Action
        ├─ (every tick) safety Guardian      : Action ─▶ safe Action
        └─ write actuators ─▶ advance time

Because the brain runs on a slower clock than the body, motion stays fluid even
when reasoning is expensive — and the exact same object drives the kinematic
twin, MuJoCo, Isaac, or the Jetson. Only the injected ``RobotIO`` changes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..common.config import HierarchyConfig, RobotConfig
from ..common.math_utils import upright_cosine
from ..common.types import (
    Action,
    Intent,
    LocomotionCommand,
    Observation,
    RobotState,
    SafetyLevel,
    SafetyVerdict,
    Skill,
    Twist,
)
from ..interfaces.robot_io import RobotIO
from ..perception.world_model import GreenhouseMapper, WorldBelief
from ..safety.guardian import SafetyGuardian
from .animation import AnimationEngine, ExpressionOverlay
from .locomotion import LocomotionPolicy
from .task import TaskGoal
from .vla_brain import VLAPolicy, build_vla


@dataclass
class StepInfo:
    """Everything one tick produced — for logging, eval, and dataset capture."""

    stamp: float
    obs: Observation
    state: RobotState
    intent: Intent
    verdict: SafetyVerdict
    world: WorldBelief
    overlay: Optional[ExpressionOverlay] = None  # the animation actually rendered


class GardenerController:
    def __init__(
        self,
        robot_config: RobotConfig,
        hierarchy: Optional[HierarchyConfig] = None,
        goal: TaskGoal | str = "tend the garden",
        vla: Optional[VLAPolicy] = None,
        mapper: Optional[GreenhouseMapper] = None,
        guardian: Optional[SafetyGuardian] = None,
        locomotion: Optional[LocomotionPolicy] = None,
    ):
        self.rc = robot_config
        self.h = hierarchy or HierarchyConfig()
        self.dt = 1.0 / self.h.locomotion_hz
        self.goal = TaskGoal.parse(goal) if isinstance(goal, str) else goal

        self.vla = vla or build_vla("scripted")
        self.mapper = mapper or GreenhouseMapper()
        self.guardian = guardian or SafetyGuardian(robot_config, self.dt)
        self.locomotion = locomotion or LocomotionPolicy(
            robot_config, policy_path=self.h.locomotion_policy_path
        )
        self.animation = AnimationEngine()  # renders the VLA's expression channel

        self._brain_decim = max(1, round(self.h.locomotion_hz / max(self.h.vla_hz, 1e-6)))
        self.reset_state()

    # -- lifecycle -------------------------------------------------------- #
    def reset_state(self) -> None:
        self._tick = 0
        self._intent = Intent(Skill.IDLE, LocomotionCommand())
        self.vla.reset()
        self.locomotion.reset()
        self.guardian.reset()
        self.animation.reset()

    def reset(self, io: RobotIO) -> Observation:
        obs = io.reset()
        self.reset_state()
        self.mapper.update(obs, io.semantics())
        return obs

    def set_goal(self, goal: TaskGoal | str) -> None:
        self.goal = TaskGoal.parse(goal) if isinstance(goal, str) else goal

    # -- one control tick ------------------------------------------------- #
    def step(self, io: RobotIO) -> StepInfo:
        obs = io.read()
        world = self.mapper.update(obs, io.semantics())
        state = self._estimate_state(obs, world)

        # System 2/1 — re-plan on the slow clock; reuse the intent in between.
        if self._tick % self._brain_decim == 0:
            self._intent = self.vla.act(obs, self.goal, world)
            self.animation.request(self._intent.expression)

        # Expression layer — render the deployed animation as a bounded style
        # overlay and blend it into the command (body language, never balance).
        overlay = self.animation.update(self.dt)
        styled = self._apply_overlay(self._intent.locomotion, overlay)

        # System 0 — realize the velocity command as a gait, every tick.
        action = self.locomotion.act(obs, styled, self.dt)
        action.water_valve_lps = (
            self._intent.dispense_rate_lps if self._intent.skill == Skill.DISPENSE_WATER else 0.0
        )

        # Safety — screen the final actuator command.
        verdict = self.guardian.check(action, state, world, obs)
        if verdict.level.value >= SafetyLevel.OVERRIDE.value:
            self.animation.suppress()  # theatrics end where safety begins
        else:
            self.animation.release()

        io.write(verdict.action)
        # Reduced-order twins move the base from the commanded twist; make sure a
        # safety override actually halts them (full-physics backends ignore this).
        effective_cmd = (
            LocomotionCommand()
            if verdict.level.value >= SafetyLevel.OVERRIDE.value
            else styled
        )
        io.set_base_command_hint(effective_cmd)
        io.step()
        self._tick += 1
        return StepInfo(obs.stamp, obs, state, self._intent, verdict, world, overlay)

    @staticmethod
    def _apply_overlay(cmd: LocomotionCommand, ov) -> LocomotionCommand:
        """Blend the animation overlay into the VLA's command. Additive on the
        style channels, multiplicative (and only ever <= 1) on gait energy."""
        return LocomotionCommand(
            vx=cmd.vx * ov.speed_scale,
            vy=cmd.vy * ov.speed_scale,
            wz=cmd.wz * ov.speed_scale,
            body_height=cmd.body_height + ov.body_height,
            look_yaw=cmd.look_yaw + ov.look_yaw,
            look_pitch=cmd.look_pitch + ov.look_pitch,
        )

    # -- state estimation ------------------------------------------------- #
    def _estimate_state(self, obs: Observation, world: WorldBelief) -> RobotState:
        """In sim, base pose/twist are ground truth. On hardware these slots are
        filled by the SLAM/EKF estimate — the contract is identical, so nothing
        downstream changes."""
        base_pose = obs.base_pose if obs.base_pose is not None else world.robot_pose
        base_twist = obs.base_twist if obs.base_twist is not None else Twist(np.zeros(3), np.zeros(3))
        return RobotState(
            stamp=obs.stamp,
            base_pose=base_pose,
            base_twist=base_twist,
            joints=obs.joints,
            battery=obs.battery,
            water=obs.water,
            upright_cos=upright_cosine(obs.imu.orientation),
        )
