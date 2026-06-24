"""The VLA brain — System 2 (reasoning) + System 1 (action).

This is the centerpiece of GardenerBDX. A single Vision-Language-Action world
model consumes raw multimodal sensors plus a natural-language goal and emits
whole-body :class:`Intent`. Two implementations share one interface:

``NeuralVLA``
    The real thing: an end-to-end, GR00T-style dual-system network (see
    :mod:`gardener_bdx.policy.vla_net`). System 2 (a vision-language module)
    reasons slowly about the scene and goal; System 1 (a flow-matching action
    head) emits continuous whole-body intent fast. Raw pixels/points/proprio +
    text in, action out. Requires PyTorch + a checkpoint (ours or GR00T N1).

``ScriptedGardenerVLA``
    A transparent, competent gardener implemented with classic control over the
    :class:`WorldBelief`. It needs zero ML dependencies, so it powers the
    runs-anywhere simulation demo and the test suite — and, crucially, it is the
    **expert demonstrator** whose rollouts we distill into ``NeuralVLA``
    (behavior cloning / DAgger). The neural policy is meant to *match then
    exceed* this baseline, learning directly from pixels what the scripted
    version gets from privileged map state.

Both return an :class:`Intent`; everything downstream (locomotion substrate,
safety Guardian, runner) is identical regardless of which brain is in charge.
"""

from __future__ import annotations

import abc
from typing import Optional

import numpy as np

from ..common.math_utils import wrap_to_pi, yaw_of
from ..common.types import Intent, LocomotionCommand, Observation, Pose, Skill
from ..perception.world_model import Plant, WorldBelief
from .task import TaskGoal, TaskKind

# Canonical continuous-action layout shared by the neural net and Intent decode.
ACTION_KEYS = ("vx", "vy", "wz", "body_height", "look_yaw", "look_pitch", "dispense")
ACTION_DIM = len(ACTION_KEYS)


class VLAPolicy(abc.ABC):
    """Vision-Language-Action policy interface."""

    @abc.abstractmethod
    def reset(self) -> None:
        """Clear episodic state (caches, committed targets, recurrent latents)."""

    @abc.abstractmethod
    def act(
        self,
        obs: Observation,
        goal: TaskGoal,
        world: Optional[WorldBelief] = None,
    ) -> Intent:
        """Map (raw observation, language goal[, optional world belief]) -> Intent.

        ``world`` is always available in simulation and used by the scripted
        expert. The neural policy is end-to-end and uses only ``obs`` + ``goal``
        for action; it may read ``world`` solely to annotate a target pose for
        logging/grounding."""


# --------------------------------------------------------------------------- #
# Scripted expert / oracle
# --------------------------------------------------------------------------- #


class ScriptedGardenerVLA(VLAPolicy):
    """A finite-state greenhouse gardener. Deterministic, debuggable, and good
    enough to be the teacher for the neural VLA."""

    def __init__(
        self,
        max_speed: float = 0.5,
        approach_radius: float = 1.1,
        standoff: float = 0.45,
        dispense_rate_lps: float = 0.06,
        human_slow_radius: float = 1.6,
        human_stop_radius: float = 0.7,
    ):
        self.max_speed = max_speed
        self.approach_radius = approach_radius
        self.standoff = standoff
        self.dispense_rate_lps = dispense_rate_lps
        self.human_slow_radius = human_slow_radius
        self.human_stop_radius = human_stop_radius
        self.reset()

    def reset(self) -> None:
        self._committed_id: Optional[int] = None
        self._patrol_idx = 0
        self._laps = 0
        self.well_watered = 0.30  # water a plant down to this, then move on
        # A perimeter sweep that, with the LiDAR/camera range, reveals the whole
        # greenhouse — used both for PATROL and as the TEND search pattern.
        self._patrol_waypoints = np.array(
            [[3.5, 3.5], [3.5, -3.5], [-3.5, -3.5], [-3.5, 3.5]], dtype=float
        )

    # -- main ------------------------------------------------------------- #
    def act(self, obs, goal: TaskGoal, world: Optional[WorldBelief] = None) -> Intent:
        if world is None:
            # Without a belief the scripted policy is blind; hold still safely.
            return self._idle("no world belief available")

        soc = obs.battery.state_of_charge
        water_l = obs.water.level_liters
        need_charge = soc <= goal.return_to_dock_soc
        out_of_water = water_l < goal.water_per_plant_l * 0.5

        if goal.kind == TaskKind.IDLE:
            return self._idle("idle goal")

        if need_charge or goal.kind == TaskKind.GO_DOCK or (
            out_of_water and self._has_dry(world, goal)
        ):
            why = "battery low" if need_charge else ("tank empty" if out_of_water else "dock requested")
            return self._dock(world, reason=why)

        if goal.kind == TaskKind.PATROL:
            return self._patrol(world)

        target = self._select_target(world, goal)
        if target is None:
            # Nothing known needs water. A real gardener doesn't quit — it goes
            # looking. Sweep the greenhouse to discover plants; only idle once
            # we've completed a survey lap with nothing thirsty.
            self._committed_id = None
            if self._laps >= 1:
                return self._idle("garden surveyed; nothing thirsty")
            return self._search(world)
        return self._tend(world, obs, goal, target)

    # -- behaviors -------------------------------------------------------- #
    def _tend(self, world, obs, goal, plant: Plant) -> Intent:
        rxy = world.robot_xy()
        rng = float(np.linalg.norm(plant.position[:2] - rxy))
        still_dry = plant.dryness > self.well_watered

        if rng > self.approach_radius:
            cmd = self._goto(world, plant.position[:2], face_xy=plant.position[:2])
            return Intent(Skill.NAVIGATE, cmd, target_pose=_xy_pose(plant.position),
                          rationale=f"navigating to plant#{plant.id} ({rng:.2f} m)")

        if rng > self.standoff + 0.06:
            cmd = self._goto(world, plant.position[:2], face_xy=plant.position[:2],
                             desired_range=self.standoff)
            return Intent(Skill.APPROACH_PLANT, cmd, target_pose=_xy_pose(plant.position),
                          rationale=f"fine-approaching plant#{plant.id}")

        # In position: face the plant, tilt the head down, and meter water while
        # the plant still reads dry. The sim wets the soil; we observe & stop.
        yaw = yaw_of(world.robot_pose.orientation)
        desired_yaw = float(np.arctan2(*(plant.position[:2] - rxy)[::-1]))
        cmd = LocomotionCommand(
            vx=0.0, vy=0.0, wz=float(np.clip(2.0 * wrap_to_pi(desired_yaw - yaw), -0.8, 0.8)),
            look_pitch=-0.5,
        )
        if still_dry and obs.water.level_liters > 0.0:
            return Intent(Skill.DISPENSE_WATER, cmd, dispense_rate_lps=self.dispense_rate_lps,
                          target_pose=_xy_pose(plant.position),
                          rationale=f"watering plant#{plant.id} (dryness {plant.dryness:.2f})")
        # Done with this one.
        self._committed_id = None
        return Intent(Skill.APPROACH_PLANT, cmd, target_pose=_xy_pose(plant.position),
                      rationale=f"plant#{plant.id} satisfied")

    def _dock(self, world, reason: str) -> Intent:
        if world.dock_pose is None:
            # Dock not yet localized — sweep to find it rather than stalling.
            intent = self._search(world)
            intent.rationale = f"{reason}; searching for dock"
            return intent
        dxy = world.dock_pose.position[:2]
        rng = float(np.linalg.norm(dxy - world.robot_xy()))
        cmd = self._goto(world, dxy, face_xy=dxy, desired_range=0.0)
        return Intent(Skill.DOCK_CHARGE, cmd, target_pose=world.dock_pose,
                      rationale=f"docking ({reason}, {rng:.2f} m)")

    def _search(self, world) -> Intent:
        """Drive the survey sweep while tending, to discover plants we can't yet
        see. Identical motion to patrol, but framed as 'looking for thirsty
        plants' and it counts laps so the task can terminate."""
        wp = self._patrol_waypoints[self._patrol_idx]
        if float(np.linalg.norm(wp - world.robot_xy())) < 0.5:
            self._patrol_idx += 1
            if self._patrol_idx >= len(self._patrol_waypoints):
                self._patrol_idx = 0
                self._laps += 1
            wp = self._patrol_waypoints[self._patrol_idx]
        cmd = self._goto(world, wp, face_xy=wp)
        return Intent(Skill.NAVIGATE, cmd, target_pose=_xy_pose(np.array([wp[0], wp[1], 0.0])),
                      rationale=f"searching for thirsty plants (lap {self._laps})")

    def _patrol(self, world) -> Intent:
        wp = self._patrol_waypoints[self._patrol_idx]
        if float(np.linalg.norm(wp - world.robot_xy())) < 0.4:
            self._patrol_idx = (self._patrol_idx + 1) % len(self._patrol_waypoints)
            wp = self._patrol_waypoints[self._patrol_idx]
        cmd = self._goto(world, wp, face_xy=wp)
        cov = world.occupancy.coverage_fraction()
        return Intent(Skill.NAVIGATE, cmd, target_pose=_xy_pose(np.array([wp[0], wp[1], 0.0])),
                      rationale=f"patrolling (coverage {cov:.0%})")

    def _idle(self, why: str) -> Intent:
        return Intent(Skill.IDLE, LocomotionCommand(), rationale=why)

    # -- helpers ---------------------------------------------------------- #
    def _has_dry(self, world, goal) -> bool:
        return world.nearest_dry_plant(goal.dryness_threshold) is not None

    def _select_target(self, world, goal) -> Optional[Plant]:
        if goal.kind == TaskKind.WATER_PLANT and goal.target_plant_id is not None:
            return world.plants.get(goal.target_plant_id)
        # Commit to a plant until it is well watered, to avoid dithering between
        # two equally-thirsty plants.
        if self._committed_id is not None:
            pl = world.plants.get(self._committed_id)
            if pl is not None and pl.dryness > self.well_watered + 0.05:
                return pl
        pl = world.nearest_dry_plant(goal.dryness_threshold)
        self._committed_id = pl.id if pl is not None else None
        return pl

    def _goto(
        self,
        world: WorldBelief,
        target_xy: np.ndarray,
        face_xy: Optional[np.ndarray] = None,
        desired_range: float = 0.0,
    ) -> LocomotionCommand:
        """Holonomic-ish go-to in body frame (a BDX can side-step). Produces a
        velocity command the locomotion substrate will realize as a gait."""
        rxy = world.robot_xy()
        yaw = yaw_of(world.robot_pose.orientation)
        d = np.asarray(target_xy, dtype=float) - rxy
        rng = float(np.linalg.norm(d))

        # Body-frame error.
        c, s = np.cos(-yaw), np.sin(-yaw)
        bx = c * d[0] - s * d[1]
        by = s * d[0] + c * d[1]
        if rng > 1e-6 and desired_range > 0.0:
            scale = max(0.0, (rng - desired_range)) / rng
            bx, by = bx * scale, by * scale

        kp = 1.2
        vx = float(np.clip(kp * bx, -self.max_speed, self.max_speed))
        vy = float(np.clip(kp * by, -0.5 * self.max_speed, 0.5 * self.max_speed))

        # Heading: face the requested point (or direction of travel).
        if face_xy is not None:
            fd = np.asarray(face_xy, dtype=float) - rxy
        else:
            fd = d
        desired_yaw = float(np.arctan2(fd[1], fd[0])) if np.linalg.norm(fd) > 1e-6 else yaw
        wz = float(np.clip(1.6 * wrap_to_pi(desired_yaw - yaw), -1.0, 1.0))

        # Politeness: slow (and, very close, stop) near people. The Guardian also
        # enforces this in hardware-grade fashion; the expert just behaves well.
        dh = world.distance_to_nearest_human()
        if dh < self.human_stop_radius:
            vx = vy = 0.0
        elif dh < self.human_slow_radius:
            f = (dh - self.human_stop_radius) / (self.human_slow_radius - self.human_stop_radius)
            vx *= f
            vy *= f

        # Light reactive obstacle steering: if a cell ~0.5 m ahead is occupied,
        # bias laterally and back off forward speed.
        ahead = rxy + np.array([np.cos(yaw), np.sin(yaw)]) * 0.5
        if world.occupancy.is_occupied(ahead):
            vx *= 0.3
            vy += 0.25 * self.max_speed * (1.0 if by >= 0 else -1.0)

        return LocomotionCommand(vx=vx, vy=vy, wz=wz)


def _xy_pose(xyz: np.ndarray) -> Pose:
    return Pose(np.asarray(xyz, dtype=float), np.array([1.0, 0.0, 0.0, 0.0]))


# --------------------------------------------------------------------------- #
# Neural VLA (end-to-end). Heavy deps are imported lazily so this module stays
# importable (and testable) without PyTorch.
# --------------------------------------------------------------------------- #


class NeuralVLA(VLAPolicy):
    """End-to-end VLA world model. Thin wrapper that owns the torch network from
    :mod:`gardener_bdx.policy.vla_net`, manages the System-2 update cadence, and
    decodes the network's continuous output into an :class:`Intent`."""

    def __init__(
        self,
        checkpoint: str = "",
        device: str = "cuda",
        system2_period: int = 10,
        backbone: str = "world_model",  # "world_model" | "groot"
    ):
        # Lazy import keeps the package usable with zero ML deps installed.
        from .vla_net import WorldModelVLANet  # noqa: WPS433 (intentional lazy import)

        self.device = device
        self.system2_period = int(system2_period)
        self.net = WorldModelVLANet(action_dim=ACTION_DIM)
        self.net.load(checkpoint, device=device, backbone=backbone)
        self._history_len = self.net.history_len
        self.reset()

    def reset(self) -> None:
        self._tick = 0
        self._latent = None  # cached System-2 reasoning latent
        self._skill = Skill.IDLE
        self._history: list = []  # rolling window of recent observations
        self.net.reset()

    def act(self, obs, goal: TaskGoal, world: Optional[WorldBelief] = None) -> Intent:
        # Maintain the temporal window the System-2 reasoner sees.
        self._history.append(obs)
        if len(self._history) > self._history_len:
            self._history.pop(0)

        refresh_system2 = (self._tick % self.system2_period) == 0
        # System 2: slow reasoning over a *history* of vision+proprio + language.
        if refresh_system2 or self._latent is None:
            self._latent, skill_id = self.net.reason(self._history, goal.instruction)
            self._skill = list(Skill)[int(skill_id) % len(Skill)]
        # System 1: fast flow-matching action head conditioned on the latent.
        a = self.net.act(obs, self._latent)  # np.ndarray (ACTION_DIM,)
        self._tick += 1

        cmd = LocomotionCommand(
            vx=float(a[0]), vy=float(a[1]), wz=float(a[2]),
            body_height=float(a[3]), look_yaw=float(a[4]), look_pitch=float(a[5]),
        )
        target = None
        if world is not None and self._skill in (Skill.NAVIGATE, Skill.APPROACH_PLANT,
                                                 Skill.DISPENSE_WATER):
            p = world.nearest_dry_plant(goal.dryness_threshold)
            target = _xy_pose(p.position) if p is not None else None
        return Intent(
            skill=self._skill,
            locomotion=cmd,
            dispense_rate_lps=max(0.0, float(a[6])),
            target_pose=target,
            confidence=1.0,
            rationale=f"neural VLA [{self._skill.value}]",
        )


def build_vla(kind: str = "scripted", **kwargs) -> VLAPolicy:
    """Factory used by the runtime. ``kind`` is "scripted" (default, runs
    anywhere) or "neural" (loads the torch world model / GR00T checkpoint)."""
    kind = kind.lower()
    if kind in ("scripted", "expert", "oracle"):
        return ScriptedGardenerVLA(**kwargs)
    if kind == "groot":
        from .groot_vla import GR00TVLA  # lazy: pulls in the gr00t stack

        return GR00TVLA(**kwargs)
    if kind in ("neural", "vla", "world_model"):
        return NeuralVLA(**kwargs)
    raise ValueError(f"unknown VLA kind: {kind!r}")
