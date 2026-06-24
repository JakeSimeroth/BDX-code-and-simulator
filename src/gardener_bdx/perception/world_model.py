"""The :class:`WorldBelief` and the :class:`GreenhouseMapper` that maintains it.

The mapper fuses three things each tick:
  * LiDAR points  -> occupancy grid (obstacles, walls, benches),
  * detections    -> tracked plants / humans / dock,
  * base pose     -> where all of the above sit in the world frame.

Detections arrive either as privileged sim semantics (``RobotIO.semantics()``)
or, on hardware, from the neural detectors in :mod:`plant_detector`. The mapper
does not care which — it only consumes :class:`Detection`."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..common.types import Detection, Observation, Pose, SceneSemantics
from .occupancy import OccupancyGrid


@dataclass
class Plant:
    id: int
    position: np.ndarray  # (3,) world
    dryness: float = 0.5  # 0 = well watered, 1 = parched
    species: str = "generic"
    last_watered_s: float = -1e9
    watered_this_episode: bool = False

    def needs_water(self, threshold: float) -> bool:
        return self.dryness >= threshold and not self.watered_this_episode


@dataclass
class Human:
    id: int
    position: np.ndarray  # (3,) world
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    last_seen_s: float = 0.0


@dataclass
class WorldBelief:
    stamp: float
    robot_pose: Pose
    occupancy: OccupancyGrid
    plants: dict[int, Plant] = field(default_factory=dict)
    humans: dict[int, Human] = field(default_factory=dict)
    dock_pose: Optional[Pose] = None

    # -- queries used by the expert policy & safety ------------------------ #
    def robot_xy(self) -> np.ndarray:
        return self.robot_pose.position[:2]

    def nearest_dry_plant(self, dryness_threshold: float) -> Optional[Plant]:
        candidates = [p for p in self.plants.values() if p.needs_water(dryness_threshold)]
        if not candidates:
            return None
        rxy = self.robot_xy()
        return min(candidates, key=lambda p: float(np.linalg.norm(p.position[:2] - rxy)))

    def distance_to_nearest_human(self) -> float:
        if not self.humans:
            return float("inf")
        rxy = self.robot_xy()
        return min(float(np.linalg.norm(h.position[:2] - rxy)) for h in self.humans.values())

    def nearest_human_xy(self) -> Optional[np.ndarray]:
        if not self.humans:
            return None
        rxy = self.robot_xy()
        h = min(self.humans.values(), key=lambda h: float(np.linalg.norm(h.position[:2] - rxy)))
        return h.position[:2]


class GreenhouseMapper:
    """Stateful: it accumulates the occupancy grid and the plant/human registry
    over time, just like a real mapping stack."""

    def __init__(self, dock_pose: Optional[Pose] = None):
        self.belief = WorldBelief(
            stamp=0.0,
            robot_pose=Pose.identity(),
            occupancy=OccupancyGrid(),
            dock_pose=dock_pose,
        )
        self._next_human_gc = 0.0

    def update(self, obs: Observation, semantics: Optional[SceneSemantics]) -> WorldBelief:
        b = self.belief
        b.stamp = obs.stamp

        # Pose: in sim we have ground truth; on hardware this slot is filled by
        # the SLAM estimate (LiDAR+camera+IMU fusion) — same downstream contract.
        if obs.base_pose is not None:
            b.robot_pose = obs.base_pose

        # LiDAR -> occupancy. Points are sensor-frame; lift to world via pose.
        if obs.lidar is not None and obs.lidar.points.shape[0] > 0:
            from ..common.math_utils import quat_rotate

            q = b.robot_pose.orientation
            t = b.robot_pose.position
            pts_world = np.array([quat_rotate(q, p) + t for p in obs.lidar.points])
            # Keep returns that are not floor/ceiling.
            mask = (pts_world[:, 2] > 0.03) & (pts_world[:, 2] < 1.5)
            b.occupancy.integrate_scan(t[:2], pts_world[mask, :2])

        # Detections -> registries.
        if semantics is not None:
            self._integrate_detections(obs.stamp, semantics.detections)
        return b

    def _integrate_detections(self, stamp: float, detections: tuple[Detection, ...]) -> None:
        b = self.belief
        seen_humans: set[int] = set()
        for d in detections:
            if d.kind == "plant":
                pl = b.plants.get(d.id)
                dry = float(d.attributes.get("dryness", 0.5))
                if pl is None:
                    b.plants[d.id] = Plant(
                        id=d.id,
                        position=np.array(d.position, dtype=float),
                        dryness=dry,
                        species=d.attributes.get("species", "generic"),
                    )
                else:
                    pl.position = np.array(d.position, dtype=float)
                    pl.dryness = dry
            elif d.kind == "human":
                seen_humans.add(d.id)
                h = b.humans.get(d.id)
                pos = np.array(d.position, dtype=float)
                if h is None:
                    b.humans[d.id] = Human(id=d.id, position=pos, last_seen_s=stamp)
                else:
                    dt = max(stamp - h.last_seen_s, 1e-3)
                    h.velocity = (pos - h.position) / dt
                    h.position = pos
                    h.last_seen_s = stamp
            elif d.kind == "dock":
                b.dock_pose = Pose(np.array(d.position, dtype=float), np.array([1.0, 0, 0, 0]))

        # Forget humans we have not seen for a while (they left the aisle).
        stale = [hid for hid, h in b.humans.items() if stamp - h.last_seen_s > 2.0]
        for hid in stale:
            del b.humans[hid]

    def mark_watered(self, plant_id: int, stamp: float) -> None:
        pl = self.belief.plants.get(plant_id)
        if pl is not None:
            pl.watered_this_episode = True
            pl.last_watered_s = stamp
            pl.dryness = max(0.0, pl.dryness - 0.6)
