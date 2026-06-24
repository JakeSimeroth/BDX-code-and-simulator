"""A pure-numpy, reduced-order greenhouse twin.

It does *not* simulate contact dynamics — the base is moved by the commanded
body twist — so it cannot validate the gait. What it *can* do, with zero heavy
dependencies, is close the **gardener** loop end-to-end: navigation, plant
detection, watering, battery/water budgets, human-proximity safety, docking. It
is the default for tests and the 10-second demo, and a fast environment for
shaping the high-level task before paying for full physics in Isaac/MuJoCo.

The scene: a rectangular greenhouse with two benches of potted plants (each with
a soil-moisture / dryness state), a charging+refill dock, and one human walking
an aisle. LiDAR is ray-cast against walls/plants/human; semantics expose plants,
the human and the dock within sensor range (so the world belief fills in as the
robot explores)."""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..common.config import RobotConfig
from ..common.math_utils import quat_from_euler
from ..common.types import (
    Action,
    BatteryState,
    CameraFrame,
    Detection,
    ImuReading,
    JointState,
    LidarScan,
    LocomotionCommand,
    Observation,
    Pose,
    SceneSemantics,
    Twist,
    WaterTankState,
)
from ..interfaces.robot_io import RobotIO


class KinematicGreenhouse(RobotIO):
    def __init__(
        self,
        robot_config: RobotConfig,
        control_dt: float,
        n_plants: int = 6,
        arena_half: float = 5.0,
        sensor_range: float = 4.0,
        seed: int = 0,
        max_speed: float = 0.6,
    ):
        super().__init__(robot_config, control_dt)
        self.rc = robot_config
        self.rng = np.random.default_rng(seed)
        self.n_plants = n_plants
        self.arena = arena_half
        self.sensor_range = sensor_range
        self.max_speed = max_speed
        self._cmd = LocomotionCommand()
        self.reset()

    # -- RobotIO ---------------------------------------------------------- #
    def is_simulation(self) -> bool:
        return True

    def reset(self) -> Observation:
        n = self.rc.n_joints
        self.t = 0.0
        self.base_xy = np.array([0.0, -self.arena + 1.0])
        self.base_yaw = np.pi / 2  # face into the greenhouse
        self.q = self.rc.default_joint_positions.copy()
        self.q_prev = self.q.copy()
        self.soc = 0.9
        self.water_l = self.rc.water_capacity_liters
        self._valve = 0.0

        # Two benches of plants along y, split left/right of the centre aisle.
        xs = self.rng.uniform(1.5, 4.0, size=self.n_plants) * np.where(
            self.rng.random(self.n_plants) < 0.5, -1.0, 1.0
        )
        ys = np.linspace(-self.arena + 2.0, self.arena - 1.0, self.n_plants)
        self.plant_xy = np.stack([xs, ys], axis=1)
        self.plant_dryness = self.rng.uniform(0.2, 0.95, size=self.n_plants)

        # Charging + refill dock in a back corner.
        self.dock_xy = np.array([-self.arena + 0.6, -self.arena + 0.6])

        # One human pacing the central aisle.
        self.human_xy = np.array([0.0, self.arena - 2.0])
        self.human_wps = np.array([[0.4, self.arena - 2.0], [0.4, -self.arena + 2.0]])
        self.human_target = 1
        self._cmd = LocomotionCommand()
        return self._observe()

    def read(self) -> Observation:
        return self._observe()

    def write(self, action: Action) -> None:
        # Reduced-order twin: tracks joint targets instantly (no contact), but we
        # still record them so velocities and the locomotion path are exercised.
        self.q_prev = self.q.copy()
        self.q = np.clip(action.joint_position_targets, self.rc.joint_lower, self.rc.joint_upper)
        self._valve = float(action.water_valve_lps)

    def set_base_command_hint(self, cmd: LocomotionCommand) -> None:
        self._cmd = cmd

    def step(self) -> None:
        dt = self.dt
        self.t += dt

        docked = float(np.linalg.norm(self.base_xy - self.dock_xy)) < 0.6
        if docked:
            # On the dock we charge and refill instead of driving.
            self.soc = min(1.0, self.soc + 0.04 * dt)
            self.water_l = min(self.rc.water_capacity_liters, self.water_l + 0.3 * dt)
        else:
            # Integrate the (post-safety) body twist into world motion.
            c, s = np.cos(self.base_yaw), np.sin(self.base_yaw)
            vx = np.clip(self._cmd.vx, -self.max_speed, self.max_speed)
            vy = np.clip(self._cmd.vy, -self.max_speed, self.max_speed)
            self.base_xy = self.base_xy + np.array([c * vx - s * vy, s * vx + c * vy]) * dt
            self.base_yaw = (self.base_yaw + self._cmd.wz * dt + np.pi) % (2 * np.pi) - np.pi
            self.base_xy = np.clip(self.base_xy, -self.arena + 0.3, self.arena - 0.3)

        # Battery drain (idle + motion).
        speed = float(np.hypot(self._cmd.vx, self._cmd.vy) + abs(self._cmd.wz))
        if not docked:
            self.soc = max(0.0, self.soc - (0.0005 + 0.004 * speed) * dt)

        # Watering: if the valve is open and a plant sits in the frontal cone,
        # wet its soil (dryness falls) and draw down the tank.
        if self._valve > 0.0 and self.water_l > 0.0:
            pid = self._plant_in_cone()
            if pid is not None:
                delivered = min(self._valve * dt, self.water_l)
                self.water_l -= delivered
                self.plant_dryness[pid] = max(0.0, self.plant_dryness[pid] - 2.5 * delivered)

        # Plants slowly dry out (so "tend the garden" is never quite finished).
        self.plant_dryness = np.clip(self.plant_dryness + 0.0008 * dt, 0.0, 1.0)

        # Human paces the aisle.
        tgt = self.human_wps[self.human_target]
        d = tgt - self.human_xy
        if np.linalg.norm(d) < 0.1:
            self.human_target ^= 1
        else:
            self.human_xy = self.human_xy + d / np.linalg.norm(d) * 0.4 * dt

    def now(self) -> float:
        return self.t

    def semantics(self) -> Optional[SceneSemantics]:
        dets: list[Detection] = []
        for i, p in enumerate(self.plant_xy):
            if np.linalg.norm(p - self.base_xy) <= self.sensor_range:
                dets.append(
                    Detection("plant", i, np.array([p[0], p[1], 0.25]),
                              {"dryness": float(self.plant_dryness[i]), "species": "potted"})
                )
        if np.linalg.norm(self.human_xy - self.base_xy) <= self.sensor_range + 1.5:
            dets.append(Detection("human", 0, np.array([self.human_xy[0], self.human_xy[1], 0.9])))
        # The dock is a fixed, known fixture (a beacon on the prior map), so it is
        # always localized — the robot never has to "discover" where to charge.
        dets.append(Detection("dock", 0, np.array([self.dock_xy[0], self.dock_xy[1], 0.0])))
        return SceneSemantics(stamp=self.t, detections=tuple(dets))

    # -- observation synthesis ------------------------------------------- #
    def _observe(self) -> Observation:
        quat = quat_from_euler(0.0, 0.0, self.base_yaw)
        vel = (self.q - self.q_prev) / self.dt
        imu = ImuReading(
            orientation=quat,
            angular_velocity=np.array([0.0, 0.0, self._cmd.wz]),
            linear_acceleration=np.array([0.0, 0.0, -9.81]),
            stamp=self.t,
        )
        joints = JointState(self.q.copy(), vel, np.zeros(self.rc.n_joints),
                            names=self.rc.joint_names, stamp=self.t)
        pose = Pose(np.array([self.base_xy[0], self.base_xy[1], self.rc.base_height_nominal]), quat)
        twist = Twist(np.array([self._cmd.vx, self._cmd.vy, 0.0]),
                      np.array([0.0, 0.0, self._cmd.wz]))
        return Observation(
            stamp=self.t,
            imu=imu,
            joints=joints,
            camera=self._render_camera(),
            lidar=self._raycast_lidar(),
            battery=BatteryState(voltage=14.8 * self.soc + 9.0, state_of_charge=self.soc,
                                 charging=bool(np.linalg.norm(self.base_xy - self.dock_xy) < 0.6),
                                 stamp=self.t),
            water=WaterTankState(level_liters=self.water_l,
                                 capacity_liters=self.rc.water_capacity_liters,
                                 dispensing=self._valve > 0, flow_rate_lps=self._valve, stamp=self.t),
            base_pose=pose,
            base_twist=twist,
        )

    def _render_camera(self) -> CameraFrame:
        # Placeholder RGB-D so the neural path has the right tensor shapes. The
        # MuJoCo/Isaac backends provide real renders; here we paint a simple
        # ground/sky gradient (never all-zeros, which degrades CNNs).
        h, w = 48, 64
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        rgb[: h // 2, :, 2] = 140  # sky
        rgb[h // 2 :, :, 1] = 90  # ground
        depth = np.full((h, w), 3.0, dtype=np.float32)
        return CameraFrame(rgb=rgb, depth=depth, frame="head_camera", stamp=self.t)

    def _raycast_lidar(self, n_rays: int = 72) -> LidarScan:
        angles = np.linspace(-np.pi, np.pi, n_rays, endpoint=False)
        circles = [(p, 0.18) for p in self.plant_xy]
        circles.append((self.human_xy, 0.30))
        pts = []
        for a in angles:
            world_a = self.base_yaw + a
            d = self._ray_distance(world_a, circles)
            if d < self.sensor_range:
                pts.append([d * np.cos(a), d * np.sin(a), 0.1])
        points = np.array(pts) if pts else np.zeros((0, 3))
        return LidarScan(points=points, frame="lidar", stamp=self.t)

    def _ray_distance(self, world_angle: float, circles) -> float:
        dirv = np.array([np.cos(world_angle), np.sin(world_angle)])
        best = self.sensor_range
        # Walls of the arena box.
        for axis, sign in [(0, 1), (0, -1), (1, 1), (1, -1)]:
            wall = sign * self.arena
            if abs(dirv[axis]) > 1e-6:
                tt = (wall - self.base_xy[axis]) / dirv[axis]
                if 0 < tt < best:
                    hit = self.base_xy + dirv * tt
                    if -self.arena <= hit[1 - axis] <= self.arena:
                        best = tt
        # Plant / human circles.
        for centre, r in circles:
            oc = self.base_xy - centre
            b = 2 * dirv @ oc
            c = oc @ oc - r * r
            disc = b * b - 4 * c
            if disc >= 0:
                tt = (-b - np.sqrt(disc)) / 2
                if 0 < tt < best:
                    best = tt
        return best

    # -- helpers ---------------------------------------------------------- #
    def _plant_in_cone(self, max_dist: float = 0.65, half_angle: float = 0.5) -> Optional[int]:
        fwd = np.array([np.cos(self.base_yaw), np.sin(self.base_yaw)])
        best, best_d = None, max_dist
        for i, p in enumerate(self.plant_xy):
            d = p - self.base_xy
            dist = float(np.linalg.norm(d))
            if dist < 1e-6 or dist > best_d:
                continue
            if (d / dist) @ fwd >= np.cos(half_angle):
                best, best_d = i, dist
        return best
