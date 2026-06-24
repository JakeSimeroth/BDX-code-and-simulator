"""MuJoCo full-physics backend.

This is the backend that actually validates the **gait** — contact, balance,
slipping, the lot — and so it is where the locomotion substrate is trained
(MJX/PPO) and where Sim2Real for walking begins. It is the same engine lineage
as the Disney BDX / Open Duck Mini RL work.

It also composites the greenhouse task scene (plants, dock, human) around the
robot so the full gardener loop runs under real physics, and can render the head
camera to feed the VLA real pixels. Requires ``mujoco`` (``pip install -e
'.[sim]'``); the package stays importable without it because this module is only
imported on demand by :func:`gardener_bdx.sim.make_backend`."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from ..common.config import RobotConfig, repo_root
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

try:
    import mujoco

    _MJ_OK = True
except Exception as _e:  # pragma: no cover
    _MJ_OK = False
    _MJ_ERR = _e


class MujocoGreenhouse(RobotIO):
    def __init__(
        self,
        robot_config: RobotConfig,
        control_dt: float,
        model_path: str = "models/robot/gardener_bdx.xml",
        n_plants: int = 6,
        arena_half: float = 5.0,
        render: bool = False,
        seed: int = 0,
    ):
        if not _MJ_OK:  # pragma: no cover
            raise ImportError(f"MuJoCo backend needs `mujoco`. Install '.[sim]'. ({_MJ_ERR})")
        super().__init__(robot_config, control_dt)
        self.rc = robot_config
        self.rng = np.random.default_rng(seed)
        self.arena = arena_half
        self.n_plants = n_plants
        self._physics_dt = None

        xml = (Path(repo_root()) / model_path).read_text()
        xml = self._composite_scene(xml)
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        self._physics_dt = self.model.opt.timestep
        self._substeps = max(1, round(control_dt / self._physics_dt))

        # Canonical joint -> qpos/dof/actuator index maps (by name, robust to XML order).
        self.qadr = np.array([self.model.joint(n).qposadr[0] for n in self.rc.joint_names])
        self.dadr = np.array([self.model.joint(n).dofadr[0] for n in self.rc.joint_names])
        self.act_id = np.array([self.model.actuator(n).id for n in self.rc.joint_names])
        self._root_qadr = self.model.joint("root").qposadr[0]
        self._root_dadr = self.model.joint("root").dofadr[0]

        self.renderer = mujoco.Renderer(self.model, 48, 64) if render else None
        self.reset()

    def is_simulation(self) -> bool:
        return True

    # -- scene compositing ------------------------------------------------ #
    def _composite_scene(self, robot_xml: str) -> str:
        """Inject plants (cylinders), a dock (box) and a mocap human into the
        robot worldbody so the full task runs under physics."""
        xs = self.rng.uniform(1.5, 4.0, self.n_plants) * np.where(
            self.rng.random(self.n_plants) < 0.5, -1.0, 1.0
        )
        ys = np.linspace(-self.arena + 2, self.arena - 1, self.n_plants)
        self.plant_xy = np.stack([xs, ys], 1)
        self.plant_dryness = self.rng.uniform(0.2, 0.95, self.n_plants)
        self.dock_xy = np.array([-self.arena + 0.6, -self.arena + 0.6])

        bodies = ['<body name="dock" pos="%f %f 0.02"><geom type="box" size="0.2 0.2 0.02" '
                  'rgba="0.9 0.7 0.1 1"/></body>' % (self.dock_xy[0], self.dock_xy[1])]
        for i, (x, y) in enumerate(self.plant_xy):
            bodies.append(
                f'<body name="plant_{i}" pos="{x:.3f} {y:.3f} 0">'
                f'<geom type="cylinder" size="0.06 0.12" pos="0 0 0.12" rgba="0.2 0.6 0.2 1"/>'
                f'<geom type="cylinder" size="0.09 0.06" rgba="0.5 0.3 0.1 1"/></body>'
            )
        # Mocap human (kinematically driven, collides for safety realism).
        bodies.append('<body name="human" mocap="true" pos="0.4 3 0.9">'
                      '<geom type="capsule" fromto="0 0 -0.9 0 0 0.0" size="0.15" '
                      'rgba="0.9 0.5 0.4 1" contype="2" conaffinity="2"/></body>')
        return robot_xml.replace("</worldbody>", "\n".join(bodies) + "\n</worldbody>")

    # -- RobotIO ---------------------------------------------------------- #
    def reset(self) -> Observation:
        mujoco.mj_resetData(self.model, self.data)
        q = self.data.qpos
        q[self._root_qadr : self._root_qadr + 3] = [0.0, -self.arena + 1.0, self.rc.base_height_nominal]
        q[self._root_qadr + 3 : self._root_qadr + 7] = [0.7071, 0, 0, 0.7071]  # face +y
        q[self.qadr] = self.rc.default_joint_positions
        self.data.ctrl[self.act_id] = self.rc.default_joint_positions
        mujoco.mj_forward(self.model, self.data)
        self.soc = 0.9
        self.water_l = self.rc.water_capacity_liters
        self._valve = 0.0
        self.human_xy = np.array([0.4, self.arena - 2.0])
        self._human_dir = -1.0
        return self._observe()

    def read(self) -> Observation:
        return self._observe()

    def write(self, action: Action) -> None:
        self.data.ctrl[self.act_id] = np.clip(
            action.joint_position_targets, self.rc.joint_lower, self.rc.joint_upper
        )
        self._valve = float(action.water_valve_lps)

    def step(self) -> None:
        # Advance physics to the next control tick (the gait emerges from contact;
        # we do NOT use the base command hint here).
        for _ in range(self._substeps):
            mujoco.mj_step(self.model, self.data)
        self._update_task_state()

    def now(self) -> float:
        return float(self.data.time)

    def semantics(self) -> Optional[SceneSemantics]:
        base = self.data.qpos[self._root_qadr : self._root_qadr + 2]
        dets = []
        for i, p in enumerate(self.plant_xy):
            if np.linalg.norm(p - base) <= 4.0:
                dets.append(Detection("plant", i, np.array([p[0], p[1], 0.25]),
                                      {"dryness": float(self.plant_dryness[i])}))
        if np.linalg.norm(self.human_xy - base) <= 5.5:
            dets.append(Detection("human", 0, np.array([self.human_xy[0], self.human_xy[1], 0.9])))
        # Dock is a fixed, known fixture — always localized.
        dets.append(Detection("dock", 0, np.array([self.dock_xy[0], self.dock_xy[1], 0.0])))
        return SceneSemantics(stamp=self.now(), detections=tuple(dets))

    # -- internals -------------------------------------------------------- #
    def _update_task_state(self):
        dt = self.dt
        base = self.data.qpos[self._root_qadr : self._root_qadr + 2]
        docked = np.linalg.norm(base - self.dock_xy) < 0.6
        if docked:
            self.soc = min(1.0, self.soc + 0.04 * dt)
            self.water_l = min(self.rc.water_capacity_liters, self.water_l + 0.3 * dt)
        else:
            self.soc = max(0.0, self.soc - 0.0008 * dt)
        if self._valve > 0 and self.water_l > 0:
            pid = self._plant_in_cone(base)
            if pid is not None:
                delivered = min(self._valve * dt, self.water_l)
                self.water_l -= delivered
                self.plant_dryness[pid] = max(0.0, self.plant_dryness[pid] - 2.5 * delivered)
        self.plant_dryness = np.clip(self.plant_dryness + 0.0008 * dt, 0, 1)
        # Drive the mocap human along the aisle.
        self.human_xy[1] += self._human_dir * 0.4 * dt
        if not (-self.arena + 2 < self.human_xy[1] < self.arena - 2):
            self._human_dir *= -1
        hid = self.model.body("human").mocapid[0]
        self.data.mocap_pos[hid] = [self.human_xy[0], self.human_xy[1], 0.9]

    def _plant_in_cone(self, base, max_dist=0.7, half_angle=0.5):
        yaw = _yaw_from_quat(self.data.qpos[self._root_qadr + 3 : self._root_qadr + 7])
        fwd = np.array([np.cos(yaw), np.sin(yaw)])
        best, best_d = None, max_dist
        for i, p in enumerate(self.plant_xy):
            d = p - base
            dist = float(np.linalg.norm(d))
            if 1e-6 < dist <= best_d and (d / dist) @ fwd >= np.cos(half_angle):
                best, best_d = i, dist
        return best

    def _observe(self) -> Observation:
        d, m = self.data, self.model
        root_q = d.qpos[self._root_qadr + 3 : self._root_qadr + 7]
        pos = d.qpos[self._root_qadr : self._root_qadr + 3]
        quat = self._sensor("imu_quat", 4, default=root_q)
        gyro = self._sensor("imu_gyro", 3)
        acc = self._sensor("imu_acc", 3, default=np.array([0, 0, -9.81]))
        imu = ImuReading(np.array(quat), np.array(gyro), np.array(acc), stamp=self.now())
        joints = JointState(d.qpos[self.qadr].copy(), d.qvel[self.dadr].copy(),
                            d.actuator_force[self.act_id].copy(),
                            names=self.rc.joint_names, stamp=self.now())
        lin = d.qvel[self._root_dadr : self._root_dadr + 3]
        ang = d.qvel[self._root_dadr + 3 : self._root_dadr + 6]
        cam = None
        if self.renderer is not None:
            self.renderer.update_scene(d, camera="head_camera")
            cam = CameraFrame(rgb=self.renderer.render().copy(), frame="head_camera", stamp=self.now())
        return Observation(
            stamp=self.now(), imu=imu, joints=joints, camera=cam,
            lidar=self._lidar(pos[:2], _yaw_from_quat(root_q)),
            battery=BatteryState(state_of_charge=self.soc, charging=bool(
                np.linalg.norm(pos[:2] - self.dock_xy) < 0.6), stamp=self.now()),
            water=WaterTankState(self.water_l, self.rc.water_capacity_liters,
                                 self._valve > 0, self._valve, stamp=self.now()),
            base_pose=Pose(np.array(pos), np.array(root_q)),
            base_twist=Twist(np.array(lin), np.array(ang)),
        )

    def _lidar(self, base_xy, yaw, n=72, rng=4.0):
        # Sample the physics contacts is overkill; raycast the known geometry.
        circles = [(p, 0.12) for p in self.plant_xy] + [(self.human_xy, 0.2)]
        pts = []
        for a in np.linspace(-np.pi, np.pi, n, endpoint=False):
            wa = yaw + a
            dirv = np.array([np.cos(wa), np.sin(wa)])
            best = rng
            for axis, sign in [(0, 1), (0, -1), (1, 1), (1, -1)]:
                if abs(dirv[axis]) > 1e-6:
                    tt = (sign * self.arena - base_xy[axis]) / dirv[axis]
                    if 0 < tt < best:
                        best = tt
            for c, r in circles:
                oc = base_xy - c
                b = 2 * dirv @ oc
                disc = b * b - 4 * (oc @ oc - r * r)
                if disc >= 0:
                    tt = (-b - np.sqrt(disc)) / 2
                    if 0 < tt < best:
                        best = tt
            if best < rng:
                pts.append([best * np.cos(a), best * np.sin(a), 0.1])
        return LidarScan(np.array(pts) if pts else np.zeros((0, 3)), frame="lidar", stamp=self.now())

    def _sensor(self, name, dim, default=None):
        try:
            adr = self.model.sensor(name).adr[0]
            return self.data.sensordata[adr : adr + dim]
        except Exception:
            return default if default is not None else np.zeros(dim)


def _yaw_from_quat(q):
    w, x, y, z = q
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
