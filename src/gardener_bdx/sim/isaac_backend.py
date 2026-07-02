"""NVIDIA Isaac Sim single-environment twin — both the eval backend and the
**interactive** one you drive from the Isaac Sim GUI.

Two ways it's used (same code, same RobotIO contract):
  * **headless** — scripted/neural eval, or as the inference twin.
  * **GUI** (``scripts/run_isaac.py``) — boot with ``headless=False`` and *watch
    and interact*: orbit the camera, pause/scrub, drag plants, change lighting,
    add props. The control loop reads live stage state each tick, so whatever you
    change in the app immediately affects perception/behavior.

It builds the full gardener task: a greenhouse (your ``greenhouse.usd`` if
present, else procedural benches+dock+human), plant soil-moisture, water/battery
budgets, and a ground-truth ``semantics()`` channel so the VLA loop closes before
perception is trained. An RTX **RGB-D camera + RTX-LiDAR are attached to the
head** by default (``rtx_sensors=True``): the head is the BDX sensor gimbal, so
``look_yaw``/``look_pitch`` — including the expression layer's glances — aim
them. Sensor attachment is best-effort: any API mismatch on your Isaac version
degrades gracefully to the privileged ``semantics()`` channel with a printed
hint, and the task still closes.

GPU-only and never imported unless constructed. Isaac's Python API is
version-sensitive (the ``omni.isaac.core`` namespace became ``isaacsim.core`` in
4.5+); the imports below try the modern path and fall back. Untested off-GPU —
treat as a template you finish against your installed version."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from ..common.config import RobotConfig, load_yaml, repo_root
from ..common.math_utils import quat_from_euler, yaw_of
from ..common.types import (
    Action,
    BatteryState,
    CameraFrame,
    Detection,
    ImuReading,
    JointState,
    LidarScan,
    Observation,
    Pose,
    SceneSemantics,
    Twist,
    WaterTankState,
)
from ..interfaces.robot_io import RobotIO


class IsaacGreenhouse(RobotIO):
    def __init__(
        self,
        robot_config: RobotConfig,
        control_dt: float,
        usd_robot: str = "models/robot/gardener_bdx.usd",
        usd_scene: str = "models/scenes/greenhouse.usd",
        headless: bool = True,
        device: str = "cuda",
        seed: int = 0,
        rtx_sensors: bool = True,
        **scene_overrides,
    ):
        super().__init__(robot_config, control_dt)
        self.rc = robot_config
        self.device = device
        self._render = not headless          # render every step when the GUI is up
        self._rtx_sensors = rtx_sensors
        self.rng = np.random.default_rng(seed)

        scene = {**_scene_defaults(), **scene_overrides}
        self.arena = float(scene["arena_half"])
        self.n_plants = int(scene["n_plants"])
        self.sensor_range = float(scene["sensor_range"])

        self._boot(headless)
        self._build_stage(usd_robot, usd_scene)
        self._init_task_state()

    def is_simulation(self) -> bool:
        return True

    @property
    def app(self):
        """The SimulationApp — used by the GUI run loop (``while io.app.is_running()``)."""
        return self._app

    # -- boot / build ----------------------------------------------------- #
    def _boot(self, headless: bool) -> None:
        try:
            from isaacsim import SimulationApp  # Isaac Sim >= 4.0
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "Isaac backend requires the Isaac Sim runtime. Install via NVIDIA "
                "Omniverse / Isaac Lab and run under Isaac's python. See "
                "docs/ISAACSIM.md.\n"
                f"(original error: {e})"
            )
        self._app = SimulationApp({"headless": headless})

    def _build_stage(self, usd_robot: str, usd_scene: str) -> None:
        # Valid only after SimulationApp exists. Try the 4.5+ namespace first.
        try:
            from isaacsim.core.api import World
            from isaacsim.core.prims import SingleArticulation as Articulation
            from isaacsim.core.utils.stage import add_reference_to_stage
        except Exception:  # pragma: no cover - older Isaac Sim
            from omni.isaac.core import World
            from omni.isaac.core.articulations import Articulation
            from omni.isaac.core.utils.stage import add_reference_to_stage

        self._world = World(physics_dt=self.dt, rendering_dt=self.dt)

        scene_path = repo_root() / usd_scene
        if scene_path.exists():
            add_reference_to_stage(str(scene_path), "/World/Greenhouse")
            self._procedural = False
        else:
            self._world.scene.add_default_ground_plane()
            self._procedural = True  # spawn benches/dock/human ourselves in reset()

        robot_path = repo_root() / usd_robot
        if not robot_path.exists():
            raise FileNotFoundError(
                f"{robot_path} not found. Convert the robot first:\n"
                "    python scripts/convert_to_usd.py"
            )
        add_reference_to_stage(str(robot_path), "/World/Robot")
        self._robot = Articulation("/World/Robot", name="gardener_bdx")
        self._world.scene.add(self._robot)
        self._world.reset()

        dof_names = list(self._robot.dof_names)
        self._dof_index = np.array([dof_names.index(n) for n in self.rc.joint_names])
        self._props: dict = {}
        # Photoreal senses on the head (the BDX sensor gimbal). Best-effort:
        # None on failure, and the privileged semantics() channel keeps feeding
        # perception either way.
        self._camera = None
        self._lidar = None
        if self._rtx_sensors:
            self._attach_sensors()

    _HEAD_PRIM = "/World/Robot/head_link"     # matches the URDF head link
    _CAMERA_RES = (128, 96)                   # (W, H); the VLA resizes to 96x96 anyway

    def _attach_sensors(self) -> None:
        """Attach the RGB-D camera and RTX-LiDAR to the head.

        GPU-only template finished against Isaac Sim 4.5+ (``isaacsim.sensors``)
        with the legacy ``omni.isaac.sensor`` fallback; sensor APIs drift across
        versions, so each attachment is individually non-fatal — a mismatch
        prints the fix location and the twin runs on ``semantics()`` meanwhile."""
        try:
            try:  # Isaac Sim >= 4.5
                from isaacsim.sensors.camera import Camera
            except Exception:  # pragma: no cover - older Isaac Sim
                from omni.isaac.sensor import Camera

            cam = Camera(
                prim_path=f"{self._HEAD_PRIM}/rgbd_camera",
                translation=np.array([0.06, 0.0, 0.03]),  # just ahead of the face
                resolution=self._CAMERA_RES,
            )
            cam.initialize()
            cam.add_distance_to_image_plane_to_frame()    # depth channel
            self._camera = cam
        except Exception as e:  # pragma: no cover - depends on installed Isaac
            print(f"[isaac] head camera not attached ({type(e).__name__}: {e}) — "
                  "finish the seam in IsaacGreenhouse._attach_sensors (docs/ISAACSIM.md).")

        try:
            try:  # Isaac Sim >= 4.5
                from isaacsim.sensors.rtx import LidarRtx
            except Exception:  # pragma: no cover - older Isaac Sim
                from omni.isaac.sensor import LidarRtx

            lidar = LidarRtx(
                prim_path=f"{self._HEAD_PRIM}/lidar",
                translation=np.array([0.0, 0.0, 0.08]),   # crown of the head
            )
            lidar.initialize()
            if hasattr(lidar, "add_point_cloud_data_to_frame"):
                lidar.add_point_cloud_data_to_frame()
            self._lidar = lidar
        except Exception as e:  # pragma: no cover - depends on installed Isaac
            print(f"[isaac] RTX-LiDAR not attached ({type(e).__name__}: {e}) — "
                  "finish the seam in IsaacGreenhouse._attach_sensors (docs/ISAACSIM.md).")

    # -- task state ------------------------------------------------------- #
    def _init_task_state(self) -> None:
        self.soc = 0.9
        self.water_l = self.rc.water_capacity_liters
        self._valve = 0.0
        xs = self.rng.uniform(1.5, 4.0, self.n_plants) * np.where(self.rng.random(self.n_plants) < 0.5, -1.0, 1.0)
        ys = np.linspace(-self.arena + 2, self.arena - 1, self.n_plants)
        self.plant_xy = np.stack([xs, ys], 1)
        self.plant_dryness = self.rng.uniform(0.2, 0.95, self.n_plants)
        self.dock_xy = np.array([-self.arena + 0.6, -self.arena + 0.6])
        self.human_xy = np.array([0.4, self.arena - 2.0])
        self._human_dir = -1.0
        if self._procedural:
            self._spawn_props()

    def _spawn_props(self) -> None:
        """Procedural greenhouse: visual prims for plants, dock and the human, so
        the task is visible in the GUI even before you author a greenhouse.usd."""
        try:
            from isaacsim.core.api.objects import VisualCuboid, VisualCylinder
        except Exception:  # pragma: no cover
            from omni.isaac.core.objects import VisualCuboid, VisualCylinder

        for i, p in enumerate(self.plant_xy):
            self._props[f"plant_{i}"] = VisualCylinder(
                prim_path=f"/World/plant_{i}", name=f"plant_{i}",
                position=np.array([p[0], p[1], 0.12]), radius=0.06, height=0.24,
                color=self._dryness_color(self.plant_dryness[i]))
        self._props["dock"] = VisualCuboid(
            prim_path="/World/dock", name="dock",
            position=np.array([self.dock_xy[0], self.dock_xy[1], 0.02]),
            scale=np.array([0.4, 0.4, 0.04]), color=np.array([0.9, 0.7, 0.1]))
        self._props["human"] = VisualCylinder(
            prim_path="/World/human", name="human",
            position=np.array([self.human_xy[0], self.human_xy[1], 0.9]),
            radius=0.15, height=1.8, color=np.array([0.9, 0.5, 0.4]))

    @staticmethod
    def _dryness_color(d: float) -> np.ndarray:
        return np.array([float(d), float(1 - d) * 0.7, 0.1])

    # -- RobotIO ---------------------------------------------------------- #
    def reset(self) -> Observation:
        self._world.reset()
        isaac_q = np.empty(self.rc.n_joints)
        isaac_q[self._dof_index] = self.rc.default_joint_positions
        self._robot.set_joint_positions(isaac_q)
        self._robot.set_world_pose(
            position=np.array([0.0, -self.arena + 1.0, self.rc.base_height_nominal]),
            orientation=quat_from_euler(0, 0, np.pi / 2),  # face +y into the greenhouse
        )
        self._init_task_state()
        return self.read()

    def write(self, action: Action) -> None:
        try:
            from isaacsim.core.utils.types import ArticulationAction
        except Exception:  # pragma: no cover
            from omni.isaac.core.utils.types import ArticulationAction

        targets = np.clip(action.joint_position_targets, self.rc.joint_lower, self.rc.joint_upper)
        isaac_targets = np.empty_like(targets)
        isaac_targets[self._dof_index] = targets
        self._robot.apply_action(ArticulationAction(joint_positions=isaac_targets))
        self._valve = float(action.water_valve_lps)

    def step(self) -> None:
        self._world.step(render=self._render)
        self._update_task_state()

    def now(self) -> float:
        return float(self._world.current_time)

    def read(self) -> Observation:
        q = np.asarray(self._robot.get_joint_positions())[self._dof_index]
        qd = np.asarray(self._robot.get_joint_velocities())[self._dof_index]
        pos, quat = self._robot.get_world_pose()
        pos, quat = np.asarray(pos), np.asarray(quat)
        lin = np.asarray(self._robot.get_linear_velocity())
        ang = np.asarray(self._robot.get_angular_velocity())
        return Observation(
            stamp=self.now(),
            imu=ImuReading(quat, ang, np.array([0.0, 0.0, -9.81]), stamp=self.now()),
            joints=JointState(q, qd, np.zeros_like(q), names=self.rc.joint_names, stamp=self.now()),
            camera=self._read_camera(),
            lidar=self._read_lidar(),
            battery=BatteryState(state_of_charge=self.soc,
                                 charging=bool(np.linalg.norm(pos[:2] - self.dock_xy) < 0.6),
                                 stamp=self.now()),
            water=WaterTankState(self.water_l, self.rc.water_capacity_liters,
                                 self._valve > 0, self._valve, stamp=self.now()),
            base_pose=Pose(pos, quat),
            base_twist=Twist(lin, ang),
        )

    def semantics(self) -> Optional[SceneSemantics]:
        base = self._base_xy()
        dets = []
        for i, p in enumerate(self.plant_xy):
            if np.linalg.norm(p - base) <= self.sensor_range:
                dets.append(Detection("plant", i, np.array([p[0], p[1], 0.25]),
                                      {"dryness": float(self.plant_dryness[i])}))
        if np.linalg.norm(self.human_xy - base) <= self.sensor_range + 1.5:
            dets.append(Detection("human", 0, np.array([self.human_xy[0], self.human_xy[1], 0.9])))
        dets.append(Detection("dock", 0, np.array([self.dock_xy[0], self.dock_xy[1], 0.0])))
        return SceneSemantics(stamp=self.now(), detections=tuple(dets))

    # -- internals -------------------------------------------------------- #
    def _base_xy(self) -> np.ndarray:
        pos, _ = self._robot.get_world_pose()
        return np.asarray(pos)[:2]

    def _update_task_state(self) -> None:
        dt = self.dt
        base = self._base_xy()
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

        self.human_xy[1] += self._human_dir * 0.4 * dt
        if not (-self.arena + 2 < self.human_xy[1] < self.arena - 2):
            self._human_dir *= -1
        self._update_props()

    def _update_props(self) -> None:
        if not self._procedural:
            return
        human = self._props.get("human")
        if human is not None:
            human.set_world_pose(position=np.array([self.human_xy[0], self.human_xy[1], 0.9]))
        for i in range(self.n_plants):  # recolor plants by live dryness (eye candy)
            pr = self._props.get(f"plant_{i}")
            if pr is not None:
                try:
                    pr.get_applied_visual_material().set_color(self._dryness_color(self.plant_dryness[i]))
                except Exception:
                    pass

    def _plant_in_cone(self, base, max_dist=0.7, half_angle=0.5):
        pos, quat = self._robot.get_world_pose()
        yaw = yaw_of(np.asarray(quat))
        fwd = np.array([np.cos(yaw), np.sin(yaw)])
        best, best_d = None, max_dist
        for i, p in enumerate(self.plant_xy):
            d = p - base
            dist = float(np.linalg.norm(d))
            if 1e-6 < dist <= best_d and (d / dist) @ fwd >= np.cos(half_angle):
                best, best_d = i, dist
        return best

    def _read_camera(self):
        """Real RGB-D from the head camera; None while the renderer warms up
        (first few frames return empty buffers) or if the seam isn't active."""
        if self._camera is None:
            return None
        try:
            rgba = np.asarray(self._camera.get_rgba())
            if rgba.size == 0:
                return None
            depth = None
            if hasattr(self._camera, "get_depth"):
                d = np.asarray(self._camera.get_depth(), np.float32)
                depth = d if d.size else None
            return CameraFrame(rgb=rgba[..., :3].astype(np.uint8), depth=depth,
                               frame="head_camera", stamp=self.now())
        except Exception:  # pragma: no cover - renderer/version quirk
            return None

    def _read_lidar(self):
        """RTX-LiDAR points in the sensor frame (what the VLA's BEV rasterizer
        expects). Accessors vary across Isaac versions — try each known one."""
        if self._lidar is None:
            return None
        try:
            pts = None
            if hasattr(self._lidar, "get_point_cloud_data"):
                pts = self._lidar.get_point_cloud_data()
            if pts is None and hasattr(self._lidar, "get_current_frame"):
                pts = (self._lidar.get_current_frame() or {}).get("point_cloud")
            if pts is None:
                return None
            pts = np.asarray(pts, np.float32).reshape(-1, 3)
            if pts.shape[0] == 0:
                return None
            return LidarScan(points=pts, frame="lidar", stamp=self.now())
        except Exception:  # pragma: no cover - renderer/version quirk
            return None

    def close(self) -> None:
        app = getattr(self, "_app", None)
        if app is not None:
            app.close()


def _scene_defaults() -> dict:
    try:
        d = load_yaml("greenhouse/scene.yaml")
    except FileNotFoundError:  # pragma: no cover
        d = {}
    return {
        "arena_half": d.get("arena_half", 5.0),
        "n_plants": d.get("n_plants", 6),
        "sensor_range": d.get("sensor_range", 4.0),
    }
