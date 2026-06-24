"""NVIDIA Isaac Sim / Isaac Lab backend — the primary digital twin.

Why Isaac is the recommended twin for this project (see docs/SIMULATION.md):
  * **Photorealistic RTX sensors** — the VLA is vision-first, so we need camera
    frames and RTX-LiDAR returns that look like the real greenhouse. Domain
    randomization over lighting/materials/foliage is what closes Sim2Real.
  * **Massively parallel RL (Isaac Lab)** — thousands of cloned greenhouses on
    one GPU to train the locomotion substrate and to RL-finetune the VLA.
  * **Same stack as the robot** — USD assets, the Jetson deployment target, and
    GR00T N1 / Cosmos world-model tooling all live in this ecosystem, so the
    twin and the robot share one pipeline.

This module is the *single-environment* inference/eval twin that satisfies the
:class:`RobotIO` contract (the parallel training envs live in
:mod:`gardener_bdx.training`). Isaac's Python API is heavy and version-specific,
so the integration points below are marked clearly rather than hard-coded; fill
them against your installed Isaac Lab version. Requires the Isaac Sim runtime —
it is never imported unless this backend is explicitly constructed."""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..common.config import RobotConfig
from ..common.types import (
    Action,
    BatteryState,
    Observation,
    SceneSemantics,
    WaterTankState,
)
from ..interfaces.robot_io import RobotIO


class IsaacGreenhouse(RobotIO):
    """RobotIO over an Isaac Sim stage. Construction boots the simulation app, so
    do it once, early, before importing other Omniverse-dependent modules."""

    def __init__(
        self,
        robot_config: RobotConfig,
        control_dt: float,
        usd_robot: str = "models/robot/gardener_bdx.usd",
        usd_scene: str = "models/scenes/greenhouse.usd",
        headless: bool = True,
        device: str = "cuda",
    ):
        super().__init__(robot_config, control_dt)
        self.rc = robot_config
        self.device = device
        self._boot(headless)
        self._build_stage(usd_robot, usd_scene)
        self.soc = 0.9
        self.water_l = robot_config.water_capacity_liters
        self._valve = 0.0

    def is_simulation(self) -> bool:
        return True

    # -- boot / build (integration seams) --------------------------------- #
    def _boot(self, headless: bool) -> None:
        try:
            from isaacsim import SimulationApp  # Isaac Sim >= 4.x
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "Isaac backend requires the Isaac Sim runtime (isaacsim / isaaclab). "
                "Install via NVIDIA Omniverse / the Isaac Lab instructions, then run "
                "this process under Isaac's python. See docs/SIMULATION.md.\n"
                f"(original error: {e})"
            )
        # SimulationApp MUST be created before any omni.* import.
        self._app = SimulationApp({"headless": headless})

    def _build_stage(self, usd_robot: str, usd_scene: str) -> None:
        # Imports are valid only after SimulationApp exists.
        from omni.isaac.core import World  # noqa: WPS433
        from omni.isaac.core.articulations import Articulation  # noqa: WPS433
        from omni.isaac.core.utils.stage import add_reference_to_stage  # noqa: WPS433

        self._world = World(physics_dt=self.dt, rendering_dt=self.dt)
        add_reference_to_stage(usd_scene, "/World/Greenhouse")
        add_reference_to_stage(usd_robot, "/World/Robot")
        self._robot = Articulation("/World/Robot", name="gardener_bdx")
        self._world.scene.add(self._robot)
        self._world.reset()
        # Reorder Isaac's DOF index to our canonical joint order once.
        dof_names = list(self._robot.dof_names)
        self._dof_index = np.array([dof_names.index(n) for n in self.rc.joint_names])
        # TODO: attach RTX camera + RTX-LiDAR prims on the head, and a contact
        # sensor per foot; cache their handles here for read().
        self._camera = None
        self._lidar = None

    # -- RobotIO ---------------------------------------------------------- #
    def reset(self) -> Observation:
        self._world.reset()
        self._robot.set_joint_positions(
            self.rc.default_joint_positions[np.argsort(self._dof_index)]
        )
        self.soc = 0.9
        self.water_l = self.rc.water_capacity_liters
        self._valve = 0.0
        return self.read()

    def write(self, action: Action) -> None:
        from omni.isaac.core.utils.types import ArticulationAction  # noqa: WPS433

        targets = np.clip(action.joint_position_targets, self.rc.joint_lower, self.rc.joint_upper)
        # Map canonical order -> Isaac DOF order.
        isaac_targets = np.empty_like(targets)
        isaac_targets[self._dof_index] = targets
        self._robot.apply_action(ArticulationAction(joint_positions=isaac_targets))
        self._valve = float(action.water_valve_lps)

    def step(self) -> None:
        self._world.step(render=self._camera is not None)
        # Battery/water/plant bookkeeping mirrors the other backends; omitted here
        # for brevity — wire to the USD plant prims' moisture attributes.

    def now(self) -> float:
        return float(self._world.current_time)

    def read(self) -> Observation:
        from ..common.types import ImuReading, JointState, Pose, Twist  # local import

        q = np.asarray(self._robot.get_joint_positions())[self._dof_index]
        qd = np.asarray(self._robot.get_joint_velocities())[self._dof_index]
        pos, quat = self._robot.get_world_pose()  # (xyz, wxyz)
        lin = np.asarray(self._robot.get_linear_velocity())
        ang = np.asarray(self._robot.get_angular_velocity())
        imu = ImuReading(np.asarray(quat), ang, np.array([0.0, 0.0, -9.81]), stamp=self.now())
        joints = JointState(q, qd, np.zeros_like(q), names=self.rc.joint_names, stamp=self.now())
        return Observation(
            stamp=self.now(),
            imu=imu,
            joints=joints,
            camera=self._read_camera(),   # RTX render -> CameraFrame
            lidar=self._read_lidar(),     # RTX-LiDAR -> LidarScan
            battery=BatteryState(state_of_charge=self.soc, stamp=self.now()),
            water=WaterTankState(self.water_l, self.rc.water_capacity_liters,
                                 self._valve > 0, self._valve, stamp=self.now()),
            base_pose=Pose(np.asarray(pos), np.asarray(quat)),
            base_twist=Twist(lin, ang),
        )

    def semantics(self) -> Optional[SceneSemantics]:
        # In Isaac, prefer training the real PlantDetector on RTX frames. Ground
        # truth is available via the replicator/semantics API if you want to
        # bootstrap; return it here mirroring the other backends if so.
        return None

    def _read_camera(self):
        if self._camera is None:
            return None
        from ..common.types import CameraFrame

        rgb = self._camera.get_rgba()[..., :3]
        depth = self._camera.get_depth()
        return CameraFrame(rgb=np.asarray(rgb, np.uint8), depth=np.asarray(depth, np.float32),
                           frame="head_camera", stamp=self.now())

    def _read_lidar(self):
        if self._lidar is None:
            return None
        from ..common.types import LidarScan

        return LidarScan(points=np.asarray(self._lidar.get_point_cloud_data()).reshape(-1, 3),
                         frame="lidar", stamp=self.now())

    def close(self) -> None:
        app = getattr(self, "_app", None)
        if app is not None:
            app.close()
