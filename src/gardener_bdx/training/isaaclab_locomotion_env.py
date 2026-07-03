"""Isaac Lab parallel locomotion environment (Direct workflow).

This is the *scaled* counterpart of :class:`gardener_bdx.training.locomotion_env`
— thousands of greenhouses on one GPU — and the place the fluid BDX walk is
actually learned. It is deliberately faithful to the runtime:

  * the observation is byte-for-byte the layout of
    :func:`gardener_bdx.policy.locomotion.locomotion_observation`, and
  * the reward mirrors :mod:`gardener_bdx.training.rewards`
    (imitation + velocity tracking + upright/height + regularizers),

so a policy trained here drops into ``LocomotionPolicy`` after export with no
observation/reward drift. Domain randomization (command curriculum, base pushes,
+ Isaac Lab material/mass events) closes Sim2Real.

Run it under Isaac Lab's Python (it imports the Omniverse stack at module load):

    # from an Isaac Lab checkout:
    python -m gardener_bdx.training.train_isaaclab --num_envs 4096 --headless

Requires a USD of the robot at ``models/robot/gardener_bdx.usd`` (convert the
MJCF/URDF once — see docs/ISAACLAB.md). This module is never imported by the
core package; it is Isaac-only by design."""

from __future__ import annotations

import math

import torch

import isaaclab.sim as sim_utils
import isaaclab.envs.mdp as mdp
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.managers import EventTermCfg, SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from ..common.config import RobotConfig, repo_root

# Robot description is the single source of truth, shared with every other backend.
_RC = RobotConfig.from_yaml()
_JOINTS = list(_RC.joint_names)
_N = _RC.n_joints
_OBS_DIM = 3 + 3 + _N + _N + _N + 3 + 2  # proj-g, ang-vel, q-def, qdot, last_a, cmd, clock


def _per_joint(values):
    return {n: float(v) for n, v in zip(_JOINTS, values)}


@configclass
class EventCfg:
    """Isaac Lab event-manager domain randomization. ``startup`` terms give the
    4096-env population a spread of dynamics (each robot keeps its draw);
    ``interval`` shoves test recovery. Switch material/mass to ``mode="reset"``
    for per-episode re-randomization (and apply reset events in ``_reset_idx``)."""

    physics_material = EventTermCfg(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.6, 1.4),
            "dynamic_friction_range": (0.5, 1.2),
            "restitution_range": (0.0, 0.1),
            "num_buckets": 64,
        },
    )
    base_mass = EventTermCfg(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "mass_distribution_params": (-0.5, 0.8),  # kg added to the torso (payload/battery spread)
            "operation": "add",
        },
    )
    push_robot = EventTermCfg(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(3.0, 7.0),
        params={"velocity_range": {"x": (-0.6, 0.6), "y": (-0.6, 0.6), "yaw": (-0.4, 0.4)}},
    )


@configclass
class GardenerBdxFlatEnvCfg(DirectRLEnvCfg):
    # --- timing: 200 Hz physics, decimation 4 => 50 Hz control (== runtime) ---
    decimation = 4
    episode_length_s = 20.0
    action_scale = _RC.action_scale
    action_space = _N
    observation_space = _OBS_DIM
    state_space = 0

    sim: SimulationCfg = SimulationCfg(dt=1.0 / 200.0, render_interval=decimation)

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
    )

    robot: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(repo_root() / "models" / "robot" / "gardener_bdx.usd"),
            activate_contact_sensors=True,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, _RC.base_height_nominal),
            joint_pos=_per_joint(_RC.default_joint_positions),
            joint_vel={".*": 0.0},
        ),
        actuators={
            "all": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                stiffness=_per_joint(_RC.kp),
                damping=_per_joint(_RC.kd),
                effort_limit_sim=_per_joint(_RC.joint_torque_limit),
                velocity_limit_sim=_per_joint(_RC.joint_velocity_limit),
            ),
        },
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=2.5)
    events: EventCfg = EventCfg()  # domain randomization via Isaac Lab's event manager

    # --- command sampling ranges (the velocity curriculum) ---
    cmd_vx = (-0.4, 0.6)
    cmd_vy = (-0.3, 0.3)
    cmd_wz = (-1.0, 1.0)
    resample_commands_s = 5.0

    # --- reward weights (mirror training/rewards.DEFAULT_WEIGHTS) ---
    w_velocity = 1.0
    w_imitation = 1.0
    w_upright = 0.5
    w_height = 0.5
    w_energy = -2e-3
    w_action_rate = -0.05
    w_alive = 0.2
    fall_penalty = -5.0


class GardenerBdxLocomotionEnv(DirectRLEnv):
    cfg: GardenerBdxFlatEnvCfg

    def __init__(self, cfg: GardenerBdxFlatEnvCfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        dev = self.device
        self._default_q = torch.tensor(_RC.default_joint_positions, device=dev, dtype=torch.float32)
        self._lower = torch.tensor(_RC.joint_lower, device=dev, dtype=torch.float32)
        self._upper = torch.tensor(_RC.joint_upper, device=dev, dtype=torch.float32)

        self._commands = torch.zeros(self.num_envs, 3, device=dev)
        self._actions = torch.zeros(self.num_envs, _N, device=dev)
        self._last_actions = torch.zeros(self.num_envs, _N, device=dev)
        self._phase = torch.zeros(self.num_envs, device=dev)

        # Reference-gait shape tensors (vectorized _CPGGait).
        side = torch.zeros(_N, device=dev)
        amp = torch.zeros(_N, device=dev)
        knee = torch.zeros(_N, dtype=torch.bool, device=dev)
        for i, nm in enumerate(_JOINTS):
            if "right" in nm:
                side[i] = math.pi
            if "hip_pitch" in nm:
                amp[i] = 0.30
            elif "knee" in nm:
                amp[i] = 0.45
                knee[i] = True
            elif "ankle_pitch" in nm:
                amp[i] = 0.15
        self._side, self._amp, self._knee = side, amp, knee
        self._steps_to_resample = int(self.cfg.resample_commands_s / (self.cfg.sim.dt * self.cfg.decimation))
        self._cmd_timer = torch.zeros(self.num_envs, dtype=torch.long, device=dev)

    # -- scene ------------------------------------------------------------ #
    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot)
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self.scene.clone_environments(copy_from_source=False)
        self.scene.articulations["robot"] = self.robot
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        light = sim_utils.DomeLightCfg(intensity=2000.0)
        light.func("/World/Light", light)

    # -- step -------------------------------------------------------------- #
    def _pre_physics_step(self, actions: torch.Tensor):
        # Clip to [-1, 1] to match the numpy RL env and the runtime
        # LocomotionPolicy exactly — otherwise the exported policy would see a
        # different action convention on the robot.
        self._actions = actions.clamp(-1.0, 1.0)
        # Advance the gait phase per env, frozen when commanded to stand still.
        speed = self._commands[:, :2].norm(dim=-1) + self._commands[:, 2].abs()
        freq = torch.where(speed < 1e-3, torch.zeros_like(speed), (1.0 + 1.2 * speed).clamp(max=2.4))
        dt = self.cfg.sim.dt * self.cfg.decimation
        self._phase = torch.remainder(self._phase + freq * dt, 1.0)
        # Velocity-command curriculum: resample each env's command periodically.
        self._cmd_timer += 1
        due = (self._cmd_timer >= self._steps_to_resample).nonzero(as_tuple=False).flatten()
        if due.numel() > 0:
            self._resample_commands(due)
            self._cmd_timer[due] = 0

    def _apply_action(self):
        targets = self._default_q + self.cfg.action_scale * self._actions
        targets = torch.clamp(targets, self._lower, self._upper)
        self.robot.set_joint_position_target(targets)

    # -- observations ------------------------------------------------------ #
    def _reference_gait(self) -> torch.Tensor:
        speed = (self._commands[:, :2].norm(dim=-1) + self._commands[:, 2].abs()).clamp(max=1.0)
        gait = torch.sin(2 * math.pi * self._phase.unsqueeze(-1) + self._side)
        motion = self._amp * gait
        motion = torch.where(self._knee, self._amp * gait.clamp(min=0.0), motion)
        return self._default_q + speed.unsqueeze(-1) * motion

    def _get_observations(self) -> dict:
        d = self.robot.data
        clock = torch.stack([torch.sin(2 * math.pi * self._phase), torch.cos(2 * math.pi * self._phase)], dim=-1)
        obs = torch.cat(
            [
                d.projected_gravity_b,                 # 3
                d.root_ang_vel_b,                      # 3
                d.joint_pos - self._default_q,         # n
                d.joint_vel,                           # n
                self._last_actions,                    # n
                self._commands,                        # 3
                clock,                                 # 2
            ],
            dim=-1,
        )
        self._last_actions = self._actions.clone()
        return {"policy": obs}

    # -- rewards ----------------------------------------------------------- #
    def _get_rewards(self) -> torch.Tensor:
        d = self.robot.data
        c = self.cfg
        lin = d.root_lin_vel_b
        upright = (-d.projected_gravity_b[:, 2]).clamp(0.0, 1.0)
        height = d.root_pos_w[:, 2]
        q_ref = self._reference_gait()

        vel = torch.exp(-((self._commands[:, :2] - lin[:, :2]) ** 2).sum(-1) / 0.25) \
            + 0.5 * torch.exp(-((self._commands[:, 2] - d.root_ang_vel_b[:, 2]) ** 2) / 0.25)
        imit = torch.exp(-((d.joint_pos - q_ref) ** 2).mean(-1) / 0.5)
        h = torch.exp(-((height - _RC.base_height_nominal) ** 2) / 0.02)
        energy = (d.applied_torque * d.joint_vel).abs().mean(-1)
        a_rate = ((self._actions - self._last_actions) ** 2).mean(-1)

        r = (
            c.w_velocity * vel + c.w_imitation * imit + c.w_upright * upright + c.w_height * h
            + c.w_energy * energy + c.w_action_rate * a_rate + c.w_alive
        ) * self.step_dt
        return r

    # -- termination ------------------------------------------------------- #
    def _get_dones(self):
        d = self.robot.data
        fell = (-d.projected_gravity_b[:, 2] < 0.4) | (d.root_pos_w[:, 2] < 0.15)
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return fell, time_out

    # -- reset ------------------------------------------------------------- #
    def _reset_idx(self, env_ids):
        if env_ids is None or len(env_ids) == 0:
            return
        super()._reset_idx(env_ids)
        d = self.robot.data
        n = len(env_ids)
        # Reset-time randomization: small joint jitter + a random initial heading,
        # so the policy must track omnidirectional commands from any yaw.
        joint_pos = d.default_joint_pos[env_ids].clone()
        joint_pos += torch.empty_like(joint_pos).uniform_(-0.05, 0.05)
        joint_vel = d.default_joint_vel[env_ids].clone()
        root_state = d.default_root_state[env_ids].clone()
        root_state[:, :3] += self.scene.env_origins[env_ids]
        yaw = torch.empty(n, device=self.device).uniform_(-math.pi, math.pi)
        root_state[:, 3] = torch.cos(yaw / 2)
        root_state[:, 4:6] = 0.0
        root_state[:, 6] = torch.sin(yaw / 2)
        self.robot.write_root_pose_to_sim(root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        self._phase[env_ids] = torch.rand(n, device=self.device)
        self._last_actions[env_ids] = 0.0
        self._cmd_timer[env_ids] = 0
        self._resample_commands(env_ids)

    def _resample_commands(self, env_ids):
        n = len(env_ids)
        rng = lambda lo, hi: torch.rand(n, device=self.device) * (hi - lo) + lo  # noqa: E731
        self._commands[env_ids, 0] = rng(*self.cfg.cmd_vx)
        self._commands[env_ids, 1] = rng(*self.cfg.cmd_vy)
        self._commands[env_ids, 2] = rng(*self.cfg.cmd_wz)
