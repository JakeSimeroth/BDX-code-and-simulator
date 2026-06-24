"""RL environment for the locomotion substrate.

Observation/action match the runtime exactly (via
:func:`gardener_bdx.policy.locomotion.locomotion_observation`), so a policy
trained here drops straight into :class:`LocomotionPolicy` on robot. Reward =
imitate a reference BDX gait + track the commanded velocity + stay upright +
be efficient. Domain randomization is applied every reset for Sim2Real.

Needs a full-physics backend (MuJoCo) to learn a real gait; it will *construct*
with the kinematic backend for plumbing tests but won't learn anything there."""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..common.config import RobotConfig
from ..common.math_utils import quat_rotate_inverse, upright_cosine, yaw_of
from ..common.types import Action, LocomotionCommand
from ..policy.locomotion import _CPGGait, locomotion_observation
from ..sim import make_backend
from . import rewards
from .domain_randomization import DomainRandomizer, DRRanges

try:
    import gymnasium as gym
    from gymnasium import spaces

    _BASE = gym.Env
    _GYM = True
except Exception:  # pragma: no cover
    _BASE = object
    _GYM = False


class LocomotionEnv(_BASE):
    metadata = {"render_modes": []}

    def __init__(
        self,
        robot_config: Optional[RobotConfig] = None,
        backend_name: str = "mujoco",
        control_hz: float = 50.0,
        episode_s: float = 20.0,
        domain_rand: bool = True,
        seed: int = 0,
        backend_kwargs: Optional[dict] = None,
    ):
        self.rc = robot_config or RobotConfig.from_yaml()
        self.dt = 1.0 / control_hz
        self.max_steps = int(episode_s * control_hz)
        self.io = make_backend(backend_name, self.rc, self.dt, **(backend_kwargs or {}))
        self.ref = _CPGGait(self.rc)
        self.dr = DomainRandomizer(DRRanges.from_yaml(), seed=seed) if domain_rand else None
        self.w = dict(rewards.DEFAULT_WEIGHTS)
        self.n = self.rc.n_joints

        obs_dim = locomotion_observation(
            self.io.reset(), LocomotionCommand(), self.rc.default_joint_positions,
            np.zeros(self.n), 0.0,
        ).shape[0]
        if _GYM:
            self.observation_space = spaces.Box(-np.inf, np.inf, (obs_dim,), np.float32)
            self.action_space = spaces.Box(-1.0, 1.0, (self.n,), np.float32)
        self._seed = seed

    # -- gym API ---------------------------------------------------------- #
    def reset(self, *, seed=None, options=None):
        obs0 = self.io.reset()
        self._params = self.dr.sample_episode() if self.dr else None
        if self._params is not None:
            self.dr.apply_dynamics(self.io, self._params)
            self.cmd = LocomotionCommand(*self._params.cmd)
        else:
            self.cmd = LocomotionCommand(vx=0.3)
        self._phase = 0.0
        self._last_action = np.zeros(self.n, dtype=np.float32)
        self._steps = 0
        return self._obs(obs0), {}

    def step(self, action: np.ndarray):
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        targets = self.rc.default_joint_positions + self.rc.action_scale * action
        self.io.write(Action(joint_position_targets=targets))
        if self.dr is not None:
            self.dr.maybe_push(self.io.now(), self._params, self.io)
        self.io.step()

        obs = self.io.read()
        speed = float(np.hypot(self.cmd.vx, self.cmd.vy) + abs(self.cmd.wz))
        self._phase = (self._phase + (0.0 if speed < 1e-3 else min(1.0 + 1.2 * speed, 2.4)) * self.dt) % 1.0
        reward, terminated = self._reward(obs, action)
        self._last_action = action
        self._steps += 1
        truncated = self._steps >= self.max_steps
        return self._obs(obs), float(reward), bool(terminated), bool(truncated), {}

    # -- internals -------------------------------------------------------- #
    def _obs(self, observation):
        vec = locomotion_observation(observation, self.cmd, self.rc.default_joint_positions,
                                     self._last_action, self._phase)
        if self.dr is not None:
            vec = self.dr.add_obs_noise(vec)
        return vec.astype(np.float32)

    def _reward(self, obs, action):
        q = obs.joints.positions
        q_ref = self.ref.step(self.cmd, self._phase, 1.0)
        up = upright_cosine(obs.imu.orientation)

        # Base linear velocity in the body frame for command tracking.
        if obs.base_twist is not None and obs.base_pose is not None:
            v_body = quat_rotate_inverse(obs.base_pose.orientation,
                                         np.array([*obs.base_twist.linear[:2], 0.0]))
            wz = obs.base_twist.angular[2]
        else:
            v_body, wz = np.zeros(3), 0.0
        base_z = obs.base_pose.position[2] if obs.base_pose is not None else self.rc.base_height_nominal

        w = self.w
        r = (
            w["velocity"] * rewards.velocity_tracking(self.cmd.vx, self.cmd.vy, self.cmd.wz,
                                                      v_body[0], v_body[1], wz)
            + w["imitation"] * rewards.imitation(q, q_ref)
            + w["upright"] * rewards.upright(up)
            + w["height"] * rewards.height_keeping(base_z, self.rc.base_height_nominal)
            + w["energy"] * rewards.energy_penalty(obs.joints.torques, obs.joints.velocities)
            + w["action_rate"] * rewards.action_rate_penalty(action, self._last_action)
            + w["joint_limit"] * rewards.joint_limit_penalty(q, self.rc.joint_lower, self.rc.joint_upper)
            + w["alive"]
        )
        # Terminate on a fall (lost balance or collapsed height).
        terminated = bool(up < 0.4 or base_z < 0.15)
        if terminated:
            r -= 5.0
        return r, terminated

    def close(self):
        self.io.close()
