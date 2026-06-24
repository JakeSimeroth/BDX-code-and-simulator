"""System 0 — the locomotion substrate.

A small feed-forward policy maps proprioception + a velocity command to joint
position targets at 50-200 Hz. This is the layer that produces the *fluid BDX
walk*; the VLA only ever hands it a velocity command, never raw joint angles.

Two deployment-friendly properties:

  * **numpy inference.** Weights load from a plain ``.npz`` and run with numpy —
    no torch needed on the robot. The trainer (PyTorch / Isaac Lab) exports to
    this format, decoupling the training stack from the runtime. This is exactly
    the seam you want for Sim2Real: train big, ship small.
  * **CPG fallback.** Before any policy is trained, a central-pattern-generator
    produces a plausible stepping gait so the whole stack runs end-to-end in the
    digital twin from day one. The real RL policy replaces it transparently.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..common.config import RobotConfig
from ..common.math_utils import projected_gravity
from ..common.types import Action, LocomotionCommand, Observation


def locomotion_observation(
    obs: Observation,
    cmd: LocomotionCommand,
    default_q: np.ndarray,
    last_action: np.ndarray,
    phase: float,
) -> np.ndarray:
    """Canonical locomotion observation vector. Single source of truth shared by
    runtime inference and the RL training environment (so sim and deploy agree
    to the bit)."""
    clock = np.array([np.sin(2 * np.pi * phase), np.cos(2 * np.pi * phase)])
    return np.concatenate(
        [
            projected_gravity(obs.imu.orientation),  # 3
            obs.imu.angular_velocity,  # 3
            obs.joints.positions - default_q,  # n
            obs.joints.velocities,  # n
            last_action,  # n
            np.array([cmd.vx, cmd.vy, cmd.wz]),  # 3
            clock,  # 2
        ]
    ).astype(np.float32)


class MLP:
    """Minimal numpy MLP for fast, dependency-free inference. Layer weights are
    stored as ``w0,b0,w1,b1,…`` in an ``.npz``; activation is configurable."""

    def __init__(self, weights: list[tuple[np.ndarray, np.ndarray]], activation: str = "elu"):
        self.weights = weights
        self.activation = activation

    def _act(self, x: np.ndarray) -> np.ndarray:
        if self.activation == "elu":
            return np.where(x > 0, x, np.expm1(np.clip(x, -20, 0)))
        if self.activation == "tanh":
            return np.tanh(x)
        return np.maximum(x, 0.0)  # relu

    def __call__(self, x: np.ndarray) -> np.ndarray:
        h = x
        for i, (w, b) in enumerate(self.weights):
            h = h @ w.T + b
            if i < len(self.weights) - 1:
                h = self._act(h)
        return h

    @staticmethod
    def load_npz(path: str) -> "MLP":
        data = np.load(path)
        layers, i = [], 0
        while f"w{i}" in data:
            layers.append((data[f"w{i}"], data[f"b{i}"]))
            i += 1
        act = str(data["activation"]) if "activation" in data else "elu"
        return MLP(layers, activation=act)

    @staticmethod
    def save_npz(path: str, weights, activation: str = "elu") -> None:
        out = {"activation": activation}
        for i, (w, b) in enumerate(weights):
            out[f"w{i}"] = np.asarray(w, dtype=np.float32)
            out[f"b{i}"] = np.asarray(b, dtype=np.float32)
        np.savez(path, **out)


class LocomotionPolicy:
    def __init__(self, robot_config: RobotConfig, policy_path: Optional[str] = None):
        self.cfg = robot_config
        self.default_q = robot_config.default_joint_positions.copy()
        self.n = robot_config.n_joints
        self.mlp: Optional[MLP] = None
        if policy_path:
            try:
                self.mlp = MLP.load_npz(policy_path)
            except (FileNotFoundError, OSError):
                self.mlp = None  # fall back to CPG until a policy is trained
        self._cpg = _CPGGait(robot_config)
        self.reset()

    def reset(self) -> None:
        self._last_action = np.zeros(self.n, dtype=np.float32)
        self._phase = 0.0

    @property
    def using_learned_policy(self) -> bool:
        return self.mlp is not None

    def act(self, obs: Observation, cmd: LocomotionCommand, dt: float) -> Action:
        speed = float(np.hypot(cmd.vx, cmd.vy) + abs(cmd.wz))
        # Advance the gait phase; freeze it when commanded to stand still.
        freq = 0.0 if speed < 1e-3 else min(1.0 + 1.2 * speed, 2.4)
        self._phase = (self._phase + freq * dt) % 1.0

        if self.mlp is not None:
            obs_vec = locomotion_observation(obs, cmd, self.default_q, self._last_action, self._phase)
            a = np.clip(self.mlp(obs_vec), -1.0, 1.0)
            self._last_action = a
            targets = self.default_q + self.cfg.action_scale * a
        else:
            targets = self._cpg.step(cmd, self._phase, freq)

        targets = self._apply_posture(targets, cmd)
        targets = np.clip(targets, self.cfg.joint_lower, self.cfg.joint_upper)
        return Action(joint_position_targets=targets)

    def _apply_posture(self, targets: np.ndarray, cmd: LocomotionCommand) -> np.ndarray:
        """Map the VLA's stylistic modifiers (crouch, head look) onto named
        joints when the robot has them. Pure body-language; balance is the
        policy's job."""
        names = self.cfg.joint_names
        for i, nm in enumerate(names):
            if "knee" in nm:
                targets[i] += 0.9 * cmd.body_height
            elif "hip_pitch" in nm:
                targets[i] -= 0.45 * cmd.body_height
            elif "neck_yaw" in nm or "head_yaw" in nm:
                targets[i] += cmd.look_yaw
            elif "head_pitch" in nm or "neck_pitch" in nm:
                targets[i] += cmd.look_pitch
        return targets


class _CPGGait:
    """Open-loop central pattern generator. Not a controller — just enough
    coordinated motion to make the droid look alive in the twin until the RL
    policy takes over."""

    def __init__(self, cfg: RobotConfig):
        self.cfg = cfg
        # Per-joint side offset (left vs right legs are anti-phase).
        self.side = np.zeros(cfg.n_joints)
        self.amp = np.zeros(cfg.n_joints)
        for i, nm in enumerate(cfg.joint_names):
            if "right" in nm:
                self.side[i] = np.pi
            if "hip_pitch" in nm:
                self.amp[i] = 0.30
            elif "knee" in nm:
                self.amp[i] = 0.45
            elif "ankle_pitch" in nm:
                self.amp[i] = 0.15

    def step(self, cmd: LocomotionCommand, phase: float, freq: float) -> np.ndarray:
        speed = float(np.hypot(cmd.vx, cmd.vy) + abs(cmd.wz))
        gait = np.sin(2 * np.pi * phase + self.side)
        # Knees only flex (rectified) for a more natural step.
        knees = np.array(["knee" in nm for nm in self.cfg.joint_names])
        motion = self.amp * gait
        motion[knees] = self.amp[knees] * np.clip(gait[knees], 0.0, None)
        return self.cfg.default_joint_positions + min(speed, 1.0) * motion
