"""Reward terms for locomotion RL.

The headline term is the **imitation reward** from Disney's BDX work (and the
Open Duck Mini reproduction): track a reference gait so the learned motion is
natural and BDX-like, while velocity-tracking makes it go where commanded and
the regularizers keep it efficient and Sim2Real-friendly. Weights live in the
env / config so they are easy to sweep."""

from __future__ import annotations

import numpy as np


def velocity_tracking(cmd_vx: float, cmd_vy: float, cmd_wz: float,
                      vx: float, vy: float, wz: float, sigma: float = 0.25) -> float:
    lin_err = (cmd_vx - vx) ** 2 + (cmd_vy - vy) ** 2
    yaw_err = (cmd_wz - wz) ** 2
    return float(np.exp(-lin_err / sigma) + 0.5 * np.exp(-yaw_err / sigma))


def imitation(q: np.ndarray, q_ref: np.ndarray, sigma: float = 0.5) -> float:
    """Match the reference gait pose. The single most important shaping term for
    a natural, transferable walk."""
    return float(np.exp(-np.mean((q - q_ref) ** 2) / sigma))


def upright(upright_cos: float) -> float:
    return float(np.clip(upright_cos, 0.0, 1.0))


def height_keeping(base_z: float, nominal: float, sigma: float = 0.02) -> float:
    return float(np.exp(-((base_z - nominal) ** 2) / sigma))


def energy_penalty(tau: np.ndarray, qd: np.ndarray) -> float:
    return float(np.mean(np.abs(tau * qd)))


def action_rate_penalty(a: np.ndarray, a_prev: np.ndarray) -> float:
    return float(np.mean((a - a_prev) ** 2))


def joint_limit_penalty(q: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    over = np.clip(q - upper, 0, None) + np.clip(lower - q, 0, None)
    return float(np.mean(over))


DEFAULT_WEIGHTS = {
    "velocity": 1.0,
    "imitation": 1.0,
    "upright": 0.5,
    "height": 0.5,
    "energy": -2e-3,
    "action_rate": -0.05,
    "joint_limit": -1.0,
    "alive": 0.2,
}
