"""Domain randomization — the core Sim2Real mechanism.

We randomize dynamics, sensing and timing every episode so the policy learns a
*robust* behavior that survives the reality gap instead of overfitting one
idealized simulator. Ranges live in ``configs/sim/domain_randomization.yaml``."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class DRRanges:
    mass_scale: tuple[float, float] = (0.85, 1.15)
    friction: tuple[float, float] = (0.6, 1.4)
    kp_scale: tuple[float, float] = (0.8, 1.2)
    kd_scale: tuple[float, float] = (0.8, 1.2)
    latency_steps: tuple[int, int] = (0, 2)  # actuation/observation delay (ticks)
    obs_noise_std: float = 0.02
    imu_bias_std: float = 0.01
    push_interval_s: tuple[float, float] = (3.0, 7.0)
    push_velocity: float = 0.6  # m/s impulse magnitude on the base
    cmd_vx: tuple[float, float] = (-0.4, 0.6)
    cmd_vy: tuple[float, float] = (-0.3, 0.3)
    cmd_wz: tuple[float, float] = (-1.0, 1.0)

    @staticmethod
    def from_yaml(path: str = "sim/domain_randomization.yaml") -> "DRRanges":
        from ..common.config import load_yaml

        try:
            d = load_yaml(path)
        except FileNotFoundError:
            return DRRanges()
        r = DRRanges()
        for k, v in d.items():
            if hasattr(r, k):
                setattr(r, k, tuple(v) if isinstance(v, list) else v)
        return r


@dataclass
class EpisodeParams:
    mass_scale: float
    friction: float
    kp_scale: float
    kd_scale: float
    latency_steps: int
    cmd: np.ndarray  # [vx, vy, wz]
    next_push_s: float


class DomainRandomizer:
    def __init__(self, ranges: Optional[DRRanges] = None, seed: int = 0):
        self.r = ranges or DRRanges()
        self.rng = np.random.default_rng(seed)

    def sample_episode(self) -> EpisodeParams:
        u = self.rng.uniform
        return EpisodeParams(
            mass_scale=u(*self.r.mass_scale),
            friction=u(*self.r.friction),
            kp_scale=u(*self.r.kp_scale),
            kd_scale=u(*self.r.kd_scale),
            latency_steps=int(self.rng.integers(self.r.latency_steps[0], self.r.latency_steps[1] + 1)),
            cmd=np.array([u(*self.r.cmd_vx), u(*self.r.cmd_vy), u(*self.r.cmd_wz)]),
            next_push_s=u(*self.r.push_interval_s),
        )

    def add_obs_noise(self, obs_vec: np.ndarray) -> np.ndarray:
        return obs_vec + self.rng.normal(0.0, self.r.obs_noise_std, size=obs_vec.shape)

    def maybe_push(self, t: float, params: EpisodeParams, backend) -> bool:
        """Apply a random base-velocity impulse to test recovery. Best-effort:
        only acts on backends exposing the necessary handle (MuJoCo)."""
        if t < params.next_push_s:
            return False
        params.next_push_s = t + self.rng.uniform(*self.r.push_interval_s)
        data = getattr(backend, "data", None)
        if data is not None and hasattr(backend, "_root_dadr"):
            theta = self.rng.uniform(-np.pi, np.pi)
            data.qvel[backend._root_dadr : backend._root_dadr + 2] += (
                self.r.push_velocity * np.array([np.cos(theta), np.sin(theta)])
            )
        return True

    def apply_dynamics(self, backend, params: EpisodeParams) -> None:
        """Apply per-episode dynamics randomization to a MuJoCo backend. No-op on
        backends without the handles (kept generic on purpose)."""
        model = getattr(backend, "model", None)
        if model is None:
            return
        if not hasattr(self, "_nominal_mass"):
            self._nominal_mass = model.body_mass.copy()
            self._nominal_gain = model.actuator_gainprm.copy()
            self._nominal_fric = model.geom_friction.copy()
        model.body_mass[:] = self._nominal_mass * params.mass_scale
        model.actuator_gainprm[:] = self._nominal_gain
        model.actuator_gainprm[:, 0] *= params.kp_scale
        model.geom_friction[:] = self._nominal_fric
        model.geom_friction[:, 0] *= params.friction
