"""Small, dependency-light math helpers used across perception, policy and sim.

Quaternions are ``(w, x, y, z)``. Functions are written to be allocation-cheap
and to work the same way in sim and on hardware (no SciPy dependency so they run
on a minimal Jetson image)."""

from __future__ import annotations

import numpy as np

GRAVITY = np.array([0.0, 0.0, -9.81])


def quat_normalize(q: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return q / n


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array([w, -x, -y, -z])


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ]
    )


def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector ``v`` by quaternion ``q`` (body->world if q is body pose)."""
    qv = np.array([0.0, v[0], v[1], v[2]])
    return quat_mul(quat_mul(q, qv), quat_conjugate(q))[1:]


def quat_rotate_inverse(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate ``v`` by the inverse of ``q`` (world->body if q is body pose)."""
    return quat_rotate(quat_conjugate(q), v)


def projected_gravity(q: np.ndarray) -> np.ndarray:
    """Gravity unit vector expressed in the body frame — the single most useful
    orientation feature for a balance/locomotion policy. Equals [0,0,-1] when
    perfectly upright."""
    g = quat_rotate_inverse(q, np.array([0.0, 0.0, -1.0]))
    return g


def upright_cosine(q: np.ndarray) -> float:
    """cos(tilt): +1 fully upright, 0 on its side, -1 inverted."""
    body_up_in_world = quat_rotate(q, np.array([0.0, 0.0, 1.0]))
    return float(body_up_in_world[2])


def quat_from_euler(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    return np.array(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ]
    )


def euler_from_quat(q: np.ndarray) -> np.ndarray:
    """Return [roll, pitch, yaw]."""
    w, x, y, z = q
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    sinp = 2 * (w * y - z * x)
    pitch = np.arcsin(np.clip(sinp, -1.0, 1.0))
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return np.array([roll, pitch, yaw])


def yaw_of(q: np.ndarray) -> float:
    return float(euler_from_quat(q)[2])


def wrap_to_pi(a: float) -> float:
    return float((a + np.pi) % (2 * np.pi) - np.pi)


def clip_vec(v: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    return np.minimum(np.maximum(v, lo), hi)


class ExponentialFilter:
    """First-order low-pass for smoothing commands/observations. ``alpha`` in
    (0, 1]; smaller = smoother/slower."""

    def __init__(self, alpha: float, init: np.ndarray | float = 0.0):
        self.alpha = float(alpha)
        self._y = np.asarray(init, dtype=float)

    def __call__(self, x: np.ndarray | float) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if self._y.shape != x.shape:
            self._y = np.array(x, copy=True)
        self._y = self.alpha * x + (1.0 - self.alpha) * self._y
        return self._y

    @property
    def value(self) -> np.ndarray:
        return self._y
