"""A small 2D log-odds occupancy grid. Enough to support navigation goals and
the safety geofence in the digital twin; on hardware it is fed by the same
LiDAR points after SLAM provides the pose."""

from __future__ import annotations

import numpy as np


class OccupancyGrid:
    def __init__(self, size_m: float = 12.0, resolution_m: float = 0.05, origin=(-6.0, -6.0)):
        self.res = float(resolution_m)
        self.origin = np.asarray(origin, dtype=float)  # world coord of cell (0,0)
        self.n = int(round(size_m / self.res))
        # Log-odds; 0 = unknown, >0 occupied, <0 free.
        self.logodds = np.zeros((self.n, self.n), dtype=np.float32)
        self._l_occ = 0.85
        self._l_free = -0.4
        self._clamp = 6.0

    def world_to_cell(self, p_xy: np.ndarray) -> tuple[int, int]:
        c = ((np.asarray(p_xy)[:2] - self.origin) / self.res).astype(int)
        return int(c[0]), int(c[1])

    def in_bounds(self, i: int, j: int) -> bool:
        return 0 <= i < self.n and 0 <= j < self.n

    def integrate_scan(self, sensor_xy: np.ndarray, points_xy: np.ndarray) -> None:
        """Bresenham-free ray integration: mark endpoints occupied and sample a
        few free cells along each ray. Cheap and adequate for our purposes."""
        si, sj = self.world_to_cell(sensor_xy)
        for p in points_xy:
            ei, ej = self.world_to_cell(p)
            # Free space sampling along the ray.
            steps = max(abs(ei - si), abs(ej - sj))
            if steps > 0:
                for t in np.linspace(0.0, 1.0, num=min(steps, 64), endpoint=False):
                    fi = int(round(si + (ei - si) * t))
                    fj = int(round(sj + (ej - sj) * t))
                    if self.in_bounds(fi, fj):
                        self.logodds[fi, fj] = np.clip(
                            self.logodds[fi, fj] + self._l_free, -self._clamp, self._clamp
                        )
            if self.in_bounds(ei, ej):
                self.logodds[ei, ej] = np.clip(
                    self.logodds[ei, ej] + self._l_occ, -self._clamp, self._clamp
                )

    def is_occupied(self, p_xy: np.ndarray, thresh: float = 0.5) -> bool:
        i, j = self.world_to_cell(p_xy)
        if not self.in_bounds(i, j):
            return True  # out of map => treat as blocked
        return bool(self.logodds[i, j] > thresh)

    def occupied_points(self, thresh: float = 0.5) -> np.ndarray:
        idx = np.argwhere(self.logodds > thresh)
        if idx.size == 0:
            return np.zeros((0, 2))
        return self.origin + (idx + 0.5) * self.res

    def coverage_fraction(self) -> float:
        """Fraction of cells that are no longer 'unknown' — a patrol/exploration
        progress signal."""
        return float(np.mean(np.abs(self.logodds) > 1e-3))
