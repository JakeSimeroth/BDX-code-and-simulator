"""Perception: turn raw LiDAR + camera (+ privileged sim semantics) into a
structured :class:`WorldBelief` — an occupancy map plus tracked plants, humans
and the charging dock. The neural VLA still consumes raw sensors end-to-end;
this belief grounds language goals, feeds the scripted expert, and backs the
safety Guardian's geofence and human-proximity logic."""

from .occupancy import OccupancyGrid
from .world_model import GreenhouseMapper, Human, Plant, WorldBelief

__all__ = ["OccupancyGrid", "GreenhouseMapper", "WorldBelief", "Plant", "Human"]
