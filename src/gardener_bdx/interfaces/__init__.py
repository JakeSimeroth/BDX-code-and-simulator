"""Hardware-abstraction seam. The simulator backends and the real-robot driver
both implement :class:`~gardener_bdx.interfaces.robot_io.RobotIO`, so identical
policy code runs in the digital twin and on the Jetson."""

from .robot_io import RobotIO

__all__ = ["RobotIO"]
