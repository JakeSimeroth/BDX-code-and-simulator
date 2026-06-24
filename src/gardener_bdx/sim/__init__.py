"""Simulator backends, all implementing :class:`RobotIO`:

  * :class:`~gardener_bdx.sim.kinematic_backend.KinematicGreenhouse` — pure
    numpy, runs anywhere, closes the gardening loop today (default for tests and
    the quick demo).
  * :mod:`~gardener_bdx.sim.mujoco_backend` — full-physics MuJoCo (fast
    locomotion RL, the Open-Duck-Mini lineage).
  * :mod:`~gardener_bdx.sim.isaac_backend` — NVIDIA Isaac Sim / Isaac Lab, the
    photorealistic digital twin used to train the VLA and run Sim2Real.

``make_backend(name, ...)`` returns one by name."""

from .kinematic_backend import KinematicGreenhouse


def make_backend(name: str, robot_config, control_dt: float, **kwargs):
    name = name.lower()
    if name in ("kinematic", "kin", "numpy"):
        return KinematicGreenhouse(robot_config, control_dt, **kwargs)
    if name in ("mujoco", "mjx", "mj"):
        from .mujoco_backend import MujocoGreenhouse

        return MujocoGreenhouse(robot_config, control_dt, **kwargs)
    if name in ("isaac", "isaaclab", "isaacsim"):
        from .isaac_backend import IsaacGreenhouse

        return IsaacGreenhouse(robot_config, control_dt, **kwargs)
    raise ValueError(f"unknown sim backend: {name!r}")


__all__ = ["KinematicGreenhouse", "make_backend"]
