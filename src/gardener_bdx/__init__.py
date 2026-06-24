"""GardenerBDX — an end-to-end, VLA-driven control stack for a BDX-inspired
bipedal greenhouse-gardener droid.

The package is organized around one idea: a single **Vision-Language-Action
(VLA) world model** is the brain of the robot. It consumes raw sensor streams
(RGB-D, LiDAR, proprioception, audio) plus a natural-language goal, and emits
whole-body intent. That intent is realized through a fast learned locomotion
substrate and screened by a deterministic safety Guardian.

Everything is written against a single hardware-abstraction seam
(`interfaces.robot_io.RobotIO`) so the *exact same* policy code runs:

  * in the **digital twin** (Isaac Sim / Isaac Lab, MuJoCo, or the bundled
    pure-numpy kinematic backend), and
  * on the **real robot** (NVIDIA Jetson + actuator/sensor drivers).

This is the foundation of the Sim2Real workflow: train and prove the VLA in
simulation, then port the identical control graph to hardware.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
