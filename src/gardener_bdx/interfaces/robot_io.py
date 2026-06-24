"""``RobotIO`` — the one abstraction that decouples *brains* from *bodies*.

A backend only has to answer four questions:

  * ``reset()``   — put the body in a start state, return the first Observation.
  * ``read()``    — what do the sensors say right now?
  * ``write(a)``  — apply this Action to the actuators / valves.
  * ``step()``    — let time advance one control tick.

In the **digital twin**, ``step()`` advances the physics engine. On the **real
robot**, ``step()`` simply paces the loop to the control period while the
physical world advances on its own. Because perception, policy and safety only
ever touch this interface, the Sim2Real port is a backend swap — not a rewrite.
"""

from __future__ import annotations

import abc

from typing import Optional

from ..common.config import RobotConfig
from ..common.types import Action, Observation, SceneSemantics


class RobotIO(abc.ABC):
    """Abstract robot input/output. Deliberately thin: it carries no policy, no
    state estimation, no safety — only sensing and actuation."""

    def __init__(self, robot_config: RobotConfig, control_dt: float):
        self._robot_config = robot_config
        self._control_dt = float(control_dt)

    # -- read-only properties ---------------------------------------------- #
    @property
    def robot_config(self) -> RobotConfig:
        return self._robot_config

    @property
    def dt(self) -> float:
        """Nominal control period in seconds."""
        return self._control_dt

    # -- lifecycle --------------------------------------------------------- #
    @abc.abstractmethod
    def reset(self) -> Observation:
        """Reset the body to an initial state and return the first observation."""

    @abc.abstractmethod
    def read(self) -> Observation:
        """Return the most recent multimodal sensor bundle."""

    @abc.abstractmethod
    def write(self, action: Action) -> None:
        """Apply an actuator/valve command for the current tick."""

    @abc.abstractmethod
    def step(self) -> None:
        """Advance one control tick.

        Sim: advance physics by (at least) ``self.dt``. Hardware: sleep until the
        next tick boundary so the loop runs at the configured rate."""

    @abc.abstractmethod
    def now(self) -> float:
        """Monotonic time in seconds in this backend's clock domain."""

    def set_base_command_hint(self, cmd) -> None:  # pragma: no cover - default no-op
        """Optional affordance for *reduced-order* twins (e.g. the kinematic
        backend) that move the floating base from a commanded body twist rather
        than from simulated foot contact. Full-physics backends (MuJoCo, Isaac)
        and hardware ignore this — their base moves because the legs push the
        ground. The runner passes the post-safety command, so a Guardian freeze
        still halts the kinematic base."""

    def semantics(self) -> Optional[SceneSemantics]:
        """Privileged ground-truth detections, if this backend is a simulator
        and chooses to expose them. Hardware returns None. Perception treats
        this as *one possible detector* — never as a requirement."""
        return None

    def close(self) -> None:  # pragma: no cover - default no-op
        """Release resources (sim context, serial buses, …)."""

    # -- convenience ------------------------------------------------------- #
    def is_simulation(self) -> bool:
        return True

    def __enter__(self) -> "RobotIO":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
