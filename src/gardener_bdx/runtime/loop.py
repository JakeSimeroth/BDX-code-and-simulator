"""The control loop. Deliberately tiny: reset, then call ``controller.step(io)``
until done. Pacing lives in the backend's ``step()`` — virtual time in sim,
wall-clock sleep on hardware — so this loop is identical everywhere."""

from __future__ import annotations

from typing import Callable, Iterator, Optional

from ..interfaces.robot_io import RobotIO
from ..policy.runner import GardenerController, StepInfo


def run(
    controller: GardenerController,
    io: RobotIO,
    steps: Optional[int] = None,
    on_step: Optional[Callable[[int, StepInfo], None]] = None,
) -> Iterator[StepInfo]:
    """Generator over per-tick :class:`StepInfo`. ``steps=None`` runs forever
    (used on the robot); a finite count is used for sim rollouts and tests."""
    controller.reset(io)
    i = 0
    while steps is None or i < steps:
        info = controller.step(io)
        if on_step is not None:
            on_step(i, info)
        yield info
        i += 1
