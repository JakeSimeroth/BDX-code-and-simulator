"""Task goals — how a natural-language instruction is presented to the VLA.

The **neural** VLA reads the raw instruction string (it is language-conditioned
end-to-end). The lightweight parser here produces an additional *structured*
view of the goal that (a) conditions the scripted expert demonstrator, (b) gives
the safety/skill layers symbolic context, and (c) provides supervision targets
when we distill the scripted expert into the neural policy."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class TaskKind(Enum):
    TEND_GARDEN = "tend_garden"  # autonomously find & water dry plants
    WATER_PLANT = "water_plant"  # water a specific plant
    PATROL = "patrol"  # map/inspect without watering
    GO_DOCK = "go_dock"  # return to charger
    IDLE = "idle"


@dataclass
class TaskGoal:
    """A goal the VLA is conditioned on for an episode/command."""

    instruction: str
    kind: TaskKind = TaskKind.TEND_GARDEN
    target_plant_id: Optional[int] = None
    # Water any plant whose estimated dryness exceeds this (0 wet … 1 parched).
    dryness_threshold: float = 0.45
    water_per_plant_l: float = 0.12
    # Return to dock below this state-of-charge regardless of task.
    return_to_dock_soc: float = 0.20

    @staticmethod
    def parse(instruction: str) -> "TaskGoal":
        """Best-effort intent extraction. Intentionally simple and transparent —
        the heavy lifting of grounding is the neural VLA's job; this just gives
        the scripted stack and the trainer a usable target."""
        text = instruction.lower().strip()
        goal = TaskGoal(instruction=instruction)

        if re.search(r"\b(dock|charge|recharge|home|station)\b", text):
            goal.kind = TaskKind.GO_DOCK
        elif re.search(r"\b(patrol|inspect|map|survey|check)\b", text):
            goal.kind = TaskKind.PATROL
        elif re.search(r"\b(idle|stop|wait|stand)\b", text):
            goal.kind = TaskKind.IDLE
        elif re.search(r"\bwater\b|\birrigat", text):
            m = re.search(r"plant\s*#?(\d+)", text)
            if m:
                goal.kind = TaskKind.WATER_PLANT
                goal.target_plant_id = int(m.group(1))
            else:
                goal.kind = TaskKind.TEND_GARDEN

        # Adverbs of thoroughness tweak the dryness threshold.
        if "thirsty" in text or "very dry" in text or "parched" in text:
            goal.dryness_threshold = 0.65
        elif "everything" in text or "all" in text:
            goal.dryness_threshold = 0.25
        return goal
