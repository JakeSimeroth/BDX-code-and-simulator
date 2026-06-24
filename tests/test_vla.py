"""The scripted VLA expert — behavior across goals. (The neural VLA needs torch
and is exercised by the training scripts / its own integration tests.)"""

import numpy as np

from gardener_bdx.common.config import RobotConfig
from gardener_bdx.common.types import (
    BatteryState,
    ImuReading,
    JointState,
    Observation,
    Pose,
    WaterTankState,
)
from gardener_bdx.perception.world_model import GreenhouseMapper, Plant
from gardener_bdx.policy.task import TaskGoal, TaskKind
from gardener_bdx.policy.vla_brain import ScriptedGardenerVLA, build_vla


def make_obs(soc=0.8, water=1.0):
    n = 12
    return Observation(
        stamp=1.0,
        imu=ImuReading(np.array([1.0, 0, 0, 0]), np.zeros(3), np.array([0, 0, -9.81])),
        joints=JointState(np.zeros(n), np.zeros(n), np.zeros(n)),
        battery=BatteryState(state_of_charge=soc),
        water=WaterTankState(level_liters=water, capacity_liters=1.5),
    )


def world_with_dry_plant():
    mapper = GreenhouseMapper(dock_pose=Pose(np.array([-4.0, -4.0, 0.0]), np.array([1.0, 0, 0, 0])))
    b = mapper.belief
    b.robot_pose = Pose(np.array([0.0, 0.0, 0.35]), np.array([1.0, 0, 0, 0]))
    b.plants[0] = Plant(0, np.array([2.0, 0.0, 0.25]), dryness=0.9)
    return b


def test_task_parsing():
    assert TaskGoal.parse("return to the dock").kind == TaskKind.GO_DOCK
    assert TaskGoal.parse("patrol the greenhouse").kind == TaskKind.PATROL
    assert TaskGoal.parse("water plant #3").target_plant_id == 3
    assert TaskGoal.parse("water the parched plants").dryness_threshold > 0.6


def test_expert_navigates_to_dry_plant():
    vla = ScriptedGardenerVLA()
    intent = vla.act(make_obs(), TaskGoal.parse("water the thirsty plants"), world_with_dry_plant())
    assert intent.skill.value in ("navigate", "approach_plant")
    # Heading/velocity should point toward +x where the plant is.
    assert intent.locomotion.vx > 0.0


def test_expert_docks_when_battery_low():
    vla = ScriptedGardenerVLA()
    intent = vla.act(make_obs(soc=0.1), TaskGoal.parse("tend the garden"), world_with_dry_plant())
    assert intent.skill.value == "dock_charge"


def test_expert_dispenses_when_in_position():
    vla = ScriptedGardenerVLA()
    b = world_with_dry_plant()
    b.robot_pose = Pose(np.array([1.58, 0.0, 0.35]), np.array([1.0, 0, 0, 0]))  # ~0.42 m from plant
    intent = vla.act(make_obs(), TaskGoal.parse("water the thirsty plants"), b)
    assert intent.skill.value == "dispense_water"
    assert intent.dispense_rate_lps > 0.0


def test_build_vla_factory_scripted():
    assert isinstance(build_vla("scripted"), ScriptedGardenerVLA)
