"""End-to-end smoke tests of the full control graph in the kinematic twin."""

import numpy as np

from gardener_bdx.common.config import HierarchyConfig, RobotConfig
from gardener_bdx.policy.runner import GardenerController
from gardener_bdx.runtime import loop
from gardener_bdx.sim import make_backend


def build(goal="tend the garden and water the thirsty plants", seed=1):
    rc = RobotConfig.from_yaml()
    h = HierarchyConfig.from_yaml()
    io = make_backend("kinematic", rc, 1.0 / h.locomotion_hz, seed=seed)
    ctrl = GardenerController(rc, h, goal=goal)
    return rc, io, ctrl


def test_rollout_runs_and_waters_plants():
    rc, io, ctrl = build()
    ctrl.reset(io)
    dry0 = io.plant_dryness.copy()
    skills = set()
    for _ in range(3000):
        info = ctrl.step(io)
        skills.add(info.intent.skill.value)
    # It must have actually navigated and dispensed (not just idled).
    assert {"navigate", "dispense_water"} <= skills
    # The thirstiest plant should be wetter than it started.
    assert io.plant_dryness[int(np.argmax(dry0))] < dry0.max()
    # It discovered the whole greenhouse.
    assert len(ctrl.mapper.belief.plants) == io.n_plants


def test_go_dock_goal_docks():
    rc, io, ctrl = build(goal="return to the dock and charge")
    ctrl.reset(io)
    saw_dock = any(ctrl.step(io).intent.skill.value == "dock_charge" for _ in range(1500))
    assert saw_dock


def test_low_battery_forces_docking():
    rc, io, ctrl = build()
    ctrl.reset(io)
    io.soc = 0.1  # below return_to_dock_soc
    info = None
    for _ in range(300):
        info = ctrl.step(io)
    assert info.intent.skill.value == "dock_charge"


def test_safety_always_within_limits_over_rollout():
    rc, io, ctrl = build()
    ctrl.reset(io)
    for _ in range(1000):
        info = ctrl.step(io)
        a = info.verdict.action.joint_position_targets
        assert np.all(a <= rc.joint_upper + 1e-6)
        assert np.all(a >= rc.joint_lower - 1e-6)
