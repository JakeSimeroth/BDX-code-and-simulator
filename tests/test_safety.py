"""The Guardian is the trust anchor — these tests pin its guarantees."""

import numpy as np
import pytest

from gardener_bdx.common.config import RobotConfig
from gardener_bdx.common.math_utils import quat_from_euler
from gardener_bdx.common.types import (
    Action,
    BatteryState,
    JointState,
    Pose,
    RobotState,
    SafetyLevel,
    Twist,
    WaterTankState,
)
from gardener_bdx.safety.guardian import GuardianConfig, SafetyGuardian


@pytest.fixture
def rc():
    return RobotConfig.from_yaml()


def make_state(rc, upright_cos=1.0, water_l=1.0):
    n = rc.n_joints
    return RobotState(
        stamp=0.0,
        base_pose=Pose.identity(),
        base_twist=Twist(np.zeros(3), np.zeros(3)),
        joints=JointState(rc.default_joint_positions.copy(), np.zeros(n), np.zeros(n)),
        battery=BatteryState(state_of_charge=0.8),
        water=WaterTankState(level_liters=water_l, capacity_liters=1.5),
        upright_cos=upright_cos,
    )


class FakeWorld:
    def __init__(self, human_dist=99.0, xy=(0.0, 0.0)):
        self._d = human_dist
        self._xy = np.array(xy)

    def distance_to_nearest_human(self):
        return self._d

    def robot_xy(self):
        return self._xy


def test_joint_limits_are_enforced(rc):
    g = SafetyGuardian(rc, 0.02)
    a = Action(joint_position_targets=rc.joint_upper + 5.0)  # way over
    v = g.check(a, make_state(rc), FakeWorld())
    assert np.all(v.action.joint_position_targets <= rc.joint_upper + 1e-6)
    assert np.all(v.action.joint_position_targets >= rc.joint_lower - 1e-6)


def test_rate_limit_caps_first_step(rc):
    g = SafetyGuardian(rc, 0.02)
    st = make_state(rc)
    far = rc.default_joint_positions + 10.0
    v = g.check(Action(joint_position_targets=far), st, FakeWorld())
    max_step = rc.joint_velocity_limit * 0.02
    assert np.all(v.action.joint_position_targets - rc.default_joint_positions <= max_step + 1e-6)


def test_tip_over_triggers_estop_and_kills_water(rc):
    g = SafetyGuardian(rc, 0.02)
    a = Action(joint_position_targets=rc.default_joint_positions.copy(), water_valve_lps=0.1)
    v = g.check(a, make_state(rc, upright_cos=0.1), FakeWorld())
    assert v.level == SafetyLevel.ESTOP
    assert v.action.water_valve_lps == 0.0


def test_human_proximity_freezes_gait(rc):
    g = SafetyGuardian(rc, 0.02)
    st = make_state(rc)
    target = rc.default_joint_positions + 0.5
    v = g.check(Action(joint_position_targets=target, water_valve_lps=0.1), st,
                FakeWorld(human_dist=0.3))
    assert v.level == SafetyLevel.OVERRIDE
    assert v.action.water_valve_lps == 0.0
    # Frozen => targets stay at current measured posture (within one rate step).
    assert np.allclose(v.action.joint_position_targets, st.joints.positions, atol=rc.joint_velocity_limit * 0.02 + 1e-6)


def test_geofence_breach_freezes(rc):
    g = SafetyGuardian(rc, 0.02)
    v = g.check(Action(joint_position_targets=rc.default_joint_positions.copy()),
                make_state(rc), FakeWorld(xy=(100.0, 0.0)))
    assert v.level == SafetyLevel.OVERRIDE


def test_water_interlock_when_empty(rc):
    g = SafetyGuardian(rc, 0.02)
    v = g.check(Action(joint_position_targets=rc.default_joint_positions.copy(), water_valve_lps=0.2),
                make_state(rc, water_l=0.0), FakeWorld())
    assert v.action.water_valve_lps == 0.0


def test_estop_latches_until_reset(rc):
    g = SafetyGuardian(rc, 0.02)
    g.trip_estop()
    v = g.check(Action(joint_position_targets=rc.default_joint_positions + 0.3, water_valve_lps=0.2),
                make_state(rc), FakeWorld())
    assert v.level == SafetyLevel.ESTOP
    assert v.action.water_valve_lps == 0.0
    g.reset_estop()
    v2 = g.check(Action(joint_position_targets=rc.default_joint_positions.copy()),
                 make_state(rc), FakeWorld())
    assert v2.level != SafetyLevel.ESTOP
