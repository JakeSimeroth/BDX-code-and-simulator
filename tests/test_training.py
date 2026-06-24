"""Tests for the training/VLA glue. Torch-heavy paths are gated so the suite
stays light where torch isn't installed."""

import numpy as np
import pytest

from gardener_bdx.common.types import (
    BatteryState,
    ImuReading,
    JointState,
    LocomotionCommand,
    Observation,
    WaterTankState,
)


def make_obs(n=12):
    return Observation(
        stamp=0.0,
        imu=ImuReading(np.array([1.0, 0, 0, 0]), np.zeros(3), np.array([0, 0, -9.81])),
        joints=JointState(np.zeros(n), np.zeros(n), np.zeros(n)),
        camera=None,
        battery=BatteryState(state_of_charge=0.7),
        water=WaterTankState(level_liters=1.0, capacity_liters=1.5),
    )


def test_lerobot_state_layout_matches_modality():
    from gardener_bdx.training.export_lerobot import _state_from_proprio

    # proprio = [pos(12), vel(12), g(3), angvel(3), payload(2)]
    p = np.arange(32, dtype=np.float32)[None]
    s = _state_from_proprio(p)
    assert s.shape == (1, 32)
    np.testing.assert_array_equal(s[0, :12], p[0, 0:12])        # joint_pos
    np.testing.assert_array_equal(s[0, 12:24], p[0, 12:24])     # joint_vel
    # imu = [ang_vel(27:30), proj_grav(24:27)]
    np.testing.assert_array_equal(s[0, 24:27], p[0, 27:30])     # ang_vel first
    np.testing.assert_array_equal(s[0, 27:30], p[0, 24:27])     # then gravity
    np.testing.assert_array_equal(s[0, 30:32], p[0, 30:32])     # payload


def test_groot_skill_inference():
    from gardener_bdx.common.types import Skill
    from gardener_bdx.policy.groot_vla import GR00TVLA

    assert GR00TVLA._infer_skill(LocomotionCommand(), 0.2) == Skill.DISPENSE_WATER
    assert GR00TVLA._infer_skill(LocomotionCommand(vx=0.3), 0.0) == Skill.NAVIGATE
    assert GR00TVLA._infer_skill(LocomotionCommand(), 0.0) == Skill.IDLE


def test_neural_vla_inference_smoke():
    pytest.importorskip("torch")
    from gardener_bdx.policy.task import TaskGoal
    from gardener_bdx.policy.vla_brain import build_vla

    vla = build_vla("neural", checkpoint="", device="cpu", system2_period=2)
    intent = vla.act(make_obs(), TaskGoal.parse("water the thirsty plants"))
    c = intent.locomotion
    assert np.all(np.isfinite([c.vx, c.vy, c.wz, intent.dispense_rate_lps]))


def test_locomotion_observation_dim_matches_isaac():
    # The runtime obs builder and the Isaac env must agree on dimensionality.
    from gardener_bdx.common.config import RobotConfig
    from gardener_bdx.policy.locomotion import locomotion_observation

    rc = RobotConfig.from_yaml()
    vec = locomotion_observation(make_obs(rc.n_joints), LocomotionCommand(),
                                 rc.default_joint_positions, np.zeros(rc.n_joints), 0.0)
    assert vec.shape[0] == 3 + 3 + 3 * rc.n_joints + 3 + 2
