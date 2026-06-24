import numpy as np

from gardener_bdx.common.config import RobotConfig
from gardener_bdx.common.types import (
    BatteryState,
    ImuReading,
    JointState,
    LocomotionCommand,
    Observation,
    WaterTankState,
)
from gardener_bdx.policy.locomotion import MLP, LocomotionPolicy


def make_obs(rc):
    n = rc.n_joints
    return Observation(
        stamp=0.0,
        imu=ImuReading(np.array([1.0, 0, 0, 0]), np.zeros(3), np.array([0, 0, -9.81])),
        joints=JointState(rc.default_joint_positions.copy(), np.zeros(n), np.zeros(n)),
        battery=BatteryState(state_of_charge=1.0),
        water=WaterTankState(level_liters=1.0, capacity_liters=1.5),
    )


def test_mlp_numpy_roundtrip(tmp_path):
    w = [(np.random.randn(8, 5).astype(np.float32), np.zeros(8, np.float32)),
         (np.random.randn(3, 8).astype(np.float32), np.zeros(3, np.float32))]
    p = tmp_path / "m.npz"
    MLP.save_npz(str(p), w, activation="tanh")
    mlp = MLP.load_npz(str(p))
    x = np.random.randn(5).astype(np.float32)
    y = mlp(x)
    assert y.shape == (3,)
    assert np.all(np.isfinite(y))


def test_cpg_fallback_respects_joint_limits():
    rc = RobotConfig.from_yaml()
    pol = LocomotionPolicy(rc, policy_path=None)  # forces CPG fallback
    assert not pol.using_learned_policy
    obs = make_obs(rc)
    for vx in [0.0, 0.4, -0.2]:
        a = pol.act(obs, LocomotionCommand(vx=vx), dt=0.02)
        assert a.joint_position_targets.shape == (rc.n_joints,)
        assert np.all(a.joint_position_targets <= rc.joint_upper + 1e-6)
        assert np.all(a.joint_position_targets >= rc.joint_lower - 1e-6)


def test_posture_modifiers_move_named_joints():
    rc = RobotConfig.from_yaml()
    pol = LocomotionPolicy(rc, policy_path=None)
    obs = make_obs(rc)
    base = pol.act(obs, LocomotionCommand(), dt=0.02).joint_position_targets.copy()
    pol.reset()
    crouch = pol.act(obs, LocomotionCommand(body_height=-0.2), dt=0.02).joint_position_targets
    knees = [i for i, nm in enumerate(rc.joint_names) if "knee" in nm]
    # A crouch command should change knee targets.
    assert not np.allclose(base[knees], crouch[knees])
