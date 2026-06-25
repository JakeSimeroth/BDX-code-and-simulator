"""CPU smoke tests for the neural VLA wiring (skipped when torch is absent).

Guards the System-2/System-1 plumbing and the quality upgrades: learned
modality fusion, the skill-conditioned action head, action standardization, and
the modality-availability mask that keeps behavior-cloning and inference fed the
same channels."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from gardener_bdx.common.types import (  # noqa: E402
    BatteryState,
    ImuReading,
    JointState,
    Observation,
    WaterTankState,
)
from gardener_bdx.policy.task import TaskGoal  # noqa: E402
from gardener_bdx.policy.vla_brain import ACTION_DIM, build_vla  # noqa: E402
from gardener_bdx.policy.vla_net import (  # noqa: E402
    EMB,
    LATENT_DIM,
    SKILL_EMB,
    WorldModelVLANet,
)


def _obs(n=12, soc=0.8, water=1.0):
    return Observation(
        stamp=1.0,
        imu=ImuReading(np.array([1.0, 0, 0, 0]), np.zeros(3), np.array([0, 0, -9.81])),
        joints=JointState(np.zeros(n), np.zeros(n), np.zeros(n)),
        battery=BatteryState(state_of_charge=soc),
        water=WaterTankState(level_liters=water, capacity_liters=1.5),
    )


def test_action_standardization_round_trips():
    net = WorldModelVLANet(action_dim=ACTION_DIM)
    a = torch.randn(16, ACTION_DIM)
    net.set_action_stats(a.mean(0).numpy(), a.std(0).numpy())
    back = net.denormalize_action(net.normalize_action(a))
    assert torch.allclose(back, a, atol=1e-4)


def test_modality_flags_and_stats_persist_in_checkpoint():
    net = WorldModelVLANet(action_dim=ACTION_DIM)
    net._warmup("cpu")
    net.set_modalities(use_depth=False, use_bev=False)
    net.set_action_stats(np.arange(ACTION_DIM), np.ones(ACTION_DIM) * 2.0)
    sd = net.state_dict()
    assert {"use_depth", "use_bev", "action_mean", "action_std"} <= set(sd)

    other = WorldModelVLANet(action_dim=ACTION_DIM)
    other._warmup("cpu")
    other.load_state_dict(sd, strict=False)
    assert float(other.use_depth) == 0.0 and float(other.use_bev) == 0.0
    assert torch.allclose(other.action_std, torch.full((ACTION_DIM,), 2.0))


def test_action_cond_has_skill_width():
    net = WorldModelVLANet(action_dim=ACTION_DIM)
    b = 3
    cond = net.action_cond(
        torch.zeros(b, LATENT_DIM), torch.zeros(b, EMB), torch.zeros(b, 6)
    )
    assert cond.shape == (b, LATENT_DIM + EMB + SKILL_EMB)


def test_neural_vla_act_returns_finite_intent():
    vla = build_vla("neural", checkpoint="", device="cpu", system2_period=2)
    goal = TaskGoal.parse("water the thirsty plants")
    vla.reset()
    intent = None
    for _ in range(3):  # crosses a System-2 refresh boundary
        intent = vla.act(_obs(), goal, None)
    cmd = intent.locomotion
    assert np.all(np.isfinite([cmd.vx, cmd.vy, cmd.wz, cmd.body_height]))
    assert intent.dispense_rate_lps >= 0.0
