"""The BDX expression layer: clip safety, engine semantics (latching, priority,
suppression, continuity), the scripted VLA's animation deployment, and the
end-to-end runner integration."""

import numpy as np

from gardener_bdx.common.config import HierarchyConfig, RobotConfig
from gardener_bdx.common.types import (
    BatteryState,
    Expression,
    ImuReading,
    JointState,
    Observation,
    Pose,
    WaterTankState,
)
from gardener_bdx.perception.world_model import GreenhouseMapper, Human, Plant
from gardener_bdx.policy.animation import (
    LIBRARY,
    MAX_DH,
    MAX_PITCH,
    MAX_YAW,
    AnimationEngine,
)
from gardener_bdx.policy.runner import GardenerController
from gardener_bdx.policy.task import TaskGoal
from gardener_bdx.policy.vla_brain import ScriptedGardenerVLA
from gardener_bdx.runtime import loop
from gardener_bdx.sim import make_backend

DT = 0.02


def make_obs(stamp=10.0, soc=0.8, water=1.0):
    n = 12
    return Observation(
        stamp=stamp,
        imu=ImuReading(np.array([1.0, 0, 0, 0]), np.zeros(3), np.array([0, 0, -9.81])),
        joints=JointState(np.zeros(n), np.zeros(n), np.zeros(n)),
        battery=BatteryState(state_of_charge=soc),
        water=WaterTankState(level_liters=water, capacity_liters=1.5),
    )


def world(human_dist=None, plant_dist=2.0, dryness=0.9):
    mapper = GreenhouseMapper(dock_pose=Pose(np.array([-4.0, -4.0, 0.0]), np.array([1.0, 0, 0, 0])))
    b = mapper.belief
    b.robot_pose = Pose(np.array([0.0, 0.0, 0.35]), np.array([1.0, 0, 0, 0]))
    b.plants[0] = Plant(0, np.array([plant_dist, 0.0, 0.25]), dryness=dryness)
    if human_dist is not None:
        b.humans[0] = Human(0, np.array([0.0, human_dist, 0.9]))
    return b


def booted_vla():
    """A scripted VLA that is past its power-on animation window."""
    vla = ScriptedGardenerVLA()
    vla.act(make_obs(stamp=0.0), TaskGoal.parse("tend the garden"), world())  # sets _t0
    return vla


# ---------------------------------------------------------------------------- #
# Clip library
# ---------------------------------------------------------------------------- #


def test_every_clip_is_bounded_and_never_speeds_up():
    for expr, clip in LIBRARY.items():
        for p in np.linspace(0.0, 0.999, 97):
            ov = clip.fn(float(p))
            assert abs(ov.body_height) <= MAX_DH, expr
            assert abs(ov.look_yaw) <= MAX_YAW, expr
            assert abs(ov.look_pitch) <= MAX_PITCH, expr
            assert 0.0 <= ov.speed_scale <= 1.0, expr


def test_library_covers_every_expression():
    assert set(LIBRARY) == set(Expression)


# ---------------------------------------------------------------------------- #
# Engine semantics
# ---------------------------------------------------------------------------- #


def test_oneshot_latches_against_lower_priority():
    eng = AnimationEngine()
    eng.request(Expression.GREET)               # one-shot, priority 7
    eng.update(DT)
    eng.request(Expression.IDLE_BREATHE)        # priority 1 — must not steal the stage
    assert eng.playing == Expression.GREET
    for _ in range(int(2.4 / DT)):              # let the greeting finish
        eng.update(DT)
    eng.request(Expression.IDLE_BREATHE)
    assert eng.playing == Expression.IDLE_BREATHE


def test_higher_priority_preempts():
    eng = AnimationEngine()
    eng.request(Expression.WATERING)
    eng.update(DT)
    eng.request(Expression.ALERT)               # person close beats everything
    assert eng.playing == Expression.ALERT


def test_overlay_is_continuous_across_preemption():
    eng = AnimationEngine()
    eng.request(Expression.SATISFIED)           # big fast wiggle
    prev = eng.update(DT).as_vec()
    for i in range(200):
        if i == 40:
            eng.request(Expression.ALERT)       # hard switch mid-wiggle
        cur = eng.update(DT).as_vec()
        assert np.all(np.abs(cur - prev) < 0.25), "overlay jumped on switch"
        prev = cur


def test_suppression_latches_and_releases():
    eng = AnimationEngine()
    eng.request(Expression.SATISFIED)
    for _ in range(20):
        eng.update(DT)
    eng.suppress()
    for _ in range(int(1.0 / DT)):
        ov = eng.update(DT)
    assert np.allclose(ov.as_vec(), [0, 0, 0, 1], atol=0.02)  # decayed to identity
    eng.request(Expression.GREET)               # requests accepted but not rendered
    for _ in range(10):
        ov = eng.update(DT)
    assert np.allclose(ov.as_vec(), [0, 0, 0, 1], atol=0.02)
    eng.release()
    eng.request(Expression.ALERT)
    for _ in range(int(1.0 / DT)):
        ov = eng.update(DT)
    assert ov.body_height > 0.005               # expressive again after release


# ---------------------------------------------------------------------------- #
# Scripted VLA deploys animations from events
# ---------------------------------------------------------------------------- #


def test_boot_animation_on_startup():
    vla = ScriptedGardenerVLA()
    intent = vla.act(make_obs(stamp=5.0), TaskGoal.parse("tend the garden"), world())
    assert intent.expression == Expression.BOOT


def test_greets_person_once_then_alert_when_close():
    vla = booted_vla()
    goal = TaskGoal.parse("tend the garden")
    # Person enters the slow radius -> one greeting...
    intent = vla.act(make_obs(stamp=10.0), goal, world(human_dist=1.2))
    assert intent.expression == Expression.GREET
    # ...but not a second one on the very next re-plan.
    intent = vla.act(make_obs(stamp=10.5), goal, world(human_dist=1.1))
    assert intent.expression != Expression.GREET
    # Very close -> ALERT dominates.
    intent = vla.act(make_obs(stamp=11.0), goal, world(human_dist=0.5))
    assert intent.expression == Expression.ALERT


def test_watering_and_curious_expressions():
    vla = booted_vla()
    goal = TaskGoal.parse("water the thirsty plants")
    # In dispense position (standoff ~0.45 m).
    b = world(plant_dist=0.42)
    intent = vla.act(make_obs(stamp=10.0), goal, b)
    assert intent.skill.value == "dispense_water"
    assert intent.expression == Expression.WATERING
    # Fine-approach range -> curious examination.
    vla2 = booted_vla()
    b2 = world(plant_dist=0.9)
    intent = vla2.act(make_obs(stamp=10.0), goal, b2)
    assert intent.skill.value == "approach_plant"
    assert intent.expression == Expression.CURIOUS


def test_low_power_droop_when_docking_far_from_dock():
    vla = booted_vla()
    intent = vla.act(make_obs(stamp=10.0, soc=0.1), TaskGoal.parse("tend the garden"), world())
    assert intent.skill.value == "dock_charge"
    assert intent.expression == Expression.LOW_POWER


def test_idle_cycles_between_breathe_and_scan():
    vla = booted_vla()
    goal = TaskGoal.parse("stand by")
    assert goal.kind.value == "idle"
    seen = set()
    for t in np.arange(4.0, 28.0, 1.0):
        seen.add(vla.act(make_obs(stamp=float(t)), goal, world()).expression)
    assert Expression.IDLE_BREATHE in seen and Expression.IDLE_SCAN in seen


# ---------------------------------------------------------------------------- #
# Full-loop integration
# ---------------------------------------------------------------------------- #


def test_rollout_plays_animations_and_stays_finite():
    rc = RobotConfig.from_yaml()
    h = HierarchyConfig.from_yaml()
    io = make_backend("kinematic", rc, 1.0 / h.locomotion_hz, seed=7)
    ctrl = GardenerController(rc, h, goal="tend the garden and water the thirsty plants")
    seen: set = set()
    for info in loop.run(ctrl, io, steps=600):
        seen.add(info.intent.expression)
        assert info.overlay is not None
        v = info.overlay.as_vec()
        assert np.all(np.isfinite(v))
        assert 0.0 <= v[3] <= 1.0 + 1e-9      # gait energy only ever reduced
    io.close()
    assert Expression.BOOT in seen            # it booted with theatrics
    assert len(seen) >= 3                     # and kept expressing while working
