"""The BDX expression layer — a library of parametric animations and the engine
that deploys them.

The VLA (scripted or neural) picks an :class:`Expression` per intent — *which*
animation to deploy given what is happening (a person appeared, a plant is being
examined, battery is low). This module owns *how* that animation looks: each
clip is a small procedural function of phase that produces a bounded
:class:`ExpressionOverlay` on the locomotion style channels
(``body_height``, ``look_yaw``, ``look_pitch``) plus a gait-energy scale.

Why overlays on style channels (and not raw joint scripts)?

  * They ride the existing ``LocomotionCommand`` plumbing, so the same
    animations play in the kinematic twin, MuJoCo, Isaac Sim, and on the Jetson
    with zero backend-specific code.
  * The locomotion policy stays in charge of *balance*; expressions are pure
    body language layered on top.
  * The Guardian still screens every resulting joint target — an animation can
    never violate limits, and safety overrides suppress expressiveness
    instantly (a droid mid-"greet" that must stop for a human, stops).

Invariants the engine guarantees:
  * ``speed_scale`` ∈ [0, 1] — an expression may slow or freeze the gait
    (ALERT), never speed it up.
  * Overlay channels are clamped to conservative bounds and low-pass smoothed,
    so switching/preempting clips can never command a step change.
  * One-shot clips latch until finished unless preempted by a strictly
    higher-priority request, so a greeting plays out even though the VLA
    re-plans several times during it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict

import numpy as np

from ..common.types import Expression

# Conservative overlay bounds (rad / m). Final joint targets are additionally
# clipped by the locomotion policy and screened by the Guardian.
MAX_DH = 0.15        # |body_height| overlay, m
MAX_YAW = 1.0        # |look_yaw| overlay, rad
MAX_PITCH = 0.6      # |look_pitch| overlay, rad
SMOOTH_TAU = 0.12    # s, low-pass time constant on the rendered overlay


@dataclass
class ExpressionOverlay:
    """Additive style deltas + a multiplicative gait-energy scale for one tick."""

    body_height: float = 0.0
    look_yaw: float = 0.0
    look_pitch: float = 0.0
    speed_scale: float = 1.0

    def as_vec(self) -> np.ndarray:
        return np.array([self.body_height, self.look_yaw, self.look_pitch, self.speed_scale])

    @staticmethod
    def from_vec(v: np.ndarray) -> "ExpressionOverlay":
        return ExpressionOverlay(float(v[0]), float(v[1]), float(v[2]), float(v[3]))


@dataclass(frozen=True)
class Clip:
    """One parametric animation. ``fn(phase)`` maps [0,1) -> overlay; ``loop``
    clips repeat until superseded, one-shots latch until phase reaches 1."""

    fn: Callable[[float], ExpressionOverlay]
    duration: float          # s, one loop / the whole one-shot
    loop: bool
    priority: int            # higher preempts lower


def _ease(p: float) -> float:
    """Smoothstep — gentle attack/decay inside clips."""
    p = float(np.clip(p, 0.0, 1.0))
    return p * p * (3.0 - 2.0 * p)


def _window(p: float, a: float, b: float) -> float:
    """Raised-cosine bump that is 0 outside (a,b) and 1 mid-window."""
    if p <= a or p >= b:
        return 0.0
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * (p - a) / (b - a))


# --------------------------------------------------------------------------- #
# The animation library. Small procedural clips — tuned for a ~0.4 m BDX with a
# 2-DOF neck. Amplitudes are deliberately theatrical: this is the personality.
# --------------------------------------------------------------------------- #


def _boot(p: float) -> ExpressionOverlay:
    """Power-on: rise from a crouch, sweep the head left-right, settle + nod."""
    rise = -0.12 * (1.0 - _ease(min(p / 0.35, 1.0)))              # crouch -> stand
    sweep = 0.7 * np.sin(2.0 * np.pi * 1.5 * p) * _window(p, 0.30, 0.80)
    nod = -0.30 * _window(p, 0.82, 0.98)
    return ExpressionOverlay(rise, sweep, nod, speed_scale=0.0)   # don't walk mid-boot


def _idle_breathe(p: float) -> ExpressionOverlay:
    s = np.sin(2.0 * np.pi * p)
    return ExpressionOverlay(0.010 * s, 0.0, 0.030 * np.sin(2.0 * np.pi * p + 0.9))


def _idle_scan(p: float) -> ExpressionOverlay:
    """Slow, curious look-around; a beat of interest at each extreme."""
    yaw = 0.8 * np.sin(2.0 * np.pi * p) * _ease(min(p / 0.15, 1.0)) * _ease(min((1 - p) / 0.15, 1.0))
    pitch = -0.12 * _window(p, 0.20, 0.45) - 0.12 * _window(p, 0.70, 0.95)
    return ExpressionOverlay(0.0, yaw, pitch)


def _curious(p: float) -> ExpressionOverlay:
    """Lean in and cock the head at the thing being examined."""
    lean = -0.035 * _ease(min(p / 0.3, 1.0)) * _ease(min((1 - p) / 0.3, 1.0))
    tilt = 0.10 * np.sin(2.0 * np.pi * 2.0 * p) * _window(p, 0.25, 0.85)
    pitch = -0.22 * _window(p, 0.10, 0.95)
    return ExpressionOverlay(lean, tilt, pitch)


def _greet(p: float) -> ExpressionOverlay:
    """Perk up and double-nod at a person — the BDX hello."""
    perk = 0.030 * _window(p, 0.05, 0.55)
    nod = -0.28 * (_window(p, 0.25, 0.50) + _window(p, 0.55, 0.80))
    bounce = 0.012 * np.sin(2.0 * np.pi * 3.0 * p) * _window(p, 0.2, 0.9)
    return ExpressionOverlay(perk + bounce, 0.0, nod, speed_scale=0.35)


def _alert(p: float) -> ExpressionOverlay:
    """Person very close: stand tall, hold still, stay visibly attentive."""
    return ExpressionOverlay(0.020, 0.0, 0.05 * np.sin(2.0 * np.pi * p), speed_scale=0.0)


def _watering(p: float) -> ExpressionOverlay:
    """Contented bob + gentle nozzle sway while dispensing."""
    bob = 0.014 * np.sin(2.0 * np.pi * p)
    sway = 0.06 * np.sin(2.0 * np.pi * 2.0 * p)
    return ExpressionOverlay(bob, sway, 0.0, speed_scale=0.5)


def _satisfied(p: float) -> ExpressionOverlay:
    """Job-done wiggle: fast decaying yaw shimmy + a hop of the body."""
    decay = 1.0 - _ease(p)
    wiggle = 0.40 * np.sin(2.0 * np.pi * 4.0 * p) * decay
    hop = 0.020 * abs(np.sin(2.0 * np.pi * 2.0 * p)) * decay
    return ExpressionOverlay(hop, wiggle, -0.08 * decay, speed_scale=0.6)


def _low_power(p: float) -> ExpressionOverlay:
    """Droop: head down, body sagging, everything a bit slower."""
    sag = 0.010 * np.sin(2.0 * np.pi * 0.5 * p)
    return ExpressionOverlay(-0.045 + sag, 0.0, 0.30, speed_scale=0.65)


def _dock_settle(p: float) -> ExpressionOverlay:
    """Settle onto the charger: ease into a crouch, chin down, power down."""
    crouch = -0.080 * _ease(min(p / 0.6, 1.0))
    return ExpressionOverlay(crouch, 0.0, 0.18 * _ease(p), speed_scale=0.4)


LIBRARY: Dict[Expression, Clip] = {
    Expression.NONE:         Clip(lambda p: ExpressionOverlay(), 1.0, loop=True, priority=0),
    Expression.IDLE_BREATHE: Clip(_idle_breathe, 4.0, loop=True, priority=1),
    Expression.IDLE_SCAN:    Clip(_idle_scan, 5.0, loop=False, priority=2),
    Expression.LOW_POWER:    Clip(_low_power, 3.0, loop=True, priority=3),
    Expression.CURIOUS:      Clip(_curious, 2.5, loop=False, priority=4),
    Expression.WATERING:     Clip(_watering, 2.0, loop=True, priority=5),
    Expression.SATISFIED:    Clip(_satisfied, 1.6, loop=False, priority=5),
    Expression.DOCK_SETTLE:  Clip(_dock_settle, 2.0, loop=False, priority=6),
    Expression.GREET:        Clip(_greet, 2.2, loop=False, priority=7),
    Expression.BOOT:         Clip(_boot, 3.0, loop=False, priority=8),
    Expression.ALERT:        Clip(_alert, 1.0, loop=True, priority=9),
}

# Canonical index order shared by demo capture, the neural expression head, and
# Intent decoding — analogous to the SKILLS list.
EXPRESSION_ORDER = tuple(Expression)


@dataclass
class AnimationEngine:
    """Plays the requested expression with latching, priorities, and smoothing.

    Call :meth:`request` whenever the VLA emits an intent (slow clock is fine)
    and :meth:`update` every control tick; blend the returned overlay into the
    ``LocomotionCommand`` before the locomotion substrate."""

    _current: Expression = Expression.NONE
    _phase: float = 0.0
    _smooth: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0, 1.0]))
    _suppressed: bool = False

    def reset(self) -> None:
        self._current = Expression.NONE
        self._phase = 0.0
        self._smooth = np.array([0.0, 0.0, 0.0, 1.0])
        self._suppressed = False

    @property
    def playing(self) -> Expression:
        return self._current

    def request(self, expr: Expression) -> None:
        """Ask the engine to play ``expr``. A running one-shot keeps the stage
        unless the request has strictly higher priority; loops yield freely.
        Requests are accepted (the clip advances) even while suppressed, but the
        rendered overlay stays identity until :meth:`release`."""
        if expr == self._current:
            return
        cur = LIBRARY[self._current]
        new = LIBRARY[expr]
        oneshot_running = (not cur.loop) and self._phase < 1.0
        if oneshot_running and new.priority <= cur.priority:
            return  # let the current gesture finish
        self._current = expr
        self._phase = 0.0

    def suppress(self) -> None:
        """Safety hook: kill expressiveness now (overlay decays to identity via
        the smoother, so even suppression is continuous). Latches until
        :meth:`release` — a persistent safety override stays unexpressive."""
        self._suppressed = True
        self._current = Expression.NONE
        self._phase = 0.0

    def release(self) -> None:
        """Safety cleared — expressions may play again."""
        self._suppressed = False

    def update(self, dt: float) -> ExpressionOverlay:
        clip = LIBRARY[self._current]
        if self._phase >= 1.0 and not clip.loop:
            # One-shot finished: fall back to a clean gait until re-requested.
            self._current = Expression.NONE
            clip = LIBRARY[self._current]
            self._phase = 0.0

        raw = clip.fn(self._phase % 1.0)
        self._phase += dt / max(clip.duration, 1e-6)
        if clip.loop:
            self._phase %= 1.0

        target = np.array([
            float(np.clip(raw.body_height, -MAX_DH, MAX_DH)),
            float(np.clip(raw.look_yaw, -MAX_YAW, MAX_YAW)),
            float(np.clip(raw.look_pitch, -MAX_PITCH, MAX_PITCH)),
            float(np.clip(raw.speed_scale, 0.0, 1.0)),
        ])
        if self._suppressed:
            target = np.array([0.0, 0.0, 0.0, 1.0])

        alpha = 1.0 - np.exp(-dt / SMOOTH_TAU)  # continuity across any switch
        self._smooth = self._smooth + alpha * (target - self._smooth)
        return ExpressionOverlay.from_vec(self._smooth)
