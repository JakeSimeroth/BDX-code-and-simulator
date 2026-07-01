# Animations — the BDX expression layer

The brief for this droid isn't just "walk and water": it must carry itself like
a BDX — boot-up theatrics, idle fidgets, curiosity at plants, a greeting for
people. This layer makes *personality a first-class output of the VLA*.

## How it works

Two cleanly separated responsibilities:

- **The VLA decides *which* animation** — `Intent.expression` (an
  `Expression` enum value) sits next to `Intent.skill`. The scripted expert
  picks expressions from events (a person entered range, a plant was finished,
  battery is low); the neural VLA has a dedicated **expression head** on its
  System-2 latent and *learns* that mapping from the expert's demos.
- **The AnimationEngine decides *how it looks*** —
  `policy/animation.py` holds a library of small **parametric clips** (pure
  functions of phase, no mocap assets) and renders the selected expression as a
  bounded overlay on the locomotion *style channels*: `body_height`,
  `look_yaw`, `look_pitch`, plus a gait-energy scale. The runner blends the
  overlay into the `LocomotionCommand` every tick.

Because overlays ride the existing command plumbing, the same animations play
in the kinematic twin, MuJoCo, Isaac Sim, and on the Jetson — and the 2-DOF
neck (`neck_yaw`, `head_pitch`) gives the head language a real mechanism.

## Safety invariants

- `speed_scale ∈ [0, 1]`: an expression can slow or freeze the gait (ALERT),
  **never** speed it up.
- Overlays are clamped (±0.15 m height, ±1.0 rad yaw, ±0.6 rad pitch) and
  low-pass smoothed — switching or preempting clips can never step-change a
  command.
- The Guardian screens the final joint targets as always, and a safety
  **OVERRIDE/ESTOP suppresses the engine** (latched until safety clears):
  theatrics end where safety begins.

## The library

| Expression | Trigger (scripted expert) | Motion sketch | Type / priority |
|---|---|---|---|
| `BOOT` | first ~3 s after reset | rise from crouch, head sweep, nod; gait frozen | one-shot / 8 |
| `IDLE_BREATHE` | idling (default phase) | subtle body/head sway | loop / 1 |
| `IDLE_SCAN` | idling (every ~12 s) | slow look-around with beats of interest | one-shot / 2 |
| `CURIOUS` | `APPROACH_PLANT` | lean in, cock the head at the plant | one-shot / 4 |
| `GREET` | person enters slow radius (once per encounter, 12 s cooldown) | perk up + double nod, gait damped | one-shot / 7 |
| `ALERT` | person inside stop radius | stand tall, freeze, attentive micro-motion | loop / 9 |
| `WATERING` | `DISPENSE_WATER` | contented bob + nozzle sway | loop / 5 |
| `SATISFIED` | a plant just reached "well watered" | decaying happy wiggle + hop | one-shot / 5 |
| `LOW_POWER` | SoC near the dock threshold | head/body droop, 0.65× gait | loop / 3 |
| `DOCK_SETTLE` | docking and < 0.9 m from dock | ease into a crouch, chin down | one-shot / 6 |

Engine semantics: **one-shots latch** until finished (a greeting plays out even
though the VLA re-plans mid-gesture) unless preempted by strictly higher
priority; loops yield freely; everything cross-fades through the smoother.

## Adding an animation

1. Add the enum value in `common/types.py::Expression`.
2. Write a clip function `phase -> ExpressionOverlay` in `policy/animation.py`
   and register it in `LIBRARY` with duration / loop / priority.
3. Emit it from the scripted expert (`ScriptedGardenerVLA._pick_expression`)
   so demos contain it — the neural VLA then learns when to deploy it
   (`collect_demos` records it; `train_vla` supervises the expression head).
4. Add a case to `tests/test_animation.py` (bounds are asserted for the whole
   library automatically).

## Seeing it

- `python scripts/view_kinematic.py --gif out/gardener.gif` — watch pauses,
  greets, and slowdowns in the top-down behavior lab.
- `python scripts/view_mujoco.py --view` / `python scripts/run_isaac.py` — the
  overlays drive the real neck joints and stance height in physics/photoreal.
- Every `StepInfo` carries the rendered `overlay`, and `Intent.rationale`
  logs the deployed expression for debugging.
