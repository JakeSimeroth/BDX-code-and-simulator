# Safety — the Guardian (Halos-aligned)

An end-to-end neural policy is great at being *capable* and terrible at
*guaranteeing* anything. Around people, water, and living plants we need
guarantees that don't depend on a network's good mood. So the learned stack only
ever **proposes** an action; a deterministic, auditable **Guardian** decides what
actually executes.

## Alignment with NVIDIA Halos for Robotics

NVIDIA **Halos for Robotics** is a full-stack safety system for physical AI
operating near people — a standardized layer between AI compute and actuators
that holds regardless of the policy. Our `safety/guardian.py` is the
robot-specific instantiation of that principle: a fixed safety layer the learned
policy cannot bypass, running at the motor rate in both sim and hardware. When
moving to production, this is the component we certify and, where applicable,
map onto the Halos architecture and a hardware safety MCU.

## What the Guardian enforces (every tick)

| Check | Trigger | Action |
|---|---|---|
| **E-stop (latched)** | external/SIGINT or fault | hold posture, cut water, damp torque; requires manual reset |
| **Tip-over arrest** | uprightness `cos < 0.35` | protective stance + cut water → `ESTOP`; blends to safe stance from `cos < 0.80` |
| **Human proximity** | person within `0.7 m` | freeze gait (hold posture), cut water → `OVERRIDE` |
| **Geofence** | base outside operating box | freeze + cut water → `OVERRIDE` |
| **Water interlock** | tank empty *or* tilted | close valve → `CAUTION` |
| **Joint limits** | target out of range | hard clamp → `CAUTION` |
| **Torque clamp** | feed-forward over limit | clamp → `CAUTION` |
| **Slew-rate limit** | target step > vel·dt | limit to actuator velocity every tick |

Output is a `SafetyVerdict{level, action, reasons}`. The `reasons` are logged so
every intervention is explainable after the fact.

## Design properties

- **Deterministic & non-learned.** No weights, no inference — pure checks on
  measured state. Reviewable line by line, unit-tested in `tests/test_safety.py`.
- **Authoritative over motion.** It edits the joint targets *and* (via the
  runner) zeroes the base command on `OVERRIDE`/`ESTOP`, so a freeze actually
  stops the robot even in the reduced-order twin.
- **Layered, not monolithic.** The expert policy is *also* polite (it slows near
  people); the Guardian is the hard floor underneath that doesn't trust anyone.
- **Same in sim and on hardware.** Identical code path; on the Jetson the e-stop
  additionally cuts servo torque via `JetsonBackend.emergency_stop()`.

## Tuning

Thresholds live in `configs/control/safety.yaml` (`GuardianConfig`). Start
conservative (large human radius, low tip threshold) and relax with evidence
from twin + hardware testing. The geofence should match the real greenhouse
footprint.

## Out of scope here (roadmap)
Redundant hardware e-stop circuit, watchdog on the control loop, current/thermal
limits on the servo bus, and a formal hazard analysis (ISO 10218 / ISO 13482
for robots near people) to pair with the Halos-aligned software layer.
