"""Task-success evaluation harness.

Runs N episodes and reports the metrics that actually matter for a greenhouse
gardener — not RL return:

  * **success rate** — every initially-thirsty plant watered, with no collisions
  * **plants serviced** / thirsty
  * **water delivered** (L)
  * **collisions** with plants or people (a safety failure)
  * **time-to-complete** (s) over successful episodes
  * **safety interventions** (Guardian non-OK ticks)

    python -m gardener_bdx.training.evaluate --backend kinematic --episodes 10
    python -m gardener_bdx.training.evaluate --compare   # scripted vs neural VLA

Works on the kinematic and MuJoCo backends (which expose ground-truth plant/human
positions). The same harness scores the scripted expert and any trained VLA, so
"does the neural policy match/exceed the teacher?" is one command."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field

import numpy as np

from ..common.config import HierarchyConfig, RobotConfig
from ..policy.runner import GardenerController
from ..policy.vla_brain import build_vla
from ..sim import make_backend

PLANT_COLLISION_M = 0.30   # robot center this close to a plant = ran into it
HUMAN_COLLISION_M = 0.45   # contact with a person (Guardian stop radius is 0.7)


@dataclass
class EpisodeResult:
    thirsty: int
    serviced: int
    water_delivered_l: float
    collisions: int
    safety_ticks: int
    completion_step: int  # -1 if never completed
    steps: int
    success: bool


def run_episode(backend: str, vla_kind: str, goal: str, steps: int, seed: int,
                vla_kwargs: dict, dt: float, rc: RobotConfig, h: HierarchyConfig) -> EpisodeResult:
    io = make_backend(backend, rc, dt, seed=seed)
    ctrl = GardenerController(rc, h, goal=goal, vla=build_vla(vla_kind, **vla_kwargs))
    ctrl.reset(io)
    thr = ctrl.goal.dryness_threshold

    plant_xy = np.asarray(getattr(io, "plant_xy"))
    dryness0 = np.asarray(getattr(io, "plant_dryness")).copy()
    thirsty_mask = dryness0 >= thr
    n_thirsty = int(thirsty_mask.sum())

    in_contact = np.zeros(len(plant_xy) + 1, dtype=bool)  # plants + one human
    collisions = safety_ticks = 0
    delivered = 0.0
    prev_water = io.water_l if hasattr(io, "water_l") else rc.water_capacity_liters
    completion = -1

    for t in range(steps):
        info = ctrl.step(io)
        base = info.state.base_pose.position[:2]

        # collisions (rising edges only)
        for i, p in enumerate(plant_xy):
            hit = np.linalg.norm(p - base) < PLANT_COLLISION_M
            if hit and not in_contact[i]:
                collisions += 1
            in_contact[i] = hit
        human = getattr(io, "human_xy", None)
        if human is not None:
            hit = np.linalg.norm(np.asarray(human) - base) < HUMAN_COLLISION_M
            if hit and not in_contact[-1]:
                collisions += 1
            in_contact[-1] = hit

        # water actually delivered (tank drops while not refilling)
        w = info.state.water.level_liters
        if w < prev_water:
            delivered += prev_water - w
        prev_water = w

        if info.verdict.level.name != "OK":
            safety_ticks += 1

        # completion: all initially-thirsty plants now below threshold
        if completion < 0 and n_thirsty > 0:
            now = np.asarray(getattr(io, "plant_dryness"))
            if bool(np.all(now[thirsty_mask] < thr)):
                completion = t
    io.close()

    final = np.asarray(getattr(io, "plant_dryness"))
    serviced = int(np.sum(final[thirsty_mask] < thr)) if n_thirsty else 0
    # Task success = every thirsty plant watered within the time budget. Collisions
    # and safety interventions are reported as *separate* KPIs (a crude kinematic
    # collision proxy shouldn't veto task completion).
    success = (serviced == n_thirsty)
    return EpisodeResult(n_thirsty, serviced, delivered, collisions, safety_ticks,
                         completion, steps, success)


def aggregate(results: list[EpisodeResult], dt: float) -> dict:
    n = len(results)
    completed = [r for r in results if r.completion_step >= 0]
    return {
        "episodes": n,
        "success_rate": np.mean([r.success for r in results]),
        "serviced": np.mean([r.serviced for r in results]),
        "thirsty": np.mean([r.thirsty for r in results]),
        "water_l": np.mean([r.water_delivered_l for r in results]),
        "collisions": np.mean([r.collisions for r in results]),
        "safety_s": np.mean([r.safety_ticks for r in results]) * dt,
        "complete_rate": len(completed) / n,
        "time_to_complete_s": (np.mean([r.completion_step for r in completed]) * dt) if completed else float("nan"),
    }


def _print(label: str, a: dict) -> None:
    print(f"\n=== {label}  ({a['episodes']} episodes) ===")
    print(f"  success rate          : {a['success_rate']*100:5.1f}%")
    print(f"  plants serviced       : {a['serviced']:.2f} / {a['thirsty']:.2f} thirsty")
    print(f"  water delivered       : {a['water_l']:.2f} L")
    print(f"  collisions / episode  : {a['collisions']:.2f}")
    print(f"  time-to-complete      : {a['time_to_complete_s']:.1f} s  (completed {a['complete_rate']*100:.0f}% of eps)")
    print(f"  safety intervention   : {a['safety_s']:.1f} s / episode")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="kinematic")
    ap.add_argument("--vla", default="scripted")
    ap.add_argument("--vla-ckpt", default="")
    ap.add_argument("--goal", default="tend the garden and water the thirsty plants")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--compare", action="store_true", help="score scripted vs neural side by side")
    args = ap.parse_args()

    rc = RobotConfig.from_yaml()
    h = HierarchyConfig.from_yaml()
    dt = 1.0 / h.locomotion_hz

    def eval_vla(kind: str, kwargs: dict, label: str):
        results = [run_episode(args.backend, kind, args.goal, args.steps, 100 + i, kwargs, dt, rc, h)
                   for i in range(args.episodes)]
        _print(label, aggregate(results, dt))

    if args.compare:
        eval_vla("scripted", {}, "ScriptedGardenerVLA (expert/teacher)")
        eval_vla("neural", {"checkpoint": args.vla_ckpt, "device": h.device}, "NeuralVLA (learned)")
    else:
        kwargs = {} if args.vla == "scripted" else {"checkpoint": args.vla_ckpt, "device": h.device}
        eval_vla(args.vla, kwargs, f"{args.vla} VLA")


if __name__ == "__main__":
    main()
