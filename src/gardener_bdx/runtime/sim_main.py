"""Run the gardener in the digital twin.

    python -m gardener_bdx.runtime.sim_main --backend kinematic --steps 3000 \
        --goal "tend the garden and water the thirsty plants"

    # full physics + VLA (needs the sim/vla extras and a trained checkpoint):
    python -m gardener_bdx.runtime.sim_main --backend mujoco --vla neural \
        --vla-ckpt models/policies/vla_bc.pt --render

Identical control graph as the robot; only ``--backend`` changes."""

from __future__ import annotations

import argparse

import numpy as np

from ..common.config import HierarchyConfig, RobotConfig
from ..policy.runner import GardenerController
from ..policy.vla_brain import build_vla
from ..safety.guardian import GuardianConfig, SafetyGuardian
from ..sim import make_backend


def main() -> None:
    ap = argparse.ArgumentParser(description="GardenerBDX digital-twin runner")
    ap.add_argument("--backend", default="kinematic", choices=["kinematic", "mujoco", "isaac"])
    ap.add_argument("--vla", default="scripted", choices=["scripted", "neural", "groot"])
    ap.add_argument("--vla-ckpt", default="")
    ap.add_argument("--goal", default="tend the garden and water the thirsty plants")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    rc = RobotConfig.from_yaml()
    h = HierarchyConfig.from_yaml()
    dt = 1.0 / h.locomotion_hz

    bk = {"seed": args.seed}
    if args.render and args.backend != "kinematic":
        bk["render"] = True
    io = make_backend(args.backend, rc, dt, **bk)

    vla_kwargs = {} if args.vla == "scripted" else {"checkpoint": args.vla_ckpt, "device": h.device}
    vla = build_vla(args.vla, **vla_kwargs)
    guardian = SafetyGuardian(rc, dt, GuardianConfig.from_yaml())
    ctrl = GardenerController(rc, h, goal=args.goal, vla=vla, guardian=guardian)

    # Single reset so our baseline measurement is honest.
    ctrl.reset(io)
    dry0 = getattr(io, "plant_dryness", np.array([])).copy()

    skills: dict[str, int] = {}
    safety_counts: dict[str, int] = {}
    last = None
    for i in range(args.steps):
        info = ctrl.step(io)
        skills[info.intent.skill.value] = skills.get(info.intent.skill.value, 0) + 1
        safety_counts[info.verdict.level.name] = safety_counts.get(info.verdict.level.name, 0) + 1
        last = info
        if not args.quiet and i % 250 == 0:
            print(f"t={info.stamp:6.1f}s  {info.intent.skill.value:<14} "
                  f"soc={info.state.battery.state_of_charge:.2f} "
                  f"water={info.state.water.level_liters:.2f}L  "
                  f"safety={info.verdict.level.name:<8} :: {info.intent.rationale}")
    io.close()

    print("\n================ rollout summary ================")
    print(f"backend={args.backend}  vla={args.vla}  steps={args.steps} ({args.steps*dt:.0f}s sim)")
    print("time per skill:", {k: f"{v*dt:.1f}s" for k, v in skills.items()})
    print("safety levels:", safety_counts)
    if last is not None:
        print(f"final SoC={last.state.battery.state_of_charge:.2f}  "
              f"water={last.state.water.level_liters:.2f}L")
        print(f"world belief: plants={len(last.world.plants)} humans={len(last.world.humans)} "
              f"coverage={last.world.occupancy.coverage_fraction()*100:.0f}%")
    if dry0.size and hasattr(io, "plant_dryness"):
        thr = ctrl.goal.dryness_threshold
        serviced = int(np.sum(io.plant_dryness < thr))
        print(f"plant dryness start: {np.round(dry0, 2)}")
        print(f"plant dryness end:   {np.round(io.plant_dryness, 2)}")
        print(f"plants below goal threshold ({thr:.2f}): {serviced}/{io.n_plants}")
    print("=================================================")


if __name__ == "__main__":
    main()
