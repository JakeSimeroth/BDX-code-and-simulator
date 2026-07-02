#!/usr/bin/env python3
"""Drive the gardener live in the **Isaac Sim GUI** — the interactive twin.

    python scripts/run_isaac.py                         # GUI window, scripted VLA
    python scripts/run_isaac.py --vla neural --vla-ckpt models/policies/vla_bc.pt \
        --policy models/policies/locomotion.npz
    python scripts/run_isaac.py --headless              # no window (eval/servers)

A window opens with the robot in the greenhouse and the gardener loop running.
While it runs you can **interact as a user**: orbit/zoom the camera, pause and
single-step, drag plants or the human around, change lighting/materials, or add
props — the control loop reads live stage state every tick, so your edits flow
straight into perception, the VLA, and the safety Guardian.

Runs under Isaac's python on a GPU box. If you haven't authored a greenhouse USD
yet, a procedural one (benches/dock/human) is spawned automatically. Convert the
robot first:  `python scripts/convert_to_usd.py`.  See docs/ISAACSIM.md."""

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


def main() -> None:
    ap = argparse.ArgumentParser(description="Interactive Isaac Sim gardener")
    ap.add_argument("--vla", default="scripted", choices=["scripted", "neural", "groot"])
    ap.add_argument("--vla-ckpt", default="")
    ap.add_argument("--policy", default="", help="locomotion .npz (else config/CPG fallback)")
    ap.add_argument("--goal", default="tend the garden and water the thirsty plants")
    ap.add_argument("--robot", default="models/robot/gardener_bdx.usd")
    ap.add_argument("--scene", default="models/scenes/greenhouse.usd")
    ap.add_argument("--headless", action="store_true", help="run without the GUI window")
    ap.add_argument("--no-rtx-sensors", action="store_true",
                    help="skip the head RGB-D/LiDAR attachment (privileged semantics only)")
    args, _ = ap.parse_known_args()

    # numpy-only imports — safe before Isaac's SimulationApp boots.
    from gardener_bdx.common.config import HierarchyConfig, RobotConfig
    from gardener_bdx.policy.locomotion import LocomotionPolicy
    from gardener_bdx.policy.runner import GardenerController
    from gardener_bdx.policy.vla_brain import build_vla
    from gardener_bdx.sim.isaac_backend import IsaacGreenhouse

    rc = RobotConfig.from_yaml()
    h = HierarchyConfig.from_yaml()
    dt = 1.0 / h.locomotion_hz

    # Constructing the backend boots Isaac Sim (must happen before the heavy
    # brain imports), so build it first, then the controller.
    io = IsaacGreenhouse(rc, dt, usd_robot=args.robot, usd_scene=args.scene,
                         headless=args.headless, device=h.device,
                         rtx_sensors=not args.no_rtx_sensors)
    kwargs = {} if args.vla == "scripted" else {"checkpoint": args.vla_ckpt, "device": h.device}
    loco = LocomotionPolicy(rc, policy_path=args.policy or h.locomotion_policy_path)
    ctrl = GardenerController(rc, h, goal=args.goal, vla=build_vla(args.vla, **kwargs), locomotion=loco)
    ctrl.reset(io)

    print("Isaac Sim live — orbit/pause/drag in the window; the loop reads it each tick.")
    print(f"  gait: {'learned policy' if loco.using_learned_policy else 'CPG fallback (train one to walk)'}")
    try:
        while io.app.is_running():
            ctrl.step(io)
    except KeyboardInterrupt:
        pass
    finally:
        io.close()


if __name__ == "__main__":
    main()
