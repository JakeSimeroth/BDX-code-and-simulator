"""Watch the gardener in the MuJoCo physics twin.

    python scripts/view_mujoco.py --view              # interactive window (needs a display)
    python scripts/view_mujoco.py --steps 2000        # headless, just step (for CI/servers)
    python scripts/view_mujoco.py --view --vla neural --vla-ckpt models/policies/vla_bc.pt

This is the right tool for **gait debugging** (contact, balance) once you've
trained a locomotion policy — set paths.locomotion_policy in
configs/control/hierarchy.yaml or pass --policy. With the CPG fallback (no
trained policy) the droid will not balance; that's expected. Needs `mujoco`."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from gardener_bdx.common.config import HierarchyConfig, RobotConfig  # noqa: E402
from gardener_bdx.policy.locomotion import LocomotionPolicy  # noqa: E402
from gardener_bdx.policy.runner import GardenerController  # noqa: E402
from gardener_bdx.policy.vla_brain import build_vla  # noqa: E402
from gardener_bdx.sim import make_backend  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--view", action="store_true", help="open the interactive MuJoCo window")
    ap.add_argument("--steps", type=int, default=3000, help="headless step count when not --view")
    ap.add_argument("--vla", default="scripted")
    ap.add_argument("--vla-ckpt", default="")
    ap.add_argument("--policy", default="", help="locomotion .npz (overrides config); empty=CPG fallback")
    ap.add_argument("--goal", default="tend the garden")
    args = ap.parse_args()

    rc = RobotConfig.from_yaml()
    h = HierarchyConfig.from_yaml()
    dt = 1.0 / h.locomotion_hz
    io = make_backend("mujoco", rc, dt, seed=0)

    kwargs = {} if args.vla == "scripted" else {"checkpoint": args.vla_ckpt, "device": h.device}
    loco = LocomotionPolicy(rc, policy_path=args.policy or h.locomotion_policy_path)
    ctrl = GardenerController(rc, h, goal=args.goal, vla=build_vla(args.vla, **kwargs), locomotion=loco)
    ctrl.reset(io)
    print(f"locomotion: {'learned policy' if loco.using_learned_policy else 'CPG fallback (train a policy to balance)'}")

    if args.view:
        try:
            import mujoco.viewer
        except Exception as e:
            raise SystemExit(f"--view needs mujoco + a display: {e}")
        with mujoco.viewer.launch_passive(io.model, io.data) as viewer:
            while viewer.is_running():
                ctrl.step(io)      # advances physics
                viewer.sync()      # reflect io.data in the window
    else:
        for t in range(args.steps):
            info = ctrl.step(io)
            if t % 500 == 0:
                z = info.state.base_pose.position[2]
                print(f"t={info.stamp:5.1f}s  z={z:.3f}  skill={info.intent.skill.value:<14} safety={info.verdict.level.name}")
        print("done (headless). Use --view on a machine with a display to watch it.")


if __name__ == "__main__":
    main()
