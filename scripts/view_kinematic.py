"""Top-down visualization of the gardener in the kinematic twin.

Fast, runs anywhere (no GPU/physics) — the quickest way to *see* the behavior:
the robot's path, plants colored by dryness (green=watered, red=parched), the
dock, the roaming human, and watering events.

    python scripts/view_kinematic.py --gif out/gardener.gif          # animated
    python scripts/view_kinematic.py --png out/gardener.png          # final frame
    python scripts/view_kinematic.py --vla neural --vla-ckpt ck.pt   # watch a trained VLA

Needs matplotlib (+ imageio for GIF): pip install -e '.[viz]'."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from gardener_bdx.common.config import HierarchyConfig, RobotConfig  # noqa: E402
from gardener_bdx.policy.runner import GardenerController  # noqa: E402
from gardener_bdx.policy.vla_brain import build_vla  # noqa: E402
from gardener_bdx.sim import make_backend  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vla", default="scripted")
    ap.add_argument("--vla-ckpt", default="")
    ap.add_argument("--goal", default="tend the garden and water the thirsty plants")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--gif", default="")
    ap.add_argument("--png", default="out/gardener.png")
    ap.add_argument("--decim", type=int, default=8, help="record every Nth tick")
    args = ap.parse_args()

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        raise SystemExit("Needs matplotlib: pip install -e '.[viz]'")

    rc = RobotConfig.from_yaml()
    h = HierarchyConfig.from_yaml()
    io = make_backend("kinematic", rc, 1.0 / h.locomotion_hz, seed=args.seed)
    kwargs = {} if args.vla == "scripted" else {"checkpoint": args.vla_ckpt, "device": h.device}
    ctrl = GardenerController(rc, h, goal=args.goal, vla=build_vla(args.vla, **kwargs))
    ctrl.reset(io)

    arena = io.arena
    frames = []  # (robot_xy, yaw, dryness, human_xy, dispensing)
    trail = []
    for t in range(args.steps):
        info = ctrl.step(io)
        trail.append(io.base_xy.copy())
        if t % args.decim == 0:
            frames.append((io.base_xy.copy(), io.base_yaw, io.plant_dryness.copy(),
                           io.human_xy.copy(), info.intent.skill.value == "dispense_water",
                           list(trail)))

    def draw(ax, robot_xy, yaw, dryness, human_xy, dispensing, tr):
        ax.clear()
        ax.set_xlim(-arena, arena); ax.set_ylim(-arena, arena); ax.set_aspect("equal")
        ax.add_patch(plt.Rectangle((-arena, -arena), 2 * arena, 2 * arena, fill=False, ec="0.7"))
        # plants colored by dryness (0 green -> 1 red)
        for p, d in zip(io.plant_xy, dryness):
            ax.add_patch(plt.Circle(p, 0.18, color=(float(d), float(1 - d) * 0.7, 0.1)))
        ax.add_patch(plt.Rectangle(io.dock_xy - 0.2, 0.4, 0.4, color="gold"))  # dock
        ax.plot(*human_xy, "o", color="crimson", ms=10)                        # human
        if tr:
            tr = np.array(tr); ax.plot(tr[:, 0], tr[:, 1], "-", color="0.6", lw=0.8)
        # robot as an oriented triangle
        ax.plot(*robot_xy, marker=(3, 0, np.degrees(yaw) - 90), color="navy", ms=16)
        if dispensing:
            ax.add_patch(plt.Circle(robot_xy + np.array([np.cos(yaw), np.sin(yaw)]) * 0.4, 0.12,
                                    color="deepskyblue", alpha=0.6))
        ax.set_title(f"GardenerBDX — {args.vla} VLA")

    out_dir = Path(args.png).parent if args.png else Path("out")
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 6))

    if args.gif:
        try:
            import imageio.v2 as imageio
        except ImportError:
            raise SystemExit("GIF needs imageio: pip install -e '.[viz]'")
        Path(args.gif).parent.mkdir(parents=True, exist_ok=True)
        with imageio.get_writer(args.gif, fps=15) as w:
            for fr in frames:
                draw(ax, *fr)
                fig.canvas.draw()
                w.append_data(np.asarray(fig.canvas.buffer_rgba())[..., :3])
        print(f"wrote {args.gif}  ({len(frames)} frames)")

    if args.png:
        draw(ax, *frames[-1])
        fig.savefig(args.png, dpi=110)
        print(f"wrote {args.png}")


if __name__ == "__main__":
    main()
