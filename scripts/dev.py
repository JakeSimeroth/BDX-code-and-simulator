#!/usr/bin/env python3
"""The GardenerBDX task runner — one canonical command per pipeline stage,
identical on Windows / Linux / macOS (no .sh/.ps1 pairs to keep in sync).

    python scripts/dev.py list                 # what can I run?
    python scripts/dev.py preflight            # is this machine ready?
    python scripts/dev.py walk-smoke           # 2-min Isaac pipeline check
    python scripts/dev.py eval                 # scoreboard -> out/eval_report.md
    python scripts/dev.py walk -- --num_envs 2048   # extra args pass through after --

Every task is a thin wrapper over the underlying `python -m ...` / script
command (printed before it runs), so nothing is hidden — copy the printed
command any time you want to customize beyond `--` passthrough. This is also
the surface a Claude Code session drives, so human and agent iterate with the
same verbs."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PY = sys.executable

# name -> (command argv, description). Ordered as the actual workflow.
TASKS: dict[str, tuple[list[str], str]] = {
    "preflight": ([PY, "scripts/preflight.py"],
                  "readiness checklist: CUDA/GPU, MuJoCo, Isaac, smoke tests"),
    "test": ([PY, "-m", "pytest", "-q"],
             "run the full test suite"),
    "gif": ([PY, "scripts/view_kinematic.py", "--gif", "out/gardener.gif"],
            "render the task behavior to out/gardener.gif (no GPU needed)"),
    "usd": ([PY, "scripts/convert_to_usd.py"],
            "convert the robot URDF -> USD for Isaac"),
    "walk-smoke": ([PY, "-m", "gardener_bdx.training.train_isaaclab", "--smoke", "--headless"],
                   "~2-min Isaac Lab pipeline check (64 envs / 20 iters) — run BEFORE 'walk'"),
    "walk": ([PY, "-m", "gardener_bdx.training.train_isaaclab",
              "--num_envs", "4096", "--headless"],
             "train the locomotion policy in Isaac Lab -> models/policies/locomotion.npz"),
    "watch": ([PY, "scripts/view_mujoco.py", "--view",
               "--policy", "models/policies/locomotion.npz"],
              "watch the trained gait in the MuJoCo viewer"),
    "isaac": ([PY, "scripts/run_isaac.py",
               "--policy", "models/policies/locomotion.npz"],
              "interactive Isaac Sim twin (drag plants/person, live loop)"),
    "demos": ([PY, "-m", "gardener_bdx.training.collect_demos",
               "--backend", "mujoco", "--render", "--episodes", "200"],
              "collect expert demonstrations (BC data) -> data/expert_demos.npz"),
    "dagger": ([PY, "-m", "gardener_bdx.training.collect_demos",
                "--backend", "mujoco", "--render", "--episodes", "100",
                "--driver", "neural", "--vla-ckpt", "models/policies/vla_bc.pt",
                "--out", "data/dagger_demos.npz"],
               "DAgger round: neural drives, expert labels -> data/dagger_demos.npz"),
    "vla": ([PY, "-m", "gardener_bdx.training.train_vla",
             "--epochs", "30", "--device", "cuda"],
            "distill the expert into the NeuralVLA (add --data a.npz,b.npz for +DAgger)"),
    "eval": ([PY, "-m", "gardener_bdx.training.evaluate", "--compare",
              "--vla-ckpt", "models/policies/vla_bc.pt"],
             "teacher-vs-learned scoreboard -> out/eval_report.{md,json}"),
}


def main(argv: list[str]) -> int:
    if not argv or argv[0] in ("list", "-h", "--help"):
        width = max(len(n) for n in TASKS)
        print("GardenerBDX tasks (python scripts/dev.py <task> [-- extra args]):\n")
        for name, (_, desc) in TASKS.items():
            print(f"  {name:<{width}}  {desc}")
        print("\nTypical first GPU day: preflight -> gif -> usd -> walk-smoke -> walk -> watch")
        print("                        -> isaac -> demos -> vla -> eval -> dagger -> vla -> eval")
        return 0

    name, *extra = argv
    if extra and extra[0] == "--":
        extra = extra[1:]
    if name not in TASKS:
        print(f"unknown task {name!r} — try: python scripts/dev.py list")
        return 2
    cmd = TASKS[name][0] + extra
    print(f"[dev:{name}] {' '.join(cmd)}")
    return subprocess.call(cmd, cwd=REPO)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
