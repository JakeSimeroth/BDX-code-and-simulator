"""First-boot preflight for GardenerBDX on a GPU machine.

Run this the moment you boot the project on your RTX box — it tells you, in one
screen, what's installed, whether CUDA is live, and what to run next:

    python scripts/preflight.py

It never modifies anything; it just probes and prints a checklist. Core failures
(package import, robot config, kinematic rollout) exit non-zero so CI can gate on
it; missing optional stacks (Isaac, GR00T) are reported as TODO, not errors."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

OK, WARN, BAD = "\033[92m✓\033[0m", "\033[93m•\033[0m", "\033[91m✗\033[0m"


def has(mod: str) -> bool:
    try:
        return importlib.util.find_spec(mod) is not None
    except Exception:
        return False


def main() -> int:
    print("=" * 64)
    print(" GardenerBDX preflight")
    print("=" * 64)
    core_ok = True

    # --- interpreter: Isaac Sim pip wheels exist ONLY for Python 3.11 ------ #
    py = sys.version_info
    print(f"\n[python] {sys.version.split()[0]}  ({sys.executable})")
    if (py.major, py.minor) == (3, 11):
        print(f"{OK} Python 3.11 — matches the Isaac Sim pip requirement")
    else:
        print(f"{WARN} Python {py.major}.{py.minor}: `pip install isaacsim` needs **3.11**. "
              "The core stack runs here, but build the Isaac venv with 3.11 "
              "(Windows: `py -3.11 -m venv ...`).")

    # --- core package ---------------------------------------------------- #
    try:
        import gardener_bdx
        from gardener_bdx.common.config import RobotConfig

        rc = RobotConfig.from_yaml()
        print(f"{OK} gardener_bdx {gardener_bdx.__version__}  |  robot '{rc.name}', {rc.n_joints} joints")
    except Exception as e:
        print(f"{BAD} core import failed: {e}")
        return 1

    # --- numerics + CUDA ------------------------------------------------- #
    if has("torch"):
        import torch

        cuda = torch.cuda.is_available()
        mark = OK if cuda else WARN
        print(f"{mark} torch {torch.__version__}  |  CUDA: {cuda}")
        if cuda:
            p = torch.cuda.get_device_properties(0)
            print(f"    GPU: {p.name}  ({p.total_memory/1e9:.1f} GB, sm_{p.major}{p.minor})")
        else:
            print("    -> CPU only. On your RTX box install a CUDA torch build:")
            print("       pip install torch --index-url https://download.pytorch.org/whl/cu124")
    else:
        print(f"{WARN} torch not installed  (pip install -e '.[vla]')")

    # --- simulators ------------------------------------------------------ #
    print(f"{OK if has('mujoco') else WARN} mujoco {'present' if has('mujoco') else 'missing  (pip install -e .[sim])'}")
    isaacsim_ok = has("isaacsim")
    print(f"{OK if isaacsim_ok else WARN} Isaac Sim {'present (pip)' if isaacsim_ok else 'not found  (pip install \"isaacsim[all,extscache]\" --extra-index-url https://pypi.nvidia.com)'}")
    if isaacsim_ok:
        import os

        if os.environ.get("OMNI_KIT_ACCEPT_EULA", "").upper() not in ("YES", "Y", "1"):
            print("    note: first Isaac boot asks for the NVIDIA EULA. For scripted runs set:")
            print("          $env:OMNI_KIT_ACCEPT_EULA='YES'   (PowerShell)  /  export OMNI_KIT_ACCEPT_EULA=YES")
        print("    note: the very first boot compiles shaders — several minutes of apparent hang is normal.")
    isaac = has("isaaclab") or has("omni.isaac.lab")
    print(f"{OK if isaac else WARN} Isaac Lab {'present' if isaac else 'not found  (install on the GPU box; see docs/ISAACLAB.md)'}")
    print(f"{OK if has('gr00t') else WARN} Isaac-GR00T {'present' if has('gr00t') else 'not found  (optional; see docs/GROOT.md)'}")
    for opt in ["gymnasium", "stable_baselines3", "pandas", "matplotlib", "imageio"]:
        print(f"{OK if has(opt) else WARN} {opt} {'present' if has(opt) else 'missing'}")

    # --- functional smoke: the gardener loop runs ------------------------ #
    print("\n[smoke] running 200 kinematic control ticks ...")
    try:
        from gardener_bdx.common.config import HierarchyConfig
        from gardener_bdx.policy.runner import GardenerController
        from gardener_bdx.sim import make_backend

        h = HierarchyConfig.from_yaml()
        io = make_backend("kinematic", rc, 1.0 / h.locomotion_hz, seed=0)
        ctrl = GardenerController(rc, h, goal="tend the garden")
        ctrl.reset(io)
        for _ in range(200):
            info = ctrl.step(io)
        print(f"{OK} control loop OK  (last skill={info.intent.skill.value}, safety={info.verdict.level.name})")
    except Exception as e:
        print(f"{BAD} control loop failed: {e}")
        core_ok = False

    # --- functional smoke: neural VLA on GPU ----------------------------- #
    if has("torch"):
        import torch

        dev = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"\n[smoke] building NeuralVLA on {dev} (first call compiles lazy layers) ...")
        try:
            import time

            from gardener_bdx.policy.vla_brain import build_vla
            from gardener_bdx.policy.task import TaskGoal
            from gardener_bdx.policy.vla_net import _zeros_obs

            t0 = time.time()
            vla = build_vla("neural", checkpoint="", device=dev)
            for _ in range(5):
                vla.act(_zeros_obs(), TaskGoal.parse("water the thirsty plants"))
            n = sum(p.numel() for p in vla.net.parameters())
            print(f"{OK} NeuralVLA OK  ({n/1e6:.1f}M params, history={vla.net.history_len}, {time.time()-t0:.1f}s incl. warmup)")
        except Exception as e:
            print(f"{WARN} NeuralVLA smoke skipped/failed: {e}")

    # --- next steps ------------------------------------------------------ #
    print("\n" + "=" * 64)
    print(" next steps")
    print("=" * 64)
    print("  the canonical task runner (same commands for you and Claude):")
    print("    python scripts/dev.py list")
    print("  see it move (kinematic, any machine):")
    print("    python scripts/dev.py gif")
    if isaac or isaacsim_ok:
        print("  robot -> USD, then validate the Isaac pipeline in ~2 min BEFORE the long run:")
        print("    python scripts/dev.py usd")
        print("    python scripts/dev.py walk-smoke")
        print("  then the real training + see it:")
        print("    python scripts/dev.py walk")
        print("    python scripts/dev.py isaac")
    else:
        setup = "scripts\\setup_omniverse.ps1" if sys.platform == "win32" else "./scripts/setup_omniverse.sh"
        print(f"  install Isaac Sim + Isaac Lab (docs/GETTING_STARTED_GPU.md), then: {setup}")
    print("  score task success (writes out/eval_report.md):")
    print("    python scripts/dev.py eval")
    print("=" * 64)
    return 0 if core_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
