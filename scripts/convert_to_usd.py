"""One-command robot → USD conversion for Isaac Sim / Isaac Lab.

    # under an Isaac Lab checkout's python:
    python scripts/convert_to_usd.py                      # URDF -> models/robot/gardener_bdx.usd
    python scripts/convert_to_usd.py --input models/robot/gardener_bdx.xml --type mjcf

This boots the Omniverse SimulationApp and runs Isaac Lab's asset converter, so
the Isaac backend / training env can spawn the robot. It must run inside the
Isaac environment (it imports the Omniverse stack); on a machine without Isaac it
prints how to get it. Converter cfg fields shift slightly across Isaac Lab
versions — the few used here are the stable core; tweak if your version differs.
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert the GardenerBDX robot to USD")
    ap.add_argument("--input", default=str(REPO / "models" / "robot" / "gardener_bdx.urdf"))
    ap.add_argument("--type", choices=["auto", "urdf", "mjcf"], default="auto")
    ap.add_argument("--out-dir", default=str(REPO / "models" / "robot"))
    ap.add_argument("--usd-name", default="gardener_bdx.usd")
    ap.add_argument("--fix-base", action="store_true", help="weld the base (for arm-style assets)")
    args, _ = ap.parse_known_args()

    try:
        from isaaclab.app import AppLauncher
    except Exception as e:  # pragma: no cover
        raise SystemExit(
            "This script must run inside an Isaac Lab / Isaac Sim environment.\n"
            "Install: https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html\n"
            f"(import error: {e})"
        )

    app_launcher = AppLauncher({"headless": True})
    simulation_app = app_launcher.app

    try:
        from isaaclab.sim.converters import MjcfConverter, MjcfConverterCfg, UrdfConverter, UrdfConverterCfg

        kind = args.type
        if kind == "auto":
            kind = "mjcf" if args.input.lower().endswith((".xml", ".mjcf")) else "urdf"

        if kind == "urdf":
            cfg = UrdfConverterCfg(
                asset_path=args.input,
                usd_dir=args.out_dir,
                usd_file_name=args.usd_name,
                fix_base=args.fix_base,
                merge_fixed_joints=True,
                force_usd_conversion=True,
                joint_drive=_joint_drive_cfg(UrdfConverterCfg),
            )
            converter = UrdfConverter(cfg)
        else:
            cfg = MjcfConverterCfg(
                asset_path=args.input,
                usd_dir=args.out_dir,
                usd_file_name=args.usd_name,
                fix_base=args.fix_base,
                import_sites=True,
                force_usd_conversion=True,
            )
            converter = MjcfConverter(cfg)

        print(f"[convert_to_usd] {kind.upper()} -> {converter.usd_path}")
        print("Next: point sim/isaac_backend.py and training/isaaclab_locomotion_env.py at this USD,")
        print("then `python -m gardener_bdx.training.train_isaaclab --num_envs 4096 --headless`.")
    except BaseException:
        # Kit's shutdown hooks can swallow the process exit code, turning a
        # traceback into a "successful" run — report failure explicitly.
        traceback.print_exc()
        simulation_app.close()
        import os

        os._exit(1)
    simulation_app.close()


def _joint_drive_cfg(UrdfConverterCfg):
    """PD drive gains for the USD, from the canonical robot yaml.

    Recent Isaac Lab makes ``joint_drive.gains.stiffness`` required; the training
    env re-applies these same values through ``ImplicitActuatorCfg`` at spawn, so
    the USD and the env can't drift apart.
    """
    from gardener_bdx.common.config import RobotConfig

    rc = RobotConfig.from_yaml()
    exact = lambda names, values: {rf"^{n}$": float(v) for n, v in zip(names, values)}
    return UrdfConverterCfg.JointDriveCfg(
        gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
            stiffness=exact(rc.joint_names, rc.kp),
            damping=exact(rc.joint_names, rc.kd),
        )
    )


if __name__ == "__main__":
    main()
