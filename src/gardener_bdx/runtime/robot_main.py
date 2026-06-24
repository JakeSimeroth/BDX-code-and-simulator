"""On-robot entrypoint for the Jetson.

    python -m gardener_bdx.runtime.robot_main --goal "tend the garden"

This is intentionally the *same* control graph as the digital twin — the only
difference is that ``build_drivers()`` returns real hardware drivers instead of
a simulator. Fill in :func:`build_drivers` with your actuator bus, IMU, cameras,
LiDAR, battery, water system, and SLAM (see
:mod:`gardener_bdx.hardware.jetson_backend`). A SIGINT trips the Guardian e-stop
before exit."""

from __future__ import annotations

import argparse
import signal

from ..common.config import HierarchyConfig, RobotConfig
from ..hardware.jetson_backend import JetsonBackend
from ..policy.runner import GardenerController
from ..policy.vla_brain import build_vla
from ..safety.guardian import GuardianConfig, SafetyGuardian
from ..runtime import loop


def build_drivers(rc: RobotConfig):
    """Return (actuators, imu, battery, water, camera, lidar, slam) for the
    physical robot. Unimplemented by design — this is the hardware bring-up seam."""
    raise NotImplementedError(
        "Wire your hardware drivers here (Feetech/Dynamixel bus, BNO085 IMU, "
        "RealSense, LiDAR, INA226, pump/valve, SLAM). See docs/HARDWARE.md."
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="GardenerBDX on-robot runner")
    ap.add_argument("--goal", default="tend the garden")
    ap.add_argument("--vla", default="neural", choices=["scripted", "neural", "groot"])
    ap.add_argument("--vla-ckpt", default="models/policies/vla_bc.pt")
    args = ap.parse_args()

    rc = RobotConfig.from_yaml()
    h = HierarchyConfig.from_yaml()
    dt = 1.0 / h.locomotion_hz

    actuators, imu, battery, water, camera, lidar, slam = build_drivers(rc)
    io = JetsonBackend(rc, dt, actuators, imu, battery, water, camera, lidar, slam)

    vla_kwargs = {} if args.vla == "scripted" else {"checkpoint": args.vla_ckpt, "device": h.device}
    guardian = SafetyGuardian(rc, dt, GuardianConfig.from_yaml())
    ctrl = GardenerController(rc, h, goal=args.goal, vla=build_vla(args.vla, **vla_kwargs),
                              guardian=guardian)

    # Hardware e-stop on Ctrl-C: latch the Guardian and cut torque/water.
    def _estop(signum, frame):
        guardian.trip_estop()
        io.emergency_stop()
        raise SystemExit("E-STOP")

    signal.signal(signal.SIGINT, _estop)

    print("GardenerBDX online. Ctrl-C to e-stop.")
    try:
        for _ in loop.run(ctrl, io):  # runs until stopped
            pass
    finally:
        io.close()


if __name__ == "__main__":
    main()
