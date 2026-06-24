#!/usr/bin/env bash
# Launch the on-robot runner on the Jetson. Requires the hardware driver layer in
# hardware/jetson_backend.py to be wired (see docs/HARDWARE.md). Ctrl-C e-stops.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"
python3 -m gardener_bdx.runtime.robot_main \
  --goal "${GOAL:-tend the garden}" \
  --vla "${VLA:-neural}" \
  --vla-ckpt "${VLA_CKPT:-models/policies/vla_bc.pt}" \
  "$@"
