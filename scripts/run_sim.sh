#!/usr/bin/env bash
# Run the gardener in the digital twin. Defaults to the runs-anywhere kinematic
# backend with the scripted VLA; override with args, e.g.:
#   ./scripts/run_sim.sh --backend mujoco --vla neural --vla-ckpt models/policies/vla_bc.pt --render
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"
python3 -m gardener_bdx.runtime.sim_main \
  --backend "${BACKEND:-kinematic}" \
  --goal "${GOAL:-tend the garden and water the thirsty plants}" \
  --steps "${STEPS:-3000}" \
  "$@"
