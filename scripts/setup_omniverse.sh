#!/usr/bin/env bash
# Bootstrap the NVIDIA Omniverse / Isaac Lab side of GardenerBDX on a GPU box.
#
# This automates the parts that can be automated and guides the rest (the
# Omniverse / Isaac Sim install is interactive and GPU-bound, so it can't run in
# a CI sandbox). Run it ON your RTX machine.
#
#   PREREQS: NVIDIA RTX GPU + recent driver (CUDA 12+), ~30 GB disk.
#
# Usage:
#   ISAACLAB_PATH=/opt/IsaacLab ./scripts/setup_omniverse.sh
set -euo pipefail
cd "$(dirname "$0")/.."
REPO="$(pwd)"
ISAACLAB_PATH="${ISAACLAB_PATH:-$HOME/IsaacLab}"

echo "==> GardenerBDX Omniverse setup"
echo "    repo:       $REPO"
echo "    IsaacLab:   $ISAACLAB_PATH"

# 1) GPU sanity check
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "!! nvidia-smi not found. Isaac Sim needs an NVIDIA RTX GPU + driver. Aborting." >&2
  exit 1
fi
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true

# 2) Isaac Lab presence (install is documented, not silently forced)
if [ ! -d "$ISAACLAB_PATH" ]; then
  cat <<EOF
!! Isaac Lab not found at $ISAACLAB_PATH.
   Install it (pulls Isaac Sim too):
     git clone https://github.com/isaac-sim/IsaacLab.git "$ISAACLAB_PATH"
     cd "$ISAACLAB_PATH" && ./isaaclab.sh --install
   Docs: https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html
   Then re-run this script.
EOF
  exit 1
fi

# Isaac Lab ships a wrapper that runs commands under its bundled python.
ISAACLAB="$ISAACLAB_PATH/isaaclab.sh"

# 3) Install this package into Isaac's python (core deps only; torch ships with Isaac)
echo "==> installing gardener-bdx into Isaac Lab's python"
"$ISAACLAB" -p -m pip install -e "$REPO"

# 4) Convert the robot to USD (one command)
echo "==> converting URDF -> USD"
"$ISAACLAB" -p "$REPO/scripts/convert_to_usd.py" --input "$REPO/models/robot/gardener_bdx.urdf"

# 5) Smoke-train a few iterations to confirm the env stands up
echo "==> smoke test: 5 training iterations on 64 envs"
"$ISAACLAB" -p -m gardener_bdx.training.train_isaaclab --num_envs 64 --max_iterations 5 --headless

cat <<EOF

==> Done. Full run:
    $ISAACLAB -p -m gardener_bdx.training.train_isaaclab --num_envs 4096 --headless --max_iterations 1500
Then deploy the exported models/policies/locomotion.npz in any twin:
    python -m gardener_bdx.runtime.sim_main --backend mujoco --render
EOF
