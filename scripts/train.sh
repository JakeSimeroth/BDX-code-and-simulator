#!/usr/bin/env bash
# Full training pipeline: (1) RL the locomotion gait and export a numpy policy,
# (2) collect scripted-expert demos, (3) distill the end-to-end VLA from pixels.
# Requires the train/vla/sim extras: pip install -e '.[all]'
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"

echo "[1/3] locomotion RL (export -> models/policies/locomotion.npz)"
python3 -m gardener_bdx.training.train_locomotion \
  --backend "${BACKEND:-mujoco}" --num-envs "${NUM_ENVS:-8}" --timesteps "${TIMESTEPS:-5000000}"

echo "[2/3] collect expert demos -> data/expert_demos.npz"
python3 -m gardener_bdx.training.collect_demos \
  --backend "${BACKEND:-mujoco}" --render --episodes "${EPISODES:-200}"

echo "[3/3] distill the VLA -> models/policies/vla_bc.pt"
python3 -m gardener_bdx.training.train_vla --epochs "${EPOCHS:-30}"
echo "done."
