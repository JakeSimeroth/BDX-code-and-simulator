"""Generate a VLA training dataset by rolling out the scripted expert in sim.

Each tick we log the raw multimodal observation the neural VLA will consume
(proprio always; RGB-D + LiDAR-BEV when the backend renders them), the language
instruction, and the expert's decision as supervision: the continuous whole-body
action vector and the discrete skill. ``train_vla.py`` distills these into
``NeuralVLA`` (behavior cloning + flow matching).

    python -m gardener_bdx.training.collect_demos --backend mujoco --render \
        --episodes 200 --out data/expert_demos.npz

Use ``--backend isaac`` for photoreal frames once your Isaac stage is set up."""

from __future__ import annotations

import argparse

import numpy as np

from ..common.config import HierarchyConfig, RobotConfig
from ..policy.runner import GardenerController
from ..policy.vla_brain import ACTION_KEYS, ScriptedGardenerVLA
from ..policy.vla_net import _hash_tokens  # reuse the exact tokenizer
from ..runtime import loop
from ..sim import make_backend

INSTRUCTIONS = [
    "tend the garden and water the thirsty plants",
    "water plant #2",
    "patrol the greenhouse and check the plants",
    "return to the dock and charge",
    "water everything that looks dry",
]
SKILLS = ["idle", "navigate", "approach_plant", "dispense_water", "dock_charge", "recover"]


def _intent_to_action_vec(intent) -> np.ndarray:
    c = intent.locomotion
    return np.array([c.vx, c.vy, c.wz, c.body_height, c.look_yaw, c.look_pitch,
                     intent.dispense_rate_lps], dtype=np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="kinematic")
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--out", default="data/expert_demos.npz")
    args = ap.parse_args()

    rc = RobotConfig.from_yaml()
    h = HierarchyConfig.from_yaml()
    proprio, tokens, actions, skills, imgs, bevs = [], [], [], [], [], []

    for ep in range(args.episodes):
        instr = INSTRUCTIONS[ep % len(INSTRUCTIONS)]
        bk = {"render": True} if (args.render and args.backend == "mujoco") else {}
        io = make_backend(args.backend, rc, 1.0 / h.locomotion_hz, seed=1000 + ep, **bk)
        ctrl = GardenerController(rc, h, goal=instr, vla=ScriptedGardenerVLA())
        for i, info in enumerate(loop.run(ctrl, io, steps=args.steps)):
            o = info.obs
            from .locomotion_env import locomotion_observation  # local to avoid cycle at import
            from ..common.math_utils import projected_gravity
            g = projected_gravity(o.imu.orientation)
            proprio.append(np.concatenate([o.joints.positions, o.joints.velocities, g,
                                           o.imu.angular_velocity,
                                           [o.battery.state_of_charge, o.water.fraction]]).astype(np.float32))
            tokens.append(_hash_tokens(instr))
            actions.append(_intent_to_action_vec(info.intent))
            skills.append(SKILLS.index(info.intent.skill.value))
            imgs.append(o.camera.rgb.astype(np.uint8) if o.camera is not None else np.zeros((48, 64, 3), np.uint8))
        io.close()
        print(f"episode {ep+1}/{args.episodes}  samples={len(actions)}")

    import os
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez_compressed(
        args.out,
        proprio=np.stack(proprio), tokens=np.stack(tokens),
        actions=np.stack(actions), skills=np.array(skills, np.int64),
        images=np.stack(imgs), action_keys=np.array(ACTION_KEYS),
    )
    print(f"saved {len(actions)} samples -> {args.out}")


if __name__ == "__main__":
    main()
