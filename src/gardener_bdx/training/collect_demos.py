"""Generate a VLA training dataset by rolling out a driver policy in sim while
the scripted expert labels every visited state.

Each tick we log the raw multimodal observation the neural VLA will consume
(proprio always; RGB-D + LiDAR-BEV when the backend renders them), the language
instruction, and the **expert's** decision as supervision: the continuous
whole-body action vector, the discrete skill, and the expression.
``train_vla.py`` distills these into ``NeuralVLA`` (behavior cloning + flow
matching).

Two drivers:

  * ``--driver expert`` (default) — classic BC data: the expert drives and
    labels its own states.
  * ``--driver neural --vla-ckpt models/policies/vla_bc.pt`` — **DAgger**: the
    *learned* policy drives (so the dataset covers the states it actually
    reaches, including its mistakes) while the expert supplies the corrective
    label at each of those states. Retraining on BC + DAgger data is the
    standard fix for behavior-cloning covariate shift: the student learns how
    to recover from its own drift.

    python -m gardener_bdx.training.collect_demos --backend mujoco --render \
        --episodes 200 --out data/expert_demos.npz
    python -m gardener_bdx.training.collect_demos --backend mujoco --render \
        --driver neural --vla-ckpt models/policies/vla_bc.pt \
        --episodes 100 --out data/dagger_demos.npz

Then retrain on both: ``train_vla --data data/expert_demos.npz,data/dagger_demos.npz``.
Use ``--backend isaac`` for photoreal frames once your Isaac stage is set up."""

from __future__ import annotations

import argparse

import numpy as np

from ..common.config import HierarchyConfig, RobotConfig
from ..policy.animation import EXPRESSION_ORDER
from ..policy.runner import GardenerController
from ..policy.vla_brain import ACTION_KEYS, ScriptedGardenerVLA
from ..policy.vla_net import BEV_HW, BEV_RANGE, _hash_tokens  # reuse the exact tokenizer/raster
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


def _lidar_to_bev(lidar) -> np.ndarray:
    """Rasterize a LiDAR scan into the *same* BEV grid the VLA consumes at
    inference (vla_net._obs_to_tensors), so collected frames match what the
    deployed net sees."""
    bev = np.zeros((BEV_HW, BEV_HW), np.float32)
    p = lidar.points
    if p.shape[0]:
        ij = ((p[:, :2] + BEV_RANGE) / (2 * BEV_RANGE) * BEV_HW).astype(int)
        m = (ij[:, 0] >= 0) & (ij[:, 0] < BEV_HW) & (ij[:, 1] >= 0) & (ij[:, 1] < BEV_HW)
        for i, j in ij[m]:
            bev[j, i] = 1.0
    return bev


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="kinematic")
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--out", default="data/expert_demos.npz")
    ap.add_argument("--driver", choices=["expert", "neural"], default="expert",
                    help="who drives the robot; the expert always labels (neural = DAgger)")
    ap.add_argument("--vla-ckpt", default="",
                    help="checkpoint for the neural driver (DAgger); '' = untrained net")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    rc = RobotConfig.from_yaml()
    h = HierarchyConfig.from_yaml()
    proprio, tokens, actions, skills, expressions, imgs = [], [], [], [], [], []
    depths, bevs = [], []
    instr_per_sample: list[str] = []
    episode_ends: list[int] = []

    def make_driver():
        """The policy that moves the robot. Supervision always comes from the
        expert; with a neural driver this is DAgger (student states, teacher
        labels)."""
        if args.driver == "neural":
            from ..policy.vla_brain import build_vla

            return build_vla("neural", checkpoint=args.vla_ckpt, device=args.device)
        return ScriptedGardenerVLA()

    for ep in range(args.episodes):
        instr = INSTRUCTIONS[ep % len(INSTRUCTIONS)]
        bk = {"render": True} if (args.render and args.backend == "mujoco") else {}
        io = make_backend(args.backend, rc, 1.0 / h.locomotion_hz, seed=1000 + ep, **bk)
        ctrl = GardenerController(rc, h, goal=instr, vla=make_driver())
        labeler = ScriptedGardenerVLA()  # fresh expert state each episode
        for i, info in enumerate(loop.run(ctrl, io, steps=args.steps)):
            o = info.obs
            # The label: what the EXPERT would do in the state the driver reached.
            label = (info.intent if args.driver == "expert"
                     else labeler.act(o, ctrl.goal, info.world))
            from ..common.math_utils import projected_gravity
            g = projected_gravity(o.imu.orientation)
            proprio.append(np.concatenate([o.joints.positions, o.joints.velocities, g,
                                           o.imu.angular_velocity,
                                           [o.battery.state_of_charge, o.water.fraction]]).astype(np.float32))
            tokens.append(_hash_tokens(instr))
            actions.append(_intent_to_action_vec(label))
            skills.append(SKILLS.index(label.skill.value))
            expressions.append(EXPRESSION_ORDER.index(label.expression))
            if o.camera is not None:
                imgs.append(o.camera.rgb.astype(np.uint8))
                depths.append((o.camera.depth if o.camera.depth is not None
                               else np.zeros(o.camera.rgb.shape[:2], np.float32)).astype(np.float32))
            else:
                imgs.append(np.zeros((48, 64, 3), np.uint8))
                depths.append(np.zeros((48, 64), np.float32))
            bevs.append(_lidar_to_bev(o.lidar) if o.lidar is not None
                        else np.zeros((BEV_HW, BEV_HW), np.float32))
            instr_per_sample.append(instr)
        io.close()
        episode_ends.append(len(actions))  # cumulative sample count = episode boundary
        print(f"episode {ep+1}/{args.episodes}  samples={len(actions)}")

    import os
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez_compressed(
        args.out,
        proprio=np.stack(proprio), tokens=np.stack(tokens),
        actions=np.stack(actions), skills=np.array(skills, np.int64),
        expressions=np.array(expressions, np.int64),
        expression_names=np.array([e.value for e in EXPRESSION_ORDER]),
        images=np.stack(imgs), depths=np.stack(depths), bev=np.stack(bevs),
        action_keys=np.array(ACTION_KEYS),
        episode_ends=np.array(episode_ends, np.int64),
        instructions=np.array(instr_per_sample),
        driver=np.array(args.driver),
    )
    mode = "DAgger (neural drives, expert labels)" if args.driver == "neural" else "BC (expert drives)"
    print(f"saved {len(actions)} samples / {len(episode_ends)} episodes [{mode}] -> {args.out}")


if __name__ == "__main__":
    main()
