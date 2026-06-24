"""GR00T N1 backbone for the VLA brain.

This adapts NVIDIA Isaac GR00T's ``Gr00tPolicy`` to our :class:`VLAPolicy`
interface, so the full foundation model can drive the gardener with the *same*
locomotion substrate, Guardian, and runtime as the lightweight ``NeuralVLA``.

GR00T is a dual-system VLA (a VLM "System 2" + a flow-matching action "System
1") that emits an **action chunk** (≈16 steps) per query. We register our droid
as a **new embodiment** via ``configs/groot/modality.json`` (which names the
state/action/video fields), run inference with ``embodiment_tag='new_embodiment'``,
and execute the chunk with re-query — the standard GR00T deployment pattern.

Pipeline to make this real (see docs/GROOT.md):
  1. Collect demos + export a LeRobot-format dataset (``training/export_lerobot.py``).
  2. Fine-tune GR00T N1 on it (``gr00t finetune`` with our modality.json).
  3. Point ``--vla groot --vla-ckpt <path>`` at the result.

``gr00t`` is heavy and GPU-only; it is imported lazily so the rest of the package
stays usable without it."""

from __future__ import annotations

import json
from typing import Optional

import numpy as np

from ..common.config import config_dir
from ..common.math_utils import projected_gravity
from ..common.types import Intent, LocomotionCommand, Observation, Pose, Skill
from ..perception.world_model import WorldBelief
from .task import TaskGoal
from .vla_brain import ACTION_KEYS, VLAPolicy


class GR00TVLA(VLAPolicy):
    """VLAPolicy backed by a (fine-tuned) GR00T N1 checkpoint."""

    def __init__(
        self,
        checkpoint: str,
        device: str = "cuda",
        modality_json: str = "groot/modality.json",
        action_horizon_exec: int = 8,
        denoising_steps: int = 4,
    ):
        if not checkpoint:
            raise ValueError("GR00TVLA requires --vla-ckpt pointing at a fine-tuned GR00T checkpoint.")
        self.device = device
        self.exec_horizon = int(action_horizon_exec)
        self.modality = self._load_modality(modality_json)
        self._policy = self._build_policy(checkpoint, denoising_steps)
        self.reset()

    # -- VLAPolicy -------------------------------------------------------- #
    def reset(self) -> None:
        self._chunk: Optional[dict] = None
        self._chunk_i = 0

    def act(self, obs: Observation, goal: TaskGoal, world: Optional[WorldBelief] = None) -> Intent:
        # Action chunking: query GR00T, then execute several steps before re-query.
        if self._chunk is None or self._chunk_i >= self.exec_horizon:
            self._chunk = self._policy.get_action(self._build_observation(obs, goal.instruction))
            self._chunk_i = 0
        a = self._chunk_step(self._chunk_i)
        self._chunk_i += 1

        cmd = LocomotionCommand(
            vx=float(a[0]), vy=float(a[1]), wz=float(a[2]),
            body_height=float(a[3]), look_yaw=float(a[4]), look_pitch=float(a[5]),
        )
        dispense = max(0.0, float(a[6]))
        skill = self._infer_skill(cmd, dispense)
        target = None
        if world is not None and skill in (Skill.NAVIGATE, Skill.APPROACH_PLANT, Skill.DISPENSE_WATER):
            p = world.nearest_dry_plant(goal.dryness_threshold)
            target = Pose(p.position, np.array([1.0, 0, 0, 0])) if p is not None else None
        return Intent(skill, cmd, dispense_rate_lps=dispense, target_pose=target,
                      rationale="GR00T N1 [new_embodiment]")

    # -- GR00T glue ------------------------------------------------------- #
    def _build_observation(self, obs: Observation, instruction: str) -> dict:
        """Pack our Observation into GR00T's LeRobot-style observation dict. Keys
        and array splits must match ``configs/groot/modality.json``. Video/state
        carry a leading time dimension of 1 (single-frame history)."""
        rgb = obs.camera.rgb if obs.camera is not None else np.zeros((48, 64, 3), np.uint8)
        g = projected_gravity(obs.imu.orientation)
        return {
            "video.head_camera": rgb[None].astype(np.uint8),                       # (1,H,W,3)
            "state.joint_pos": obs.joints.positions[None].astype(np.float32),       # (1,12)
            "state.joint_vel": obs.joints.velocities[None].astype(np.float32),      # (1,12)
            "state.imu": np.concatenate([obs.imu.angular_velocity, g])[None].astype(np.float32),  # (1,6)
            "state.payload": np.array([[obs.battery.state_of_charge, obs.water.fraction]], np.float32),  # (1,2)
            "annotation.human.task_description": [instruction],
        }

    def _chunk_step(self, i: int) -> np.ndarray:
        """Reassemble the i-th action of the chunk into our ACTION_KEYS order
        [vx, vy, wz, body_height, look_yaw, look_pitch, dispense]."""
        c = self._chunk
        base = np.asarray(c["action.base"])[i]      # (3,)
        head = np.asarray(c["action.head"])[i]      # (3,)
        water = np.asarray(c["action.water"])[i]    # (1,)
        a = np.concatenate([base, head, np.atleast_1d(water)])
        assert a.shape[0] == len(ACTION_KEYS), f"GR00T action dim {a.shape[0]} != {len(ACTION_KEYS)}"
        return a

    @staticmethod
    def _infer_skill(cmd: LocomotionCommand, dispense: float) -> Skill:
        if dispense > 1e-3:
            return Skill.DISPENSE_WATER
        if abs(cmd.vx) + abs(cmd.vy) + abs(cmd.wz) > 0.05:
            return Skill.NAVIGATE
        return Skill.IDLE

    def _load_modality(self, modality_json: str) -> dict:
        path = config_dir() / modality_json
        with open(path) as f:
            return json.load(f)

    def _build_policy(self, checkpoint: str, denoising_steps: int):
        try:
            from gr00t.experiment.data_config import DATA_CONFIG_MAP  # noqa: F401
            from gr00t.model.policy import Gr00tPolicy
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "GR00T backbone requires NVIDIA Isaac-GR00T. Install it from "
                "https://github.com/NVIDIA/Isaac-GR00T and run on a GPU. See docs/GROOT.md.\n"
                f"(import error: {e})"
            )
        data_config = GardenerGR00TDataConfig(self.modality)
        return Gr00tPolicy(
            model_path=checkpoint,
            modality_config=data_config.modality_config(),
            modality_transform=data_config.transform(),
            embodiment_tag="new_embodiment",
            denoising_steps=denoising_steps,
            device=self.device,
        )


class GardenerGR00TDataConfig:
    """The new-embodiment DataConfig for the gardener, mirroring the structure of
    the configs in NVIDIA's Isaac-GR00T ``getting_started`` examples. It declares
    which observation/action keys exist and how to transform them; the concrete
    field dimensions live in ``configs/groot/modality.json``."""

    def __init__(self, modality: dict):
        self.modality = modality
        self.video_keys = [f"video.{k}" for k in modality["video"]]
        self.state_keys = [f"state.{k}" for k in modality["state"]]
        self.action_keys = [f"action.{k}" for k in modality["action"]]
        self.language_keys = ["annotation.human.task_description"]
        self.observation_indices = [0]
        self.action_indices = list(range(modality.get("action_horizon", 16)))

    def modality_config(self) -> dict:
        from gr00t.data.dataset import ModalityConfig

        return {
            "video": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.video_keys),
            "state": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.state_keys),
            "action": ModalityConfig(delta_indices=self.action_indices, modality_keys=self.action_keys),
            "language": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.language_keys),
        }

    def transform(self):
        # Compose GR00T's standard transforms (video tensor + normalize, state/
        # action min-max from modality.json statistics, language tokenize). Kept
        # explicit so it is easy to align with your fine-tuning DataConfig.
        from gr00t.data.transform.base import ComposedModalityTransform
        from gr00t.data.transform.concat import ConcatTransform
        from gr00t.data.transform.state_action import StateActionToTensor, StateActionTransform
        from gr00t.data.transform.video import VideoToTensor, VideoCrop, VideoColorJitter, VideoToNumpy
        from gr00t.model.transforms import GR00TTransform

        return ComposedModalityTransform(transforms=[
            VideoToTensor(apply_to=self.video_keys),
            VideoCrop(apply_to=self.video_keys, scale=0.95),
            VideoToNumpy(apply_to=self.video_keys),
            StateActionToTensor(apply_to=self.state_keys),
            StateActionToTensor(apply_to=self.action_keys),
            ConcatTransform(video_concat_order=self.video_keys,
                            state_concat_order=self.state_keys,
                            action_concat_order=self.action_keys),
            GR00TTransform(state_horizon=1, action_horizon=len(self.action_indices),
                           max_state_dim=64, max_action_dim=32),
        ])
