"""Configuration loading. Configs are plain YAML so they are equally readable by
the sim, the trainer, and the on-robot runtime. Typed dataclasses give us
autocomplete and validation without a heavy schema dependency."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def repo_root() -> Path:
    """Locate the repo root from this file (……/src/gardener_bdx/common/config.py)."""
    return Path(__file__).resolve().parents[3]


def config_dir() -> Path:
    return Path(os.environ.get("GARDENER_BDX_CONFIGS", repo_root() / "configs"))


def load_yaml(path: str | os.PathLike) -> dict[str, Any]:
    p = Path(path)
    if not p.is_absolute():
        p = config_dir() / p
    with open(p, "r") as f:
        return yaml.safe_load(f) or {}


@dataclass
class RobotConfig:
    """Kinematic/actuation description shared by every backend. Joint ordering
    here is canonical: perception, the policy, the sim model, and the hardware
    driver all index joints in this order."""

    name: str
    joint_names: tuple[str, ...]
    joint_lower: np.ndarray  # (n,) rad
    joint_upper: np.ndarray  # (n,) rad
    joint_velocity_limit: np.ndarray  # (n,) rad/s
    joint_torque_limit: np.ndarray  # (n,) N·m
    default_joint_positions: np.ndarray  # (n,) rad — nominal stance
    kp: np.ndarray  # (n,) PD position gain
    kd: np.ndarray  # (n,) PD damping gain
    action_scale: float  # policy output -> rad delta scale
    base_height_nominal: float  # m
    foot_link_names: tuple[str, ...]
    mass_kg: float
    water_capacity_liters: float

    @property
    def n_joints(self) -> int:
        return len(self.joint_names)

    @staticmethod
    def from_yaml(path: str | os.PathLike = "robot/gardener_bdx.yaml") -> "RobotConfig":
        d = load_yaml(path)
        joints = d["joints"]
        names = tuple(joints["names"])

        def arr(key: str, default: float | None = None) -> np.ndarray:
            if key in joints:
                return np.asarray(joints[key], dtype=float)
            if default is not None:
                return np.full(len(names), default, dtype=float)
            raise KeyError(f"joints.{key} missing in {path}")

        return RobotConfig(
            name=d["name"],
            joint_names=names,
            joint_lower=arr("lower"),
            joint_upper=arr("upper"),
            joint_velocity_limit=arr("velocity_limit", 20.0),
            joint_torque_limit=arr("torque_limit", 6.0),
            default_joint_positions=arr("default"),
            kp=arr("kp", 20.0),
            kd=arr("kd", 0.5),
            action_scale=float(d.get("action_scale", 0.5)),
            base_height_nominal=float(d.get("base_height_nominal", 0.35)),
            foot_link_names=tuple(d.get("foot_link_names", ["left_foot", "right_foot"])),
            mass_kg=float(d.get("mass_kg", 4.0)),
            water_capacity_liters=float(d.get("water_capacity_liters", 1.5)),
        )


@dataclass
class HierarchyConfig:
    """Control-rate schedule for the three-tier hierarchy + safety.

    The locomotion substrate runs fastest; the VLA brain reasons slowly. In sim
    we are free to run the VLA as fast as we like — the rate decoupling is what
    keeps motion fluid regardless of brain latency, and it's what makes the
    eventual hardware port a matter of *budget*, not *architecture*."""

    locomotion_hz: float = 50.0
    skill_hz: float = 20.0
    vla_hz: float = 5.0
    safety_hz: float = 50.0  # safety runs at the motor rate

    locomotion_policy_path: str = "models/policies/locomotion.npz"
    vla_checkpoint: str = ""  # GR00T / custom VLA weights; empty => scripted mock
    device: str = "cuda"

    @staticmethod
    def from_yaml(path: str | os.PathLike = "control/hierarchy.yaml") -> "HierarchyConfig":
        d = load_yaml(path)
        rates = d.get("rates", {})
        paths = d.get("paths", {})
        return HierarchyConfig(
            locomotion_hz=float(rates.get("locomotion_hz", 50.0)),
            skill_hz=float(rates.get("skill_hz", 20.0)),
            vla_hz=float(rates.get("vla_hz", 5.0)),
            safety_hz=float(rates.get("safety_hz", 50.0)),
            locomotion_policy_path=paths.get("locomotion_policy", "models/policies/locomotion.npz"),
            vla_checkpoint=paths.get("vla_checkpoint", ""),
            device=d.get("device", "cuda"),
        )
