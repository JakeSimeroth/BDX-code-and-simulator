"""The neural guts of :class:`~gardener_bdx.policy.vla_brain.NeuralVLA`.

A compact but faithful **dual-system VLA world model**, mirroring the structure
NVIDIA describes for Isaac GR00T N1:

  * **System 2 (slow, ~5-10 Hz):** encode RGB-D + LiDAR-BEV + proprio + the
    language instruction, fuse them with a cross-modal transformer, and emit a
    reasoning *latent* plus a discrete *skill*. This is the deliberative step.
  * **System 1 (fast, every tick):** a **flow-matching** action head. Starting
    from noise it integrates a learned conditional velocity field, conditioned
    on the System-2 latent and the freshest proprioception, into a continuous
    whole-body action. This is the reflexive step that keeps motion fluid even
    though System 2 thinks slowly.

This module imports PyTorch at import time and is therefore loaded *lazily*
(only when ``NeuralVLA`` is constructed). The runs-anywhere scripted path never
touches it.

GR00T integration: set ``backbone="groot"`` and point ``load()`` at a GR00T N1
checkpoint. In that mode System 2 is replaced by the pretrained GR00T VLM and
System 1 by its diffusion/flow action expert; this file's modules become the
embodiment-specific encoders/decoders (the "adapter") around them. See
docs/SIM2REAL.md.
"""

from __future__ import annotations

import zlib
from typing import Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    _TORCH_OK = True
except Exception as _e:  # pragma: no cover - exercised only without torch
    _TORCH_OK = False
    _IMPORT_ERR = _e


VOCAB = 4096  # hashed-token vocabulary for the lightweight text encoder
RGB_HW = 96  # square resolution images are resized to
BEV_HW = 64  # LiDAR bird's-eye-view raster size
BEV_RANGE = 6.0  # metres mapped across the BEV grid
LATENT_DIM = 256


def _require_torch():
    if not _TORCH_OK:  # pragma: no cover
        raise ImportError(
            "NeuralVLA requires PyTorch. Install the ML extra:\n"
            "    pip install -e '.[vla]'\n"
            f"(original import error: {_IMPORT_ERR})"
        )


# --------------------------------------------------------------------------- #
# Encoders
# --------------------------------------------------------------------------- #

if _TORCH_OK:

    class _ConvStack(nn.Module):
        def __init__(self, in_ch: int, out_dim: int = 128):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(in_ch, 32, 5, stride=2, padding=2), nn.GroupNorm(8, 32), nn.SiLU(),
                nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.GroupNorm(8, 64), nn.SiLU(),
                nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.GroupNorm(8, 128), nn.SiLU(),
                nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            )
            self.proj = nn.Linear(128, out_dim)

        def forward(self, x):
            return self.proj(self.net(x))

    class _TextEncoder(nn.Module):
        """Hashed bag-of-tokens + GRU. A stand-in for a real tokenizer/VLM text
        tower; replaced by GR00T's language tower in ``backbone="groot"``."""

        def __init__(self, out_dim: int = 128):
            super().__init__()
            self.emb = nn.Embedding(VOCAB, 128)
            self.gru = nn.GRU(128, out_dim, batch_first=True)

        def forward(self, token_ids):  # (B, T)
            e = self.emb(token_ids)
            _, h = self.gru(e)
            return h[-1]

    class _FlowActionHead(nn.Module):
        """Conditional flow-matching head. ``velocity(x, t, cond)`` defines an ODE
        whose integration from x0~N(0,I) at t=0 to t=1 yields the action."""

        def __init__(self, action_dim: int, cond_dim: int, hidden: int = 256):
            super().__init__()
            self.action_dim = action_dim
            self.net = nn.Sequential(
                nn.Linear(action_dim + 1 + cond_dim, hidden), nn.SiLU(),
                nn.Linear(hidden, hidden), nn.SiLU(),
                nn.Linear(hidden, action_dim),
            )

        def velocity(self, x, t, cond):
            return self.net(torch.cat([x, t, cond], dim=-1))

        @torch.no_grad()
        def sample(self, cond, steps: int = 4):
            b = cond.shape[0]
            x = torch.randn(b, self.action_dim, device=cond.device)
            dt = 1.0 / steps
            for k in range(steps):
                t = torch.full((b, 1), k * dt, device=cond.device)
                x = x + self.velocity(x, t, cond) * dt
            return x

    class WorldModelVLANet(nn.Module):
        def __init__(self, action_dim: int):
            super().__init__()
            self.action_dim = action_dim
            self.rgb_enc = _ConvStack(in_ch=4, out_dim=128)  # RGB + depth
            self.bev_enc = _ConvStack(in_ch=1, out_dim=128)  # LiDAR BEV
            self.proprio_enc = nn.Sequential(nn.LazyLinear(128), nn.SiLU(), nn.Linear(128, 128))
            self.text_enc = _TextEncoder(out_dim=128)

            # System 2: fuse 4 modality tokens with a small transformer encoder.
            self.tok = nn.Linear(128, LATENT_DIM)
            enc_layer = nn.TransformerEncoderLayer(
                d_model=LATENT_DIM, nhead=4, dim_feedforward=512, batch_first=True
            )
            self.fusion = nn.TransformerEncoder(enc_layer, num_layers=2)
            self.skill_head = nn.Linear(LATENT_DIM, 6)  # len(Skill)

            # System 1: flow head conditioned on [latent, proprio].
            self.action_head = _FlowActionHead(action_dim, cond_dim=LATENT_DIM + 128)

            self._backbone = "world_model"

        # -- shared encoder (batched, differentiable) ---------------------- #
        def encode(self, img, bev, proprio, tokens):
            """System 2 forward over a batch of tensors. Returns (latent,
            skill_logits, proprio_embed). Used by both inference and training."""
            proprio_embed = self.proprio_enc(proprio)
            toks = torch.stack(
                [
                    self.tok(self.rgb_enc(img)),
                    self.tok(self.bev_enc(bev)),
                    self.tok(proprio_embed),
                    self.tok(self.text_enc(tokens)),
                ],
                dim=1,
            )  # (B, 4, D)
            latent = self.fusion(toks).mean(dim=1)  # (B, D)
            return latent, self.skill_head(latent), proprio_embed

        def flow_matching_loss(self, cond, target_actions):
            """Conditional flow-matching (rectified-flow) loss for System 1."""
            x1 = target_actions
            x0 = torch.randn_like(x1)
            t = torch.rand(x1.shape[0], 1, device=x1.device)
            xt = (1 - t) * x0 + t * x1
            v_pred = self.action_head.velocity(xt, t, cond)
            return ((v_pred - (x1 - x0)) ** 2).mean()

        # -- inference API used by NeuralVLA ------------------------------- #
        def reset(self):
            pass

        def reason(self, obs, instruction: str) -> Tuple[np.ndarray, int]:
            img, bev, proprio, tokens = _obs_to_tensors(obs, instruction, self._device())
            with torch.no_grad():
                latent, skill_logits, _ = self.encode(img, bev, proprio, tokens)
                skill = int(skill_logits.argmax(dim=-1).item())
            return latent.cpu().numpy()[0], skill

        def act(self, obs, latent_np: np.ndarray) -> np.ndarray:
            _, _, proprio, _ = _obs_to_tensors(obs, "", self._device())
            with torch.no_grad():
                latent = torch.as_tensor(latent_np, device=self._device()).unsqueeze(0)
                cond = torch.cat([latent, self.proprio_enc(proprio)], dim=-1)
                a = self.action_head.sample(cond, steps=4)
            return a.cpu().numpy()[0]

        # -- loading ------------------------------------------------------- #
        def load(self, checkpoint: str, device: str = "cuda", backbone: str = "world_model"):
            self._backbone = backbone
            dev = device if (device == "cpu" or torch.cuda.is_available()) else "cpu"
            if backbone == "groot":
                # Documented integration seam. The real call wraps NVIDIA's
                # released GR00T N1 policy and adapts its action space to ours.
                raise NotImplementedError(
                    "GR00T backbone: load via the Isaac GR00T package, e.g.\n"
                    "    from gr00t.model.policy import Gr00tPolicy\n"
                    "    self.groot = Gr00tPolicy.from_pretrained(checkpoint)\n"
                    "and map its action chunk onto ACTION_KEYS. See docs/SIM2REAL.md."
                )
            self.to(dev)
            # Materialize Lazy* params with a dummy pass before loading weights.
            self._warmup(dev)
            if checkpoint:
                state = torch.load(checkpoint, map_location=dev)
                self.load_state_dict(state.get("model", state), strict=False)
            self.eval()
            return self

        def _warmup(self, dev):
            dummy = _zeros_obs()
            self.reason(dummy, "warmup")

        def _device(self):
            return next(self.parameters()).device


def _obs_to_tensors(obs, instruction: str, device):
    """Pack a raw :class:`Observation` into network tensors. This is the only
    place that knows the on-the-wire layout; encoders stay generic."""
    _require_torch()

    # --- RGB-D ---
    if obs.camera is not None and obs.camera.rgb is not None:
        rgb = torch.as_tensor(obs.camera.rgb, dtype=torch.float32).permute(2, 0, 1) / 255.0
        if obs.camera.depth is not None:
            d = torch.as_tensor(obs.camera.depth, dtype=torch.float32).unsqueeze(0)
            d = torch.nan_to_num(d / 5.0, nan=0.0)
        else:
            d = torch.zeros(1, rgb.shape[1], rgb.shape[2])
        img = torch.cat([rgb, d], dim=0).unsqueeze(0)
    else:
        img = torch.zeros(1, 4, RGB_HW, RGB_HW)
    img = F.interpolate(img, size=(RGB_HW, RGB_HW), mode="bilinear", align_corners=False)

    # --- LiDAR BEV ---
    bev = torch.zeros(1, 1, BEV_HW, BEV_HW)
    if obs.lidar is not None and obs.lidar.points.shape[0] > 0:
        p = obs.lidar.points
        ij = ((p[:, :2] + BEV_RANGE) / (2 * BEV_RANGE) * BEV_HW).astype(int)
        m = (ij[:, 0] >= 0) & (ij[:, 0] < BEV_HW) & (ij[:, 1] >= 0) & (ij[:, 1] < BEV_HW)
        for i, j in ij[m]:
            bev[0, 0, j, i] = 1.0

    # --- Proprioception ---
    from ..common.math_utils import projected_gravity

    g = projected_gravity(obs.imu.orientation)
    proprio = np.concatenate(
        [
            obs.joints.positions,
            obs.joints.velocities,
            g,
            obs.imu.angular_velocity,
            [obs.battery.state_of_charge, obs.water.fraction],
        ]
    ).astype(np.float32)
    proprio_t = torch.as_tensor(proprio).unsqueeze(0)

    # --- Language (hashed tokens) ---
    tokens = _hash_tokens(instruction)
    tok_t = torch.as_tensor(tokens, dtype=torch.long).unsqueeze(0)

    return (img.to(device), bev.to(device), proprio_t.to(device), tok_t.to(device))


def _hash_tokens(text: str, max_len: int = 16):
    # Stable, process-independent hashing (Python's built-in hash() is salted per
    # process via PYTHONHASHSEED, which would desync tokens between data
    # collection, training, and inference). crc32 is deterministic everywhere.
    words = (text or "<pad>").lower().split()[:max_len]
    ids = [(zlib.crc32(w.encode("utf-8")) % (VOCAB - 1)) + 1 for w in words] or [0]
    ids += [0] * (max_len - len(ids))
    return np.asarray(ids[:max_len], dtype=np.int64)


def _zeros_obs():
    """A minimal all-zeros Observation for Lazy-layer materialization."""
    from ..common.types import BatteryState, ImuReading, JointState, Observation, WaterTankState

    n = 12
    return Observation(
        stamp=0.0,
        imu=ImuReading(np.array([1.0, 0, 0, 0]), np.zeros(3), np.array([0, 0, -9.81])),
        joints=JointState(np.zeros(n), np.zeros(n), np.zeros(n)),
        camera=None,
        lidar=None,
        battery=BatteryState(state_of_charge=1.0),
        water=WaterTankState(level_liters=1.0, capacity_liters=1.5),
    )
