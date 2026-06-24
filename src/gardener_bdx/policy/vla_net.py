"""The neural guts of :class:`~gardener_bdx.policy.vla_brain.NeuralVLA`.

A compact but faithful **dual-system VLA world model**, mirroring NVIDIA Isaac
GR00T N1:

  * **System 2 (slow, ~5-10 Hz):** encode a short *history* of RGB-D + LiDAR-BEV
    + proprio frames with a deep **residual** vision encoder, fuse them with the
    language instruction through a **temporal transformer**, and emit a reasoning
    *latent* + a discrete *skill*. The temporal context lets it perceive motion
    (an approaching person, its own drift) — not just a single frame.
  * **System 1 (fast, every tick):** a **flow-matching** action head integrates a
    learned conditional velocity field, conditioned on the System-2 latent and
    the freshest proprioception, into a continuous whole-body action.

Imports PyTorch at import time → loaded lazily (only when ``NeuralVLA`` is
constructed). The runs-anywhere scripted path never touches it.

GR00T integration lives in :mod:`gardener_bdx.policy.groot_vla`; this file is the
dependency-light stand-in that makes the pipeline runnable + testable today."""

from __future__ import annotations

import zlib
from typing import List, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    _TORCH_OK = True
except Exception as _e:  # pragma: no cover - exercised only without torch
    _TORCH_OK = False
    _IMPORT_ERR = _e


VOCAB = 4096        # hashed-token vocabulary for the lightweight text encoder
RGB_HW = 96         # square resolution images are resized to
BEV_HW = 64         # LiDAR bird's-eye-view raster size
BEV_RANGE = 6.0     # metres mapped across the BEV grid
EMB = 256           # per-modality embedding width
LATENT_DIM = 256    # System-2 reasoning latent width
HISTORY_LEN = 4     # number of past frames the VLA reasons over


def _require_torch():
    if not _TORCH_OK:  # pragma: no cover
        raise ImportError(
            "NeuralVLA requires PyTorch. Install the ML extra:\n"
            "    pip install -e '.[vla]'\n"
            f"(original import error: {_IMPORT_ERR})"
        )


if _TORCH_OK:

    class _ResidualBlock(nn.Module):
        def __init__(self, ch: int):
            super().__init__()
            self.c1 = nn.Conv2d(ch, ch, 3, padding=1)
            self.n1 = nn.GroupNorm(8, ch)
            self.c2 = nn.Conv2d(ch, ch, 3, padding=1)
            self.n2 = nn.GroupNorm(8, ch)

        def forward(self, x):
            h = F.silu(self.n1(self.c1(x)))
            h = self.n2(self.c2(h))
            return F.silu(x + h)

    class _ResidualConvEncoder(nn.Module):
        """Deeper residual CNN (vs. the previous 3-conv stack) — more capacity for
        the gardener's fine visual cues (leaf wilt, soil, a person's pose)."""

        def __init__(self, in_ch: int, out_dim: int = EMB):
            super().__init__()
            self.stem = nn.Sequential(nn.Conv2d(in_ch, 32, 5, 2, 2), nn.GroupNorm(8, 32), nn.SiLU())
            self.l1 = nn.Sequential(nn.Conv2d(32, 64, 3, 2, 1), nn.GroupNorm(8, 64), nn.SiLU(), _ResidualBlock(64))
            self.l2 = nn.Sequential(nn.Conv2d(64, 128, 3, 2, 1), nn.GroupNorm(8, 128), nn.SiLU(), _ResidualBlock(128))
            self.l3 = nn.Sequential(nn.Conv2d(128, 256, 3, 2, 1), nn.GroupNorm(8, 256), nn.SiLU(), _ResidualBlock(256))
            self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(256, out_dim))

        def forward(self, x):
            return self.head(self.l3(self.l2(self.l1(self.stem(x)))))

    class _TextEncoder(nn.Module):
        """Hashed bag-of-tokens + GRU. Stand-in for a real tokenizer/VLM text
        tower (replaced by GR00T's language tower in the GR00T backbone)."""

        def __init__(self, out_dim: int = EMB):
            super().__init__()
            self.emb = nn.Embedding(VOCAB, 128)
            self.gru = nn.GRU(128, out_dim, batch_first=True)

        def forward(self, token_ids):  # (B, L)
            _, h = self.gru(self.emb(token_ids))
            return h[-1]  # (B, out_dim)

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
        def __init__(self, action_dim: int, history_len: int = HISTORY_LEN):
            super().__init__()
            self.action_dim = action_dim
            self.history_len = history_len
            self.rgb_enc = _ResidualConvEncoder(in_ch=4, out_dim=EMB)   # RGB + depth
            self.bev_enc = _ResidualConvEncoder(in_ch=1, out_dim=EMB)   # LiDAR BEV
            self.proprio_enc = nn.Sequential(nn.LazyLinear(EMB), nn.SiLU(), nn.Linear(EMB, EMB))
            self.text_enc = _TextEncoder(out_dim=EMB)

            # System 2: a temporal transformer over [language, frame_1..frame_T].
            self.pos = nn.Parameter(torch.zeros(1, history_len + 1, EMB))
            layer = nn.TransformerEncoderLayer(d_model=EMB, nhead=4, dim_feedforward=512, batch_first=True)
            self.temporal = nn.TransformerEncoder(layer, num_layers=2)
            self.skill_head = nn.Linear(LATENT_DIM, 6)  # len(Skill)

            # System 1: flow head conditioned on [latent, current proprio].
            self.action_head = _FlowActionHead(action_dim, cond_dim=LATENT_DIM + EMB)
            self._backbone = "world_model"

        # -- shared encoder (batched, differentiable, temporal) ------------ #
        def encode(self, img, bev, proprio, tokens):
            """img:(B,T,4,H,W) bev:(B,T,1,Hb,Wb) proprio:(B,T,P) tokens:(B,L).
            Returns (latent, skill_logits, proprio_embed_last)."""
            B, T = img.shape[0], img.shape[1]
            rgb = self.rgb_enc(img.flatten(0, 1)).view(B, T, EMB)
            bv = self.bev_enc(bev.flatten(0, 1)).view(B, T, EMB)
            pro = self.proprio_enc(proprio)                       # (B,T,EMB)
            per_step = (rgb + bv + pro) / 3.0                     # fuse modalities per frame
            text = self.text_enc(tokens).unsqueeze(1)             # (B,1,EMB)
            seq = torch.cat([text, per_step], dim=1) + self.pos[:, : T + 1]
            latent = self.temporal(seq)[:, -1]                    # last position summarizes history+goal
            return latent, self.skill_head(latent), pro[:, -1]

        def flow_matching_loss(self, cond, target_actions):
            """Rectified-flow loss for System 1."""
            x1 = target_actions
            x0 = torch.randn_like(x1)
            t = torch.rand(x1.shape[0], 1, device=x1.device)
            xt = (1 - t) * x0 + t * x1
            return ((self.action_head.velocity(xt, t, cond) - (x1 - x0)) ** 2).mean()

        # -- inference API used by NeuralVLA ------------------------------- #
        def reset(self):
            pass

        def reason(self, obs_history: List, instruction: str) -> Tuple[np.ndarray, int]:
            img, bev, proprio, tokens = _history_to_tensors(
                obs_history, instruction, self._device(), self.history_len
            )
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
                raise NotImplementedError(
                    "Use gardener_bdx.policy.groot_vla.GR00TVLA for the GR00T backbone "
                    "(build_vla('groot', ...)); this lightweight net is the stand-in."
                )
            self.to(dev)
            self._warmup(dev)  # materialize Lazy* params before loading weights
            if checkpoint:
                state = torch.load(checkpoint, map_location=dev)
                self.load_state_dict(state.get("model", state), strict=False)
            self.eval()
            return self

        def _warmup(self, dev):
            self.reason([_zeros_obs()], "warmup")

        def _device(self):
            return next(self.parameters()).device


def _obs_to_tensors(obs, instruction: str, device):
    """Pack a single raw :class:`Observation` into per-frame tensors
    (img:(1,4,H,W), bev:(1,1,Hb,Wb), proprio:(1,P), tokens:(1,L))."""
    _require_torch()

    if obs.camera is not None and obs.camera.rgb is not None:
        rgb = torch.as_tensor(obs.camera.rgb, dtype=torch.float32).permute(2, 0, 1) / 255.0
        if obs.camera.depth is not None:
            d = torch.nan_to_num(torch.as_tensor(obs.camera.depth, dtype=torch.float32).unsqueeze(0) / 5.0, nan=0.0)
        else:
            d = torch.zeros(1, rgb.shape[1], rgb.shape[2])
        img = torch.cat([rgb, d], dim=0).unsqueeze(0)
    else:
        img = torch.zeros(1, 4, RGB_HW, RGB_HW)
    img = F.interpolate(img, size=(RGB_HW, RGB_HW), mode="bilinear", align_corners=False)

    bev = torch.zeros(1, 1, BEV_HW, BEV_HW)
    if obs.lidar is not None and obs.lidar.points.shape[0] > 0:
        p = obs.lidar.points
        ij = ((p[:, :2] + BEV_RANGE) / (2 * BEV_RANGE) * BEV_HW).astype(int)
        m = (ij[:, 0] >= 0) & (ij[:, 0] < BEV_HW) & (ij[:, 1] >= 0) & (ij[:, 1] < BEV_HW)
        for i, j in ij[m]:
            bev[0, 0, j, i] = 1.0

    from ..common.math_utils import projected_gravity

    g = projected_gravity(obs.imu.orientation)
    proprio = np.concatenate([
        obs.joints.positions, obs.joints.velocities, g, obs.imu.angular_velocity,
        [obs.battery.state_of_charge, obs.water.fraction],
    ]).astype(np.float32)
    proprio_t = torch.as_tensor(proprio).unsqueeze(0)
    tok_t = torch.as_tensor(_hash_tokens(instruction), dtype=torch.long).unsqueeze(0)
    return img.to(device), bev.to(device), proprio_t.to(device), tok_t.to(device)


def _history_to_tensors(obs_list: List, instruction: str, device, T: int):
    """Stack the most recent ``T`` observations into temporal tensors
    (1,T,...). Pads at the front by repeating the oldest frame."""
    _require_torch()
    frames, bevs, pros = [], [], []
    for o in obs_list[-T:]:
        img, bev, pro, _ = _obs_to_tensors(o, "", device)
        frames.append(img); bevs.append(bev); pros.append(pro)
    while len(frames) < T:  # left-pad with the oldest available frame
        frames.insert(0, frames[0]); bevs.insert(0, bevs[0]); pros.insert(0, pros[0])
    img = torch.stack(frames, dim=1)   # (1,T,4,H,W)
    bev = torch.stack(bevs, dim=1)     # (1,T,1,Hb,Wb)
    pro = torch.stack(pros, dim=1)     # (1,T,P)
    tok = torch.as_tensor(_hash_tokens(instruction), dtype=torch.long).unsqueeze(0).to(device)
    return img, bev, pro, tok


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
