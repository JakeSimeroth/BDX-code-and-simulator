"""Convert collected expert demos into a LeRobot-format dataset that NVIDIA
Isaac-GR00T can fine-tune on (``gr00t finetune --dataset-path ...``).

    python -m gardener_bdx.training.collect_demos --backend mujoco --render \
        --episodes 200 --out data/expert_demos.npz
    python -m gardener_bdx.training.export_lerobot --data data/expert_demos.npz \
        --out data/gardener_lerobot

Produces the LeRobot v2 layout GR00T expects:

    <out>/meta/info.json          schema, fps, counts
    <out>/meta/tasks.jsonl        unique language instructions
    <out>/meta/episodes.jsonl     per-episode index/length/task
    <out>/meta/modality.json      copy of configs/groot/modality.json
    <out>/data/chunk-000/episode_000000.parquet   state+action+indices
    <out>/videos/chunk-000/observation.images.head_camera/episode_000000.mp4

State is laid out exactly as configs/groot/modality.json:
[joint_pos(12), joint_vel(12), imu=ang_vel+proj_grav(6), payload=soc+water(2)] = 32;
action is [base(3), head(3), water(1)] = 7."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from ..common.config import config_dir


def _state_from_proprio(p: np.ndarray) -> np.ndarray:
    """proprio = [pos12, vel12, g3, angvel3, payload2]  ->  modality state order."""
    joint_pos, joint_vel = p[:, 0:12], p[:, 12:24]
    g, ang_vel, payload = p[:, 24:27], p[:, 27:30], p[:, 30:32]
    imu = np.concatenate([ang_vel, g], axis=1)  # modality.json: imu = [ang_vel, proj_grav]
    return np.concatenate([joint_pos, joint_vel, imu, payload], axis=1).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/expert_demos.npz")
    ap.add_argument("--out", default="data/gardener_lerobot")
    ap.add_argument("--fps", type=int, default=50)
    ap.add_argument("--no-video", action="store_true", help="skip mp4 encoding (state/action only)")
    args = ap.parse_args()

    try:
        import pandas as pd  # noqa: WPS433
    except ImportError as e:
        raise SystemExit(f"export_lerobot needs pandas + pyarrow: pip install pandas pyarrow ({e})")

    d = np.load(args.data, allow_pickle=True)
    state = _state_from_proprio(d["proprio"])
    action = d["actions"].astype(np.float32)
    images = d["images"]
    instructions = d["instructions"].astype(str)
    ends = d["episode_ends"].tolist()
    starts = [0] + ends[:-1]

    out = Path(args.out)
    (out / "meta").mkdir(parents=True, exist_ok=True)
    (out / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
    vid_dir = out / "videos" / "chunk-000" / "observation.images.head_camera"
    vid_dir.mkdir(parents=True, exist_ok=True)

    # tasks.jsonl (unique instructions -> task_index)
    tasks = sorted(set(instructions.tolist()))
    task_index = {t: i for i, t in enumerate(tasks)}
    with open(out / "meta" / "tasks.jsonl", "w") as f:
        for t, i in task_index.items():
            f.write(json.dumps({"task_index": i, "task": t}) + "\n")

    episodes_meta = []
    global_index = 0
    for ep, (s, e) in enumerate(zip(starts, ends)):
        n = e - s
        ep_task = instructions[s]
        df = pd.DataFrame({
            "observation.state": list(state[s:e]),
            "action": list(action[s:e]),
            "timestamp": (np.arange(n) / args.fps).astype(np.float32),
            "frame_index": np.arange(n, dtype=np.int64),
            "episode_index": np.full(n, ep, np.int64),
            "index": np.arange(global_index, global_index + n, dtype=np.int64),
            "task_index": np.full(n, task_index[ep_task], np.int64),
            "annotation.human.task_description": [ep_task] * n,
        })
        df.to_parquet(out / "data" / "chunk-000" / f"episode_{ep:06d}.parquet")
        if not args.no_video:
            _write_video(vid_dir / f"episode_{ep:06d}.mp4", images[s:e], args.fps)
        episodes_meta.append({"episode_index": ep, "tasks": [ep_task], "length": int(n)})
        global_index += n

    with open(out / "meta" / "episodes.jsonl", "w") as f:
        for em in episodes_meta:
            f.write(json.dumps(em) + "\n")

    # info.json — minimal LeRobot v2 schema GR00T reads.
    h, w = int(images.shape[1]), int(images.shape[2])
    info = {
        "codebase_version": "v2.0",
        "robot_type": "gardener_bdx",
        "fps": args.fps,
        "total_episodes": len(ends),
        "total_frames": int(ends[-1]),
        "total_tasks": len(tasks),
        "chunks_size": 1000,
        "features": {
            "observation.state": {"dtype": "float32", "shape": [state.shape[1]]},
            "action": {"dtype": "float32", "shape": [action.shape[1]]},
            "observation.images.head_camera": {"dtype": "video", "shape": [h, w, 3]},
            "annotation.human.task_description": {"dtype": "string", "shape": [1]},
        },
    }
    with open(out / "meta" / "info.json", "w") as f:
        json.dump(info, f, indent=2)

    # modality.json — GR00T's split map travels with the dataset.
    shutil.copy(config_dir() / "groot" / "modality.json", out / "meta" / "modality.json")

    print(f"wrote LeRobot dataset -> {out}  ({len(ends)} episodes, {ends[-1]} frames)")
    print("next: gr00t finetune --dataset-path", out, "--embodiment-tag new_embodiment")


def _write_video(path: Path, frames: np.ndarray, fps: int) -> None:
    try:
        import imageio.v2 as imageio
    except ImportError:
        # Fall back to a frames/ dir so the data isn't lost; encode later.
        fdir = path.with_suffix("")
        fdir.mkdir(exist_ok=True)
        for i, fr in enumerate(frames):
            np.save(fdir / f"{i:05d}.npy", fr)
        return
    with imageio.get_writer(path, fps=fps, macro_block_size=1) as w:
        for fr in frames:
            w.append_data(fr)


if __name__ == "__main__":
    main()
