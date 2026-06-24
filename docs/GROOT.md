# GR00T N1 as the VLA brain

Run the gardener with NVIDIA's **Isaac GR00T N1** foundation model as the
end-to-end brain, in place of the lightweight `NeuralVLA`. GR00T is a dual-system
VLA (a VLM "System 2" + a flow-matching action "System 1") that emits a ~16-step
action chunk; we register the droid as a **new embodiment** and execute the chunk
through the same locomotion substrate + Guardian + runtime as everything else.

## How it wires in

- `policy/groot_vla.py::GR00TVLA` implements `VLAPolicy` over `Gr00tPolicy`:
  it builds GR00T's observation dict from our `Observation` + language goal,
  calls `get_action`, buffers the action chunk (re-querying every
  `action_horizon_exec` steps), and maps the chunk onto our whole-body command
  `[vx, vy, wz, body_height, look_yaw, look_pitch, dispense]`.
- `configs/groot/modality.json` declares the new embodiment's state/action/video
  fields (state = `joint_pos|joint_vel|imu|payload` = 32; action =
  `base|head|water` = 7; video = `head_camera`). It travels with the dataset and
  is used at both fine-tune and inference time.
- `build_vla("groot", checkpoint=..., device=...)` returns the adapter;
  `--vla groot --vla-ckpt <path>` selects it in `sim_main` / `robot_main`.

## Pipeline

```bash
# 1) generate demonstrations in the photoreal twin (real pixels matter for GR00T)
python -m gardener_bdx.training.collect_demos --backend mujoco --render \
    --episodes 300 --out data/expert_demos.npz
#    (or --backend isaac once your Isaac stage + RTX camera are set up)

# 2) convert to a LeRobot dataset GR00T fine-tunes on
python -m gardener_bdx.training.export_lerobot --data data/expert_demos.npz \
    --out data/gardener_lerobot

# 3) fine-tune GR00T N1 on the new embodiment (in the Isaac-GR00T repo)
python scripts/gr00t_finetune.py --dataset-path /path/to/data/gardener_lerobot \
    --embodiment-tag new_embodiment --data-config <your_data_config> \
    --output-dir checkpoints/gardener_gr00t

# 4) drive the gardener with GR00T
python -m gardener_bdx.runtime.sim_main --backend mujoco --vla groot \
    --vla-ckpt checkpoints/gardener_gr00t --render
```

The `export_lerobot` step is verified end-to-end here (it produces the
`meta/` + `data/chunk-000/*.parquet` + `modality.json` layout GR00T reads). Video
encoding needs `imageio[ffmpeg]`; without it, frames are dropped to `.npy` so no
data is lost.

## What you need to do

1. **GPU + install.** GR00T inference/fine-tuning needs a capable NVIDIA GPU
   (data-center class for full fine-tuning; an RTX 4090/5090 works for LoRA /
   smaller runs). Install from https://github.com/NVIDIA/Isaac-GR00T .
2. **Base checkpoint.** Pull `nvidia/GR00T-N1.5-3B` (or the current N1.x) from
   Hugging Face as the fine-tune starting point.
3. **Define the fine-tune DataConfig.** `GR00TVLA` ships
   `GardenerGR00TDataConfig` mirroring the repo's `getting_started` configs; align
   its transforms/normalization stats with your dataset, or register a matching
   DataConfig in the Isaac-GR00T repo and pass its name in step 3.
4. **Photoreal demos.** For best transfer, collect step-1 demos in Isaac with the
   RTX head camera (so GR00T sees realistic greenhouse pixels), not the kinematic
   placeholder frames.
5. **(Deploy phase, later) optimize for Jetson.** Quantize/distill the System-2
   VLM (TensorRT-LLM) for on-robot rates; System 1 + the locomotion policy stay
   at full rate. The architecture already assumes a slow brain, so this is a
   budgeting step, not a redesign.

## Why both GR00T and the bundled NeuralVLA?

`NeuralVLA` (torch, no gr00t) is the dependency-light stand-in that makes the
whole pipeline runnable and testable today and is great for fast iteration. GR00T
is the production brain with web-scale pretraining. They share the exact same
`VLAPolicy` seam, so you can develop against `NeuralVLA` and swap to GR00T with a
flag once the heavy stack is installed.
