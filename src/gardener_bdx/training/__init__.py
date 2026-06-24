"""Training: RL for the locomotion substrate (the fluid BDX walk) and
expert-distillation / RL-finetuning for the VLA world model. The locomotion
trainer exports a numpy policy the runtime loads with zero ML deps; the VLA
trainer produces the checkpoint :class:`NeuralVLA` loads."""
