# Setup

```
sudo apt install ffmpeg
uv sync --extra cu124
```

## Training

The training script consumes 16-frame windows from `dataset/Chunk_*` sequences, uses the first 8 frames (+ pose deltas) as input, and predicts the next 8 frames conditioned on the next 8 pose deltas via cross-attention.

Basic run (logs to Weights & Biases by default):

```
python -m nanogaia.train \
  --dataset-root /absolute/path/to/dataset \
  --batch-size 1 \
  --epochs 1 \
  --wandb-project nanogaia \
  --wandb-run-name exp1
```

Artifacts:
- Metrics: `train/loss`, `train/epoch` every `--log-interval`.
- Videos: side-by-side `pred | target` mp4 logged every `--video-interval` steps.

To disable W&B entirely (no metrics or videos): add `--disable-wandb`.
