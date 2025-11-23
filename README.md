# Setup

```
sudo apt install ffmpeg
uv sync --extra cu124
```

## Training

The training script now consumes pre-encoded latent windows from an LMDB (see `nanogaia/train.yaml` for defaults). Specify a YAML config explicitly and start training:

```
python -m nanogaia.train --config nanogaia/train.yaml
```

Artifacts:
- Metrics: `train/loss`, `train/epoch` every `log_interval` (YAML).
- Videos: side-by-side `pred | target` mp4 logged every `video_interval` steps if wandb is enabled.

To disable W&B entirely (no metrics or videos): add `--disable-wandb`.
