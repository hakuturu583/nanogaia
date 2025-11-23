import argparse
import os
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import cv2
import yaml

from nanogaia.latent_dataset import LatentDataset
from nanogaia.model import CosmosVideoTokenizer, VideoARTCoreCV8x8x8


@dataclass
class WandbConfig:
    project: str = "nanogaia"
    run_name: str | None = None
    disable: bool = False

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WandbConfig":
        base = cls()
        return cls(
            project=data.get("project", base.project),
            run_name=data.get("run_name", base.run_name),
            disable=bool(data.get("disable", base.disable)),
        )


@dataclass
class TrainConfig:
    lmdb_path: Path = Path(__file__).resolve().parent / "dataset" / "latent.lmdb"
    batch_size: int = 1
    num_workers: int = 0
    epochs: int = 1
    lr: float = 1e-4
    weight_decay: float = 1e-4
    log_interval: int = 10
    video_interval: int = 200
    video_fps: int = 4
    max_steps: int = 0
    device: str | None = None
    wandb: WandbConfig = field(default_factory=WandbConfig)

    @classmethod
    def from_dict(cls, data: Dict[str, Any], base_path: Path | None = None) -> "TrainConfig":
        base = cls()
        wandb_cfg = WandbConfig.from_dict(data.get("wandb", {}))

        lmdb_path_raw = data.get("lmdb_path", base.lmdb_path)
        lmdb_path = Path(lmdb_path_raw)
        if base_path and not lmdb_path.is_absolute():
            lmdb_path = (base_path / lmdb_path).resolve()

        return cls(
            lmdb_path=lmdb_path,
            batch_size=int(data.get("batch_size", base.batch_size)),
            num_workers=int(data.get("num_workers", base.num_workers)),
            epochs=int(data.get("epochs", base.epochs)),
            lr=float(data.get("lr", base.lr)),
            weight_decay=float(data.get("weight_decay", base.weight_decay)),
            log_interval=int(data.get("log_interval", base.log_interval)),
            video_interval=int(data.get("video_interval", base.video_interval)),
            video_fps=int(data.get("video_fps", base.video_fps)),
            max_steps=int(data.get("max_steps", base.max_steps)),
            device=data.get("device", base.device),
            wandb=wandb_cfg,
        )

    def to_log_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["lmdb_path"] = str(self.lmdb_path)
        return payload


def load_train_config(path: Path) -> TrainConfig:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r") as handle:
        raw = yaml.safe_load(handle) or {}
    return TrainConfig.from_dict(raw, base_path=path.parent)


def apply_cli_overrides(config: TrainConfig, args: argparse.Namespace) -> TrainConfig:
    if args.device:
        config.device = args.device
    if args.wandb_run_name:
        config.wandb.run_name = args.wandb_run_name
    if args.disable_wandb:
        config.wandb.disable = True
    return config


def to_tensor(sample: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
    return {k: torch.from_numpy(v) for k, v in sample.items()}


def make_dataloader(lmdb_path: Path, batch_size: int, num_workers: int) -> DataLoader:
    dataset = LatentDataset(lmdb_path, transform=to_tensor)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )


def prepare_batch(
    batch: dict, device: torch.device, dtype: torch.dtype
) -> Tuple[torch.Tensor, ...]:
    """
    Returns:
        z_past:         (B, T_lat, C_lat, H_lat, W_lat)
        z_future:       (B, T_lat, C_lat, H_lat, W_lat)
        actions_past:   (B, 8, 3)
        actions_future: (B, 8, 3)
    """
    latents_past = batch["latent_past"].to(device=device, dtype=dtype)
    latents_future = batch["latent_future"].to(device=device, dtype=dtype)
    actions_past = batch["actions_past"].to(device=device, dtype=dtype)
    actions_future = batch["actions_future"].to(device=device, dtype=dtype)

    z_past = latents_past.permute(0, 2, 1, 3, 4)  # (B, T, C, H, W)
    z_future = latents_future.permute(0, 2, 1, 3, 4)
    return z_past, z_future, actions_past, actions_future


def build_model(device: torch.device, dtype: torch.dtype) -> VideoARTCoreCV8x8x8:
    model = VideoARTCoreCV8x8x8(
        t_in_latent=1,
        frames_per_latent=8,
        action_dim_raw=3,
        d_model=256,
        num_layers=4,
        num_heads=4,
        dim_feedforward=1024,
        t_future_latent=1,
    ).to(device=device, dtype=dtype)
    return model


def to_uint8_video(frames: torch.Tensor) -> np.ndarray:
    """
    frames: (T, C, H, W) in [-1, 1]
    returns: (T, H, W, 3) uint8 RGB
    """
    frames = torch.clamp(frames, -1.0, 1.0)
    frames = (frames + 1.0) * 127.5
    frames = frames.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
    return frames


def save_video(pred: torch.Tensor, target: torch.Tensor, fps: int = 4) -> str:
    """
    pred/target: (1, C, T, H, W)
    Writes an mp4 stacking pred|target per frame. Returns file path.
    """
    pred = pred[0].permute(1, 0, 2, 3)  # (T, C, H, W)
    target = target[0].permute(1, 0, 2, 3)
    pred_np = to_uint8_video(pred)
    target_np = to_uint8_video(target)

    T, H, W, _ = pred_np.shape
    frames = []
    for t in range(T):
        pred_frame = cv2.cvtColor(pred_np[t], cv2.COLOR_RGB2BGR)
        tgt_frame = cv2.cvtColor(target_np[t], cv2.COLOR_RGB2BGR)
        frames.append(np.concatenate([pred_frame, tgt_frame], axis=1))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fd, path = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    writer = cv2.VideoWriter(
        path, fourcc, fps, (frames[0].shape[1], frames[0].shape[0])
    )
    for frame in frames:
        writer.write(frame)
    writer.release()
    return path


def decode_and_save_video(
    pred_latent: torch.Tensor,
    target_latent: torch.Tensor,
    tokenizer: CosmosVideoTokenizer,
    fps: int,
) -> str:
    """
    Decode latent predictions and targets to video and write to disk.
    """
    with torch.no_grad():
        pred_video = tokenizer.decode(
            pred_latent.to(device=tokenizer.device, dtype=tokenizer.dtype)
        )
        target_video = tokenizer.decode(
            target_latent.to(device=tokenizer.device, dtype=tokenizer.dtype)
        )
    return save_video(pred_video, target_video, fps=fps)


def train(args: argparse.Namespace) -> None:
    config = load_train_config(args.config)
    config = apply_cli_overrides(config, args)

    device = torch.device(
        config.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    use_wandb = not config.wandb.disable
    wandb_run = None
    wandb = None
    if use_wandb:
        try:
            import wandb
        except ImportError as exc:
            raise ImportError(
                "wandb is required for logging; install with `pip install wandb` or pass --disable-wandb."
            ) from exc
        default_run_name = config.wandb.run_name
        if default_run_name is None:
            from datetime import datetime, timezone

            default_run_name = datetime.now(timezone.utc).isoformat()
        wandb_run = wandb.init(
            project=config.wandb.project,
            name=default_run_name,
            config=config.to_log_dict(),
        )

    dataloader = make_dataloader(config.lmdb_path, config.batch_size, config.num_workers)
    model = build_model(device, dtype)
    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )

    tokenizer_for_logging = None
    if use_wandb and config.video_interval > 0:
        tokenizer_for_logging = CosmosVideoTokenizer(
            device=str(device),
            dtype=dtype,
        )

    global_step = 0
    for epoch in range(config.epochs):
        for batch in dataloader:
            z_past, z_future, actions_past, actions_future = prepare_batch(
                batch, device, dtype
            )

            optimizer.zero_grad(set_to_none=True)
            pred_future = model(z_past, actions_past, actions_future)

            # Align target length with prediction; extra frames are truncated.
            target = z_future.permute(0, 2, 1, 3, 4)
            target = target[:, :, : pred_future.shape[2]]
            loss = F.l1_loss(pred_future, target)
            loss.backward()
            optimizer.step()

            if global_step % config.log_interval == 0:
                print(
                    f"epoch {epoch} step {global_step} "
                    f"loss {loss.item():.4f} device {device} dtype {dtype}"
                )
                if use_wandb:
                    wandb.log(
                        {
                            "train/loss": loss.item(),
                            "train/epoch": epoch,
                        },
                        step=global_step,
                    )

            if (
                use_wandb
                and config.video_interval > 0
                and global_step > 0
                and global_step % config.video_interval == 0
                and tokenizer_for_logging is not None
            ):
                with torch.no_grad():
                    video_path = decode_and_save_video(
                        pred_future.detach(),
                        target.detach(),
                        tokenizer_for_logging,
                        fps=config.video_fps,
                    )
                wandb.log(
                    {
                        "train/sample_video": wandb.Video(
                            video_path, fps=config.video_fps, caption="pred | target"
                        )
                    },
                    step=global_step,
                )
                os.remove(video_path)

            global_step += 1
            if config.max_steps and global_step >= config.max_steps:
                print("Reached max_steps, stopping training loop.")
                if wandb_run:
                    wandb_run.finish()
                return

    if wandb_run:
        wandb_run.finish()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train VideoARTCoreCV8x8x8 on latent LMDB windows."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent / "train.yaml",
        help="Path to YAML config defining training hyperparameters.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override device from config (e.g., cuda:0 or cpu).",
    )
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default=None,
        help="Override W&B run name specified in config.",
    )
    parser.add_argument(
        "--disable-wandb",
        action="store_true",
        help="Force disable wandb logging regardless of config.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    try:
        train(parse_args())
    except KeyboardInterrupt:
        print("Interrupted by user, exiting.", file=sys.stderr)
