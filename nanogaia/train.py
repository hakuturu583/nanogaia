import argparse
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import cv2

from nanogaia.latent_dataset import LatentDataset
from nanogaia.model import CosmosVideoTokenizer, VideoARTCoreCV8x8x8


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
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    use_wandb = not args.disable_wandb
    wandb_run = None
    wandb = None
    if use_wandb:
        try:
            import wandb
        except ImportError as exc:
            raise ImportError(
                "wandb is required for logging; install with `pip install wandb` or pass --disable-wandb."
            ) from exc
        default_run_name = args.wandb_run_name
        if default_run_name is None:
            from datetime import datetime, timezone

            default_run_name = datetime.now(timezone.utc).isoformat()
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=default_run_name,
            config=vars(args),
        )

    dataloader = make_dataloader(Path(args.lmdb_path), args.batch_size, args.num_workers)
    model = build_model(device, dtype)
    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    tokenizer_for_logging = None
    if use_wandb and args.video_interval > 0:
        tokenizer_for_logging = CosmosVideoTokenizer(
            device=str(device),
            dtype=dtype,
        )

    global_step = 0
    for epoch in range(args.epochs):
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

            if global_step % args.log_interval == 0:
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
                and args.video_interval > 0
                and global_step > 0
                and global_step % args.video_interval == 0
                and tokenizer_for_logging is not None
            ):
                with torch.no_grad():
                    video_path = decode_and_save_video(
                        pred_future.detach(),
                        target.detach(),
                        tokenizer_for_logging,
                        fps=args.video_fps,
                    )
                wandb.log(
                    {
                        "train/sample_video": wandb.Video(
                            video_path, fps=args.video_fps, caption="pred | target"
                        )
                    },
                    step=global_step,
                )
                os.remove(video_path)

            global_step += 1
            if args.max_steps and global_step >= args.max_steps:
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
        "--lmdb-path",
        type=Path,
        default=Path(__file__).resolve().parent / "dataset" / "latent.lmdb",
        help="Path to latent LMDB produced by CommaDataset.export_as_latent_data.",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument(
        "--video-interval",
        type=int,
        default=200,
        help="Global step interval to log mp4 to wandb.",
    )
    parser.add_argument(
        "--video-fps", type=int, default=4, help="FPS for logged videos."
    )
    parser.add_argument(
        "--max-steps", type=int, default=0, help="0 disables early stop."
    )
    parser.add_argument(
        "--device", type=str, default=None, help="Override device e.g. cuda:0 or cpu"
    )
    parser.add_argument("--wandb-project", type=str, default="nanogaia")
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument(
        "--disable-wandb",
        action="store_true",
        help="Disable wandb logging (metrics and videos).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    try:
        train(parse_args())
    except KeyboardInterrupt:
        print("Interrupted by user, exiting.", file=sys.stderr)
