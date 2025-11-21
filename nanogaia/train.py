import argparse
import os
import sys
import tempfile
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import cv2

from nanogaia.comma2k19_dataset import CommaDataset, ToTensor
from nanogaia.model import CosmosVideoARModel, CosmosVideoTokenizer


def compute_action_deltas(
    orientations: torch.Tensor, positions: torch.Tensor
) -> torch.Tensor:
    """
    Compute per-step 2D pose deltas for an entire batch.

    Args:
        orientations: (B, T, 4) quaternions
        positions:    (B, T, 3) ECEF positions

    Returns:
        actions: (B, T, 3) [x, y, yaw] deltas; last step is zero-padded.
    """
    B, T, _ = orientations.shape
    actions = []
    for b in range(B):
        ori_b = orientations[b].cpu().numpy()
        pos_b = positions[b].cpu().numpy()
        deltas = []
        for t in range(T):
            if t + 1 < T:
                delta = CommaDataset._get_diff_2d_pose(
                    ori_b[t], pos_b[t], ori_b[t + 1], pos_b[t + 1]
                )
            else:
                delta = np.zeros(3, dtype=np.float64)
            deltas.append(delta)
        actions.append(torch.from_numpy(np.stack(deltas, axis=0)))
    return torch.stack(actions, dim=0)


def make_dataloader(
    dataset_root: Path, batch_size: int, num_workers: int
) -> DataLoader:
    dataset = CommaDataset(dataset_root, window_size=32, transform=ToTensor())
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
        video_past:    (B, 3, 16, H, W)
        video_future:  (B, 3, 16, H, W)
        actions_past:  (B, 16, 3)
        actions_future:(B, 16, 3)
    """
    frames = batch["image"].float()  # (B, 32, H, W, 3)
    frames = frames.permute(0, 4, 1, 2, 3)  # (B, 3, 32, H, W)
    frames = frames / 127.5 - 1.0  # normalize to [-1, 1]

    orientations = batch["orientations"].float()
    positions = batch["positions"].float()

    actions = compute_action_deltas(orientations, positions).float()  # (B, 32, 3)

    video_past = frames[:, :, :16].to(device=device, dtype=dtype)
    video_future = frames[:, :, 16:32].to(device=device, dtype=dtype)
    actions_past = actions[:, :16].to(device=device, dtype=dtype)
    actions_future = actions[:, 16:32].to(device=device, dtype=dtype)
    return video_past, video_future, actions_past, actions_future


def build_model(device: torch.device, dtype: torch.dtype) -> CosmosVideoARModel:
    tokenizer = CosmosVideoTokenizer(
        device=str(device),
        dtype=dtype,
    )
    model = CosmosVideoARModel(
        tokenizer=tokenizer,
        t_in_latent=16,
        frames_per_latent=8,
        action_dim_raw=3,
        d_model=512,
        num_layers=8,
        num_heads=8,
        dim_feedforward=2048,
        t_future_latent=16,
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


def train(args: argparse.Namespace) -> None:
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    use_wandb = not args.disable_wandb
    wandb_run = None
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

    dataloader = make_dataloader(
        Path(args.dataset_root), args.batch_size, args.num_workers
    )
    model = build_model(device, dtype)
    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    global_step = 0
    for epoch in range(args.epochs):
        for batch in dataloader:
            video_past, video_future, actions_past, actions_future = prepare_batch(
                batch, device, dtype
            )

            optimizer.zero_grad(set_to_none=True)
            pred_future = model(video_past, actions_past, actions_future)

            # Align target length with prediction; extra frames are truncated.
            target = video_future[:, :, : pred_future.shape[2]]
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
            ):
                with torch.no_grad():
                    video_path = save_video(
                        pred_future.detach(), target.detach(), fps=args.video_fps
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
        description="Train Cosmos Video AR model on Comma2k19 windows."
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default=str(Path(__file__).resolve().parent / "dataset"),
        help="Path to dataset root containing Chunk_* directories.",
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
