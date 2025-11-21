import argparse
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

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


def prepare_batch(batch: dict, device: torch.device) -> Tuple[torch.Tensor, ...]:
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

    video_past = frames[:, :, :16].to(device)
    video_future = frames[:, :, 16:32].to(device)
    actions_past = actions[:, :16].to(device)
    actions_future = actions[:, 16:32].to(device)
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


def train(args: argparse.Namespace) -> None:
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    dtype = torch.float16 if device.type == "cuda" else torch.float32

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
                batch, device
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

            global_step += 1
            if args.max_steps and global_step >= args.max_steps:
                print("Reached max_steps, stopping training loop.")
                return


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
        "--max-steps", type=int, default=0, help="0 disables early stop."
    )
    parser.add_argument(
        "--device", type=str, default=None, help="Override device e.g. cuda:0 or cpu"
    )
    return parser.parse_args()


if __name__ == "__main__":
    try:
        train(parse_args())
    except KeyboardInterrupt:
        print("Interrupted by user, exiting.", file=sys.stderr)
