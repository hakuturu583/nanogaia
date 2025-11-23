from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import cv2
import numpy as np
import torch

from nanogaia.model import CosmosVideoTokenizer


def load_frames(raw_dir: Path, start: int, end: int) -> np.ndarray:
    """
    Load frames [start, end] inclusive from raw_images directory.
    Returns RGB array shaped (T, H, W, 3) float32.
    """
    frames: List[np.ndarray] = []
    for idx in range(start, end + 1):
        frame_path = raw_dir / f"frame_{idx:05d}.jpg"
        frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if frame is None:
            raise FileNotFoundError(f"Missing frame: {frame_path}")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame.astype(np.float32))
    return np.stack(frames, axis=0)


def main() -> None:
    default_raw = (
        Path(__file__).resolve().parent
        / "dataset"
        / "Chunk_1"
        / "b0c9d2329ad1606b|2018-07-27--06-03-57"
        / "3"
        / "raw_images"
    )
    parser = argparse.ArgumentParser(
        description="Encode/decode a small clip with CosmosVideoTokenizer"
    )
    parser.add_argument(
        "--raw-images",
        type=Path,
        default=default_raw,
        help=f"Path to raw_images directory (default: {default_raw})",
    )
    parser.add_argument("--start", type=int, default=0, help="First frame index")
    parser.add_argument("--end", type=int, default=7, help="Last frame index (inclusive)")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("encoder_test.mp4"),
        help="Output mp4 path",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for tokenizer (default: cuda if available)",
    )
    args = parser.parse_args()

    frames = load_frames(args.raw_images, args.start, args.end)
    print(f"Loaded frames: {frames.shape}")

    # Normalize to [-1, 1] and shape to (B, C, T, H, W)
    video = torch.from_numpy(frames).permute(3, 0, 1, 2)  # (T, C, H, W)
    video = (video / 127.5) - 1.0
    video = video.unsqueeze(0)  # (1, C, T, H, W)

    # dtype = torch.bfloat16 if args.device != "cpu" else torch.float32
    tokenizer = CosmosVideoTokenizer(device=args.device, dtype=torch.bfloat16)

    with torch.no_grad():
        latents = tokenizer.encode(video)
        print(f"Latents shape: {latents.shape}")
        decoded = tokenizer.decode(latents)
        tokenizer.decode_as_video(latents[0], str(args.output), fps=4)

    # print(f"Wrote decoded video to: {out_path}")


if __name__ == "__main__":
    main()
