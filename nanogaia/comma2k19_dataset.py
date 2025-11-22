from __future__ import annotations

import os
import io
from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from nanogaia.model import CosmosVideoTokenizer
from nanogaia.util import (
    ecef_vector_from_pose,
    quaternion_difference,
    quaternion_to_yaw,
)


POSE_FILES = {
    "gps_times": "frame_gps_times",
    "orientations": "frame_orientations",
    "positions": "frame_positions",
    "times": "frame_times",
    "velocities": "frame_velocities",
}


@dataclass
class SequenceMeta:
    path: Path
    raw_images_dir: Path
    pose_arrays: Dict[str, np.memmap]
    num_frames: int
    num_samples: int


class ToTensor(object):

    def __call__(self, sample):
        return {key: torch.from_numpy(value) for key, value in sample.items()}


class CommaDataset(Dataset):

    FRAME_DIGITS = 5

    def __init__(
        self, root_dir: str | os.PathLike, window_size: int = 32, transform=None
    ):
        self.root_dir = Path(root_dir)
        self.window_size = window_size
        self.transform = transform

        if not self.root_dir.exists():
            raise FileNotFoundError(f"Dataset root does not exist: {self.root_dir}")

        self.sequences: List[SequenceMeta] = []
        self.sequence_offsets: List[int] = []
        total_samples = 0

        for seq_path in self._discover_sequences():
            meta = self._build_sequence_meta(seq_path)
            if meta.num_samples == 0:
                continue
            total_samples += meta.num_samples
            self.sequence_offsets.append(total_samples)
            self.sequences.append(meta)

        if total_samples == 0:
            raise ValueError(
                f"No usable sequences found under {self.root_dir} with window_size={self.window_size}"
            )

        self.total_samples = total_samples

    def _discover_sequences(self) -> List[Path]:
        """Return every directory that includes raw_images/global_pose payloads."""
        candidates = sorted(
            path
            for path in self.root_dir.glob("Chunk_*/*/*")
            if (path / "global_pose").is_dir() and (path / "raw_images").is_dir()
        )
        if not candidates:
            raise ValueError(f"No sequences found under {self.root_dir}")
        return candidates

    def _build_sequence_meta(self, seq_path: Path) -> SequenceMeta:
        pose_dir = seq_path / "global_pose"
        pose_arrays: Dict[str, np.memmap] = {}
        num_frames = None

        for key, filename in POSE_FILES.items():
            arr_path = pose_dir / filename
            if not arr_path.exists():
                raise FileNotFoundError(f"Missing pose file: {arr_path}")
            arr = np.load(arr_path, mmap_mode="r")
            pose_arrays[key] = arr
            if num_frames is None:
                num_frames = len(arr)
            elif len(arr) != num_frames:
                raise ValueError(
                    f"Mismatched pose lengths in {pose_dir}: expected {num_frames}, got {len(arr)} for {filename}"
                )

        assert num_frames is not None
        num_samples = num_frames // self.window_size

        return SequenceMeta(
            path=seq_path,
            raw_images_dir=seq_path / "raw_images",
            pose_arrays=pose_arrays,
            num_frames=num_frames,
            num_samples=num_samples,
        )

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx: int):
        if idx < 0 or idx >= self.total_samples:
            raise IndexError(idx)

        sequence_index, data_index = self._resolve_indices(idx)
        sequence = self.sequences[sequence_index]
        start = data_index * self.window_size
        end = start + self.window_size

        image_sequence = self._load_image_window(sequence, start, end)

        sample = {
            "image": image_sequence,
        }

        for key, arr in sequence.pose_arrays.items():
            sample[key] = np.asarray(arr[start:end]).copy()

        if self.transform:
            sample = self.transform(sample)

        return sample

    def export_as_latent_data(
        self,
        lmdb_path: str | os.PathLike,
        map_size: int | None = None,
        commit_interval: int = 512,
        tokenizer: CosmosVideoTokenizer | None = None,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """
        Export 16-frame windows into an LMDB with 8-frame past/future latents and actions.

        Each record stores:
            latent_past:    (C_lat, T_lat, H_lat, W_lat) float16
            latent_future:  (C_lat, T_lat, H_lat, W_lat) float16
            actions_past:   (8, 3) float32
            actions_future: (8, 3) float32
        """
        import lmdb

        if self.window_size != 16:
            raise ValueError(
                f"export_as_latent_data requires window_size=16, got {self.window_size}"
            )

        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if dtype is None:
            dtype = torch.float16 if str(device) == "cuda" else torch.float32
        if tokenizer is None:
            tokenizer = CosmosVideoTokenizer(device=str(device), dtype=dtype)

        lmdb_path = Path(lmdb_path)
        lmdb_path.parent.mkdir(parents=True, exist_ok=True)

        sample = self[0]
        frames = sample["image"].astype(np.float32)
        # latents_past_sample, latents_future_sample = self._encode_latents(
        #     frames, tokenizer
        # )
        # bytes_per_sample = (
        #     latents_past_sample.size * np.dtype(np.float16).itemsize
        #     + latents_future_sample.size * np.dtype(np.float16).itemsize
        #     + frames.shape[0] * 3 * np.dtype(np.float32).itemsize
        # )

        env = lmdb.open(
            str(lmdb_path),
            map_size=100 * 1024 * 1024 * 1024,
            subdir=True,
            lock=True,
            readahead=False,
            writemap=True,
        )
        with env.begin(write=True) as txn:
            txn.put(b"length", str(len(self)).encode("utf-8"))

        txn = env.begin(write=True)
        for idx in tqdm(range(len(self)), desc="export_as_latent_data"):
            sample = self[idx]
            frames = sample["image"].astype(np.float32)
            orientations = np.asarray(sample["orientations"])
            positions = np.asarray(sample["positions"])

            actions: List[np.ndarray] = []
            for t in range(frames.shape[0]):
                if t + 1 < frames.shape[0]:
                    delta = self._get_diff_2d_pose(
                        orientations[t],
                        positions[t],
                        orientations[t + 1],
                        positions[t + 1],
                    )
                else:
                    delta = np.zeros(3, dtype=np.float64)
                actions.append(delta)

            actions_np = np.stack(actions, axis=0).astype(np.float32)
            latents_past, latents_future = self._encode_latents(frames, tokenizer)
            payload = {
                "latent_past": latents_past,
                "latent_future": latents_future,
                "actions_past": actions_np[:8],
                "actions_future": actions_np[8:],
            }

            buffer = io.BytesIO()
            np.savez_compressed(buffer, **payload)
            key = f"{idx:08d}".encode("utf-8")
            txn.put(key, buffer.getvalue())

            if (idx + 1) % commit_interval == 0:
                txn.commit()
                txn = env.begin(write=True)

        txn.commit()
        env.sync()

    def _encode_latents(
        self, frames: np.ndarray, tokenizer: CosmosVideoTokenizer
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert 16-frame RGB uint8/float32 (T, H, W, 3) into past/future latents.
        """
        video = torch.from_numpy(frames).permute(3, 0, 1, 2)  # (C, T, H, W)
        video = video.unsqueeze(0).to(tokenizer.device)  # (1, C, T, H, W)

        video_past, video_future = torch.split(video, 8, dim=2)  # (C, 8, H, W)

        latents_past = tokenizer.encode(video_past).cpu().numpy().astype(np.float16)[0]
        latents_future = (
            tokenizer.encode(video_future).cpu().numpy().astype(np.float16)[0]
        )
        return latents_past, latents_future

    def _resolve_indices(self, index: int) -> Tuple[int, int]:
        sequence_idx = bisect_right(self.sequence_offsets, index)
        sequence_start = (
            0 if sequence_idx == 0 else self.sequence_offsets[sequence_idx - 1]
        )
        data_idx = index - sequence_start
        return sequence_idx, data_idx

    def _load_image_window(
        self, sequence: SequenceMeta, start: int, end: int
    ) -> np.ndarray:
        frames: List[np.ndarray] = []
        for frame_idx in range(start, end):
            frame_path = (
                sequence.raw_images_dir / f"frame_{frame_idx:0{self.FRAME_DIGITS}d}.jpg"
            )
            frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
            if frame is None:
                raise FileNotFoundError(f"Unable to read frame: {frame_path}")
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame.astype(np.float32))

        return np.stack(frames, axis=0)

    @staticmethod
    def _get_diff_2d_pose(
        current_orientation: np.ndarray,
        current_position: np.ndarray,
        next_orientation: np.ndarray,
        next_position: np.ndarray,
    ) -> np.ndarray:
        """Return [x, y, yaw] from current pose to the next in the local frame."""

        vector = ecef_vector_from_pose(
            origin_position=current_position,
            origin_orientation=current_orientation,
            target_position=next_position,
        )
        components = vector.components
        delta_quaternion = quaternion_difference(current_orientation, next_orientation)
        yaw = quaternion_to_yaw(delta_quaternion)
        return np.array([components[0], components[1], yaw], dtype=np.float64)


def main():
    dataset_dir = Path(os.path.dirname(os.path.abspath(__file__))) / "dataset"
    comma_dataset = CommaDataset(dataset_dir)
    comma_dataloader = DataLoader(
        comma_dataset, batch_size=1, shuffle=False, num_workers=0
    )
    sample = next(iter(comma_dataloader))
    print("image window:", sample["image"].shape)
    print("velocities window:", sample["velocities"].shape)
    print(
        "diff 2d pose between first two frames:",
        CommaDataset._get_diff_2d_pose(
            sample["orientations"][0, 0].numpy(),
            sample["positions"][0, 0].numpy(),
            sample["orientations"][0, 1].numpy(),
            sample["positions"][0, 1].numpy(),
        ),
    )


if __name__ == "__main__":
    main()
