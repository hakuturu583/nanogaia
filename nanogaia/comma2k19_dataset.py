from __future__ import annotations

import os
from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

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
