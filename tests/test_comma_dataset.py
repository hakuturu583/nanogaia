from pathlib import Path

import cv2
import numpy as np
import pytest

from nanogaia.comma2k19_dataset import CommaDataset


def _save_array(path: Path, array: np.ndarray):
    with open(path, "wb") as fp:
        np.save(fp, array)


def _write_sequence(seq_path: Path, num_frames: int, start_value: int):
    raw_dir = seq_path / "raw_images"
    pose_dir = seq_path / "global_pose"
    raw_dir.mkdir(parents=True, exist_ok=True)
    pose_dir.mkdir(parents=True, exist_ok=True)

    for idx in range(num_frames):
        img = np.full((2, 3, 3), start_value + idx, dtype=np.uint8)
        cv2.imwrite(str(raw_dir / f"frame_{idx:05d}.jpg"), img)

    base = np.arange(num_frames, dtype=np.float32) + float(start_value)
    _save_array(pose_dir / "frame_gps_times", base)
    _save_array(pose_dir / "frame_times", base + 1)
    _save_array(
        pose_dir / "frame_positions", np.stack([base, base + 1, base + 2], axis=1)
    )
    _save_array(
        pose_dir / "frame_orientations",
        np.stack([base, base + 1, base + 2, base + 3], axis=1),
    )
    _save_array(
        pose_dir / "frame_velocities",
        np.stack([base, base + 1, base + 2], axis=1),
    )

    return {"num_frames": num_frames, "start_value": start_value}


@pytest.fixture()
def toy_dataset(tmp_path):
    root = tmp_path / "toy_dataset"
    info = []
    info.append(
        _write_sequence(
            root / "Chunk_1" / "session_a" / "0", num_frames=8, start_value=5
        )
    )
    info.append(
        _write_sequence(
            root / "Chunk_2" / "session_b" / "1", num_frames=4, start_value=100
        )
    )
    return root, info


def test_dataset_length_matches_total_windows(toy_dataset):
    root, info = toy_dataset
    dataset = CommaDataset(root, window_size=4)
    expected = sum(seq["num_frames"] // 4 for seq in info)
    assert len(dataset) == expected


def test_sequence_and_data_indices_are_resolved(toy_dataset):
    root, _ = toy_dataset
    dataset = CommaDataset(root, window_size=4)
    assert dataset._resolve_indices(0) == (0, 0)
    assert dataset._resolve_indices(1) == (0, 1)
    assert dataset._resolve_indices(2) == (1, 0)


def test_getitem_returns_correct_window_slices(toy_dataset):
    root, info = toy_dataset
    dataset = CommaDataset(root, window_size=4)

    first = dataset[0]
    assert first["image"].shape == (4, 2, 3, 3)
    np.testing.assert_allclose(
        first["velocities"][:, 0], np.arange(4) + info[0]["start_value"]
    )

    last = dataset[len(dataset) - 1]
    np.testing.assert_allclose(
        last["velocities"][:, 0], np.arange(4) + info[1]["start_value"]
    )
