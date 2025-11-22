from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Callable, Dict

import lmdb
import numpy as np
from torch.utils.data import Dataset


class LatentDataset(Dataset):
    """
    LMDB-backed dataset produced by CommaDataset.export_as_latent_data.

    Each record contains:
        video_past:   (8, H, W, 3) uint8
        video_future: (8, H, W, 3) uint8
        actions_past: (8, 3) float32
        actions_future:(8, 3) float32
    """

    def __init__(
        self,
        lmdb_path: str | Path,
        transform: Callable[[Dict[str, np.ndarray]], Dict[str, Any]] | None = None,
    ):
        self.lmdb_path = Path(lmdb_path)
        if not self.lmdb_path.exists():
            raise FileNotFoundError(f"LMDB path does not exist: {self.lmdb_path}")

        self.env = lmdb.open(
            str(self.lmdb_path), readonly=True, lock=False, readahead=False
        )

        with self.env.begin(write=False) as txn:
            length_bytes = txn.get(b"length")
            if length_bytes is None:
                raise ValueError("LMDB is missing 'length' metadata")
            self.length = int(length_bytes.decode("utf-8"))

        self.transform = transform

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> Dict[str, np.ndarray] | Dict[str, Any]:
        if index < 0 or index >= self.length:
            raise IndexError(index)

        key = f"{index:08d}".encode("utf-8")
        with self.env.begin(write=False) as txn:
            raw = txn.get(key)
            if raw is None:
                raise KeyError(f"Missing key {key!r} in LMDB")

        with io.BytesIO(raw) as buffer:
            npz = np.load(buffer)
            sample: Dict[str, np.ndarray] = {
                "video_past": npz["video_past"],
                "video_future": npz["video_future"],
                "actions_past": npz["actions_past"],
                "actions_future": npz["actions_future"],
            }

        if self.transform:
            sample = self.transform(sample)
        return sample

    def close(self) -> None:
        if getattr(self, "env", None) is not None:
            self.env.close()
            self.env = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
