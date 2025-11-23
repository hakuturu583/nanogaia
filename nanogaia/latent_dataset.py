from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Callable, Dict

import lmdb
import numpy as np
from torch.utils.data import Dataset
from nanogaia.model import CosmosVideoTokenizer


class LatentDataset(Dataset):
    """
    LMDB-backed dataset produced by CommaDataset.export_as_latent_data.

    Each record contains:
        latent_past:    (C_lat, T_lat, H_lat, W_lat) float16
        latent_future:  (C_lat, T_lat, H_lat, W_lat) float16
        actions_past:   (8, 3) float32
        actions_future: (8, 3) float32
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
                "latent_past": npz["latent_past"],
                "latent_future": npz["latent_future"],
                "actions_past": npz["actions_past"],
                "actions_future": npz["actions_future"],
            }

        if self.transform:
            sample = self.transform(sample)
        return sample

    def decode(
        self,
        index: int,
        tokenizer: CosmosVideoTokenizer,
        output_path: str,
        fps: int = 4,
    ) -> str:
        """
        Decode past+future latents at the given index and write an mp4 video.
        """
        sample = self[index]
        latents = np.concatenate(
            [sample["latent_past"], sample["latent_future"]], axis=1
        )
        return tokenizer.decode_as_video(latents, output_path, fps=fps)

    def close(self) -> None:
        if getattr(self, "env", None) is not None:
            self.env.close()
            self.env = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


if __name__ == "__main__":
    import argparse

    default_lmdb = Path(__file__).resolve().parent / "dataset" / "latent.lmdb"
    parser = argparse.ArgumentParser(description="Inspect latent LMDB dataset")
    parser.add_argument(
        "--lmdb",
        type=Path,
        default=default_lmdb,
        help=f"Path to latent LMDB (default: {default_lmdb})",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Sample index to inspect",
    )
    args = parser.parse_args()

    dataset = LatentDataset(args.lmdb)
    print(f"Dataset length: {len(dataset)}")
    sample = dataset[args.index]
    print("Sample keys:", list(sample.keys()))
    print("Latent past shape:", sample["latent_past"].shape)
    print("Latent future shape:", sample["latent_future"].shape)
    print("Actions past shape:", sample["actions_past"].shape)
    print("Actions future shape:", sample["actions_future"].shape)
    dataset.decode(
        args.index,
        tokenizer=CosmosVideoTokenizer(),
        output_path=f"sample_{args.index:08d}.mp4",
    )
