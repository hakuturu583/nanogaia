from __future__ import annotations

from typing import List

import av
import numpy as np


class FrameReader:
    """
    FrameReader that loads all frames of an MP4 video into memory at initialization.

    - Input: MP4 file (e.g., H.264 / HEVC in MP4 container)
    - Attributes:
        - pix_fmt: pixel format to decode into ("rgb24", "bgr24", or "gray", default "rgb24")
        - frame_count: total number of frames (int)
        - frames: list of np.ndarray, one per frame
    - Methods:
        - get(fidx) -> np.ndarray
          Returns frame as numpy array (random access in O(1))
    """

    def __init__(
        self,
        filename: str,
        pix_fmt: str = "rgb24",
    ) -> None:
        self.filename = filename
        self.pix_fmt = pix_fmt

        if self.pix_fmt not in ("rgb24", "bgr24", "gray"):
            raise ValueError(f"Unsupported pix_fmt: {self.pix_fmt}")

        # Decode the entire video once and keep all frames in memory.
        self.frames: List[np.ndarray] = []
        self._load_all_frames()

        self.frame_count: int = len(self.frames)
        if self.frame_count == 0:
            raise RuntimeError(f"No video frames found in file: {self.filename}")

    def _load_all_frames(self) -> None:
        """
        Decode all frames from the video and store them as numpy arrays.
        """
        container = av.open(self.filename)
        try:
            stream = container.streams.video[0]

            for frame in container.decode(stream):
                if self.pix_fmt == "gray":
                    arr = frame.to_ndarray(format="gray")   # (H, W)
                else:
                    # "rgb24" or "bgr24"
                    arr = frame.to_ndarray(format=self.pix_fmt)  # (H, W, 3)

                self.frames.append(arr)

        finally:
            container.close()

    def get(self, fidx: int) -> np.ndarray:
        """
        Return frame fidx as a numpy array.
        """
        if fidx < 0:
            raise IndexError(f"Negative frame index: {fidx}")
        if fidx >= self.frame_count:
            raise IndexError(f"Frame index out of range: {fidx} (total={self.frame_count})")

        return self.frames[fidx]

    def __len__(self) -> int:
        """Return total number of frames."""
        return self.frame_count

    def close(self) -> None:
        """
        Release references to cached frames.
        Call this if you want to free memory explicitly.
        """
        self.frames = []
        self.frame_count = 0

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
