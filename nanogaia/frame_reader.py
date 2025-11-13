from __future__ import annotations

from collections import OrderedDict
from typing import Optional

import cv2
import numpy as np


class FrameReader:
    """
    Simplified HEVC FrameReader with an API similar to openpilot's FrameReader.

    - Input: HEVC (.hevc / .hvec) file
    - Attributes:
        - pix_fmt: pixel format to decode into ("rgb24" or "bgr24", default "rgb24")
    - Methods:
        - get(fidx) -> np.ndarray
          Returns frame as numpy array (with LRU caching)
    """

    def __init__(
        self,
        filename: str,
        pix_fmt: str = "rgb24",
        cache_size: int = 16,
    ) -> None:
        self.filename = filename
        self.pix_fmt = pix_fmt
        self.cache_size = cache_size

        if self.pix_fmt not in ("rgb24", "bgr24"):
            raise ValueError(f"Unsupported pix_fmt: {self.pix_fmt}")

        # Open the HEVC file using OpenCV VideoCapture
        self._cap = cv2.VideoCapture(self.filename)
        if not self._cap.isOpened():
            raise IOError(f"Failed to open video file: {self.filename}")

        # Total number of frames (may be -1 depending on codec/container)
        self.frame_count: int = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # LRU cache: {frame_index: np.ndarray}
        self._cache: "OrderedDict[int, np.ndarray]" = OrderedDict()

    def _read_frame(self, fidx: int) -> np.ndarray:
        """
        Internal method that seeks and decodes one frame from VideoCapture.
        Also converts pixel format according to pix_fmt.
        """
        # Seek to the target frame
        # NOTE: Seeking precision depends on the codec/container.
        ok = self._cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
        if not ok:
            raise IOError(f"Failed to seek to frame {fidx}")

        # Read the frame
        ret, frame = self._cap.read()
        if not ret or frame is None:
            raise IOError(f"Failed to read frame {fidx}")

        # OpenCV uses BGR; convert to RGB if necessary
        if self.pix_fmt == "rgb24":
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # "bgr24" requires no conversion

        return frame

    def get(self, fidx: int) -> np.ndarray:
        """
        Return frame fidx as a numpy array.
        """
        if fidx < 0:
            raise IndexError(f"Negative frame index: {fidx}")

        if self.frame_count > 0 and fidx >= self.frame_count:
            # Only check upper bound when frame_count is known
            raise IndexError(
                f"Frame index out of range: {fidx} (total={self.frame_count})"
            )

        # Cache hit → return cached frame (update LRU)
        if fidx in self._cache:
            frame = self._cache.pop(fidx)
            self._cache[fidx] = frame  # Move to end (most recently used)
            return frame

        # Cache miss → decode frame
        frame = self._read_frame(fidx)

        # Insert into cache; drop the oldest if exceeding capacity
        self._cache[fidx] = frame
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)  # Remove oldest (LRU eviction)

        return frame

    def close(self) -> None:
        """Release underlying resources."""
        if getattr(self, "_cap", None) is not None:
            self._cap.release()
            self._cap = None

    def __del__(self) -> None:
        # Ensure resources are released (though calling close() explicitly is preferred)
        try:
            self.close()
        except Exception:
            pass
