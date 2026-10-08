"""Captured frames and the single-slot hand-off between capture and tracking."""

from __future__ import annotations

import threading
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class Frame:
    index: int
    # RGB, already mirrored if the camera settings ask for it. Read-only: the
    # same array is shared by the tracking thread and the UI.
    image: np.ndarray
    # time.perf_counter_ns() right after the driver returned the frame. Sensor
    # exposure and driver buffering happen BEFORE this instant and are not measurable here.
    captured_at_ns: int
    # time.perf_counter_ns() after mirror + BGR→RGB, when the frame was put in the slot.
    published_at_ns: int

    @property
    def width(self) -> int:
        return int(self.image.shape[1])

    @property
    def height(self) -> int:
        return int(self.image.shape[0])


class LatestFrameSlot:
    """Holds only the most recent frame.

    If the consumer is slower than the camera, older frames are overwritten
    (and counted as dropped) instead of queuing up and adding latency.
    """

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._frame: Frame | None = None
        self._closed = False
        self.produced = 0
        self.dropped = 0

    def put(self, frame: Frame) -> None:
        with self._cond:
            if self._frame is not None:
                self.dropped += 1
            self._frame = frame
            self.produced += 1
            self._cond.notify()

    def take(self, timeout: float | None = None) -> Frame | None:
        """Waits for a frame newer than the last one taken. None on timeout or close."""
        with self._cond:
            self._cond.wait_for(lambda: self._frame is not None or self._closed, timeout)
            frame, self._frame = self._frame, None
            return frame

    @property
    def pending(self) -> int:
        """Frames waiting to be taken: by construction 0 or 1."""
        with self._cond:
            return 0 if self._frame is None else 1

    def close(self) -> None:
        with self._cond:
            self._closed = True
            self._cond.notify_all()
