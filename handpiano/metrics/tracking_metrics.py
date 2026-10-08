"""Tracking-quality metrics computed from what the tracker actually reported."""

from __future__ import annotations

import threading
from collections import deque


class RollingFraction:
    """Fraction of recent samples that were True. ``None`` until there is at least one sample.

    Used for "frames without this hand" (detector output only: a hand taken out of
    view on purpose also counts) and for the left/right spatial-order check.
    """

    def __init__(self, window: int = 300) -> None:
        self._samples: deque[bool] = deque(maxlen=window)
        self._lock = threading.Lock()

    def add(self, value: bool) -> None:
        with self._lock:
            self._samples.append(value)

    def fraction(self) -> float | None:
        with self._lock:
            if not self._samples:
                return None
            return sum(self._samples) / len(self._samples)

    def count(self) -> int:
        with self._lock:
            return len(self._samples)
