"""Tracking-quality metrics computed from what the tracker actually reported."""

from __future__ import annotations

import threading
from collections import deque


class PresenceRate:
    """Fraction of recent processed frames in which something was NOT tracked.

    It only measures detector output: if the user takes a hand out of view on
    purpose, that also counts as lost. It is a tracking-stability signal, not
    an accuracy score.
    """

    def __init__(self, window: int = 300) -> None:
        self._samples: deque[bool] = deque(maxlen=window)
        self._lock = threading.Lock()

    def add(self, present: bool) -> None:
        with self._lock:
            self._samples.append(present)

    def lost_rate(self) -> float | None:
        with self._lock:
            if not self._samples:
                return None
            return 1.0 - sum(self._samples) / len(self._samples)
