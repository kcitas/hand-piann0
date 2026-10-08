"""Rolling statistics for latency-like measurements (milliseconds)."""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class StatsSummary:
    count: int
    last: float | None
    mean: float | None
    p50: float | None
    p95: float | None
    max: float | None


EMPTY_SUMMARY = StatsSummary(count=0, last=None, mean=None, p50=None, p95=None, max=None)


class RollingStats:
    """Keeps the last ``window`` samples. Thread-safe: written by a worker, read by the UI."""

    def __init__(self, window: int = 300) -> None:
        if window < 1:
            raise ValueError("window must be >= 1")
        self._samples: deque[float] = deque(maxlen=window)
        self._lock = threading.Lock()

    def add(self, value: float) -> None:
        with self._lock:
            self._samples.append(float(value))

    def clear(self) -> None:
        with self._lock:
            self._samples.clear()

    def summary(self) -> StatsSummary:
        with self._lock:
            if not self._samples:
                return EMPTY_SUMMARY
            data = np.fromiter(self._samples, dtype=np.float64, count=len(self._samples))
            last = self._samples[-1]
        p50, p95 = np.percentile(data, [50, 95])
        return StatsSummary(
            count=len(data),
            last=last,
            mean=float(data.mean()),
            p50=float(p50),
            p95=float(p95),
            max=float(data.max()),
        )
