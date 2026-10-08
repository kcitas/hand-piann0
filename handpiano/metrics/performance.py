"""Frame-rate and CPU measurements based on real timestamps."""

from __future__ import annotations

import threading
import time
from collections import deque


class RateCounter:
    """Events per second over a sliding time window.

    Returns ``None`` until at least two events exist in the window, so the UI
    shows "N/A" instead of a fabricated number.
    """

    def __init__(self, window_s: float = 2.0) -> None:
        self._window_s = window_s
        self._ticks: deque[float] = deque()
        self._lock = threading.Lock()

    def tick(self, now: float | None = None) -> None:
        now = time.perf_counter() if now is None else now
        with self._lock:
            self._ticks.append(now)
            self._trim(now)

    def rate(self, now: float | None = None) -> float | None:
        now = time.perf_counter() if now is None else now
        with self._lock:
            self._trim(now)
            if len(self._ticks) < 2:
                return None
            span = self._ticks[-1] - self._ticks[0]
            if span <= 0:
                return None
            return (len(self._ticks) - 1) / span

    def _trim(self, now: float) -> None:
        while self._ticks and now - self._ticks[0] > self._window_s:
            self._ticks.popleft()


class ProcessCpuMonitor:
    """CPU used by this process, as a percentage of ONE core (can exceed 100 % on multicore).

    Uses ``time.process_time`` (user + system time of all threads) against wall time.
    """

    def __init__(self, clock=time.perf_counter, cpu_clock=time.process_time) -> None:
        self._clock = clock
        self._cpu_clock = cpu_clock
        self._last_wall: float | None = None
        self._last_cpu: float | None = None

    def sample(self) -> float | None:
        wall, cpu = self._clock(), self._cpu_clock()
        previous_wall, previous_cpu = self._last_wall, self._last_cpu
        self._last_wall, self._last_cpu = wall, cpu
        if previous_wall is None or previous_cpu is None or wall <= previous_wall:
            return None
        return 100.0 * (cpu - previous_cpu) / (wall - previous_wall)
