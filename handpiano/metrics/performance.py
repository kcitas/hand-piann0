"""Frame-rate and CPU measurements based on real timestamps."""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass


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


class CpuComponent:
    """Names of the parts of the program whose CPU time is accounted separately."""

    CAPTURE = "captura"  # camera read + mirror + BGR→RGB, capture thread
    INFERENCE = "inferencia"  # MediaPipe call, measured on the tracking thread only
    TRACKER = "tracker"  # identity + outliers + smoothing + finger state
    UI = "UI"  # Qt main thread: polling, painting, stats


class CpuAccounting:
    """Accumulates CPU seconds per component.

    Each thread measures its own ``time.thread_time()`` around its work and adds
    the delta here. Threads that HandPiano does not own (MediaPipe's internal
    worker pool, Qt/OS threads) cannot be measured this way; their CPU shows up
    as the difference between process CPU and the sum of components.
    """

    def __init__(self) -> None:
        self._totals: defaultdict[str, float] = defaultdict(float)
        self._lock = threading.Lock()

    def add(self, component: str, cpu_seconds: float) -> None:
        if cpu_seconds > 0:
            with self._lock:
                self._totals[component] += cpu_seconds

    def totals(self) -> dict[str, float]:
        with self._lock:
            return dict(self._totals)


@dataclass(frozen=True, slots=True)
class CpuBreakdown:
    """CPU use over one sampling interval, in % of ONE core (can exceed 100 %)."""

    process_pct: float
    components_pct: dict[str, float]

    @property
    def unattributed_pct(self) -> float:
        """Process CPU not measured by any component (MediaPipe worker threads, Qt, OS)."""
        return max(0.0, self.process_pct - sum(self.components_pct.values()))


class CpuSampler:
    """Turns cumulative CPU counters into per-interval percentages."""

    def __init__(
        self,
        accounting: CpuAccounting | None = None,
        clock=time.perf_counter,
        process_clock=time.process_time,
    ) -> None:
        self._accounting = accounting
        self._clock = clock
        self._process_clock = process_clock
        self._last: tuple[float, float, dict[str, float]] | None = None

    def sample(self) -> CpuBreakdown | None:
        wall, cpu = self._clock(), self._process_clock()
        totals = self._accounting.totals() if self._accounting else {}
        previous, self._last = self._last, (wall, cpu, totals)
        if previous is None:
            return None
        prev_wall, prev_cpu, prev_totals = previous
        elapsed = wall - prev_wall
        if elapsed <= 0:
            return None
        components = {
            name: 100.0 * (seconds - prev_totals.get(name, 0.0)) / elapsed for name, seconds in totals.items()
        }
        return CpuBreakdown(process_pct=100.0 * (cpu - prev_cpu) / elapsed, components_pct=components)
