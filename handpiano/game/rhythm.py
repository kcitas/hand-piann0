"""4-lane rhythm game logic: judging, score, combo, accuracy. No Qt, no audio.

The clock is the caller's (``time.perf_counter()`` seconds, the same clock as
press events, whose ``t`` is the camera frame capture time, so pipeline latency
does not count against the player). Each PRESS (one event, or one key down) is
judged at most once and can hit at most one note; a finger held down does not
generate more presses.

Windows (|press − note|): PERFECT ≤ 50 ms, GREAT ≤ 100 ms, GOOD ≤ 160 ms; a note
not hit 160 ms after its time is a MISS (combo resets). A press with no note in
its window is ignored (no penalty). Accuracy = Σ weight / judged notes with
weights 1, 0.7, 0.4, 0.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from handpiano.game.chart import LANES, Chart, ChartNote

WINDOWS = (("PERFECT", 0.050, 300, 1.0), ("GREAT", 0.100, 200, 0.7), ("GOOD", 0.160, 100, 0.4))
MISS_AFTER = WINDOWS[-1][1]
LEAD_IN_S = 2.0


@dataclass(frozen=True, slots=True)
class Judgment:
    note: ChartNote
    grade: str  # PERFECT, GREAT, GOOD, MISS
    offset_s: float | None  # press − note time; None for a miss
    t: float  # song time of the judgment


@dataclass
class RhythmGame:
    chart: Chart | None = None
    score: int = 0
    combo: int = 0
    max_combo: int = 0
    judgments: list[Judgment] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=lambda: {"PERFECT": 0, "GREAT": 0, "GOOD": 0, "MISS": 0})
    # Called once per judgment (e.g. the audio plays the note on a hit). Must not raise.
    on_judgment: list[Callable[[Judgment], None]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._t0: float | None = None
        self._pending: list[list[ChartNote]] = [[] for _ in range(LANES)]
        self._weights = 0.0

    def start(self, chart: Chart, now: float, lead_in: float = LEAD_IN_S) -> None:
        self.chart = chart
        self._t0 = now + lead_in
        self.score = self.combo = self.max_combo = 0
        self.judgments = []
        self.counts = {k: 0 for k in self.counts}
        self._weights = 0.0
        self._pending = [[n for n in chart.notes if n.lane == lane] for lane in range(LANES)]

    def stop(self) -> None:
        self._t0 = None

    @property
    def running(self) -> bool:
        return self._t0 is not None

    def song_time(self, now: float) -> float:
        return 0.0 if self._t0 is None else now - self._t0

    def _judge(self, j: Judgment, weight: float, points: int) -> Judgment:
        self.judgments.append(j)
        self.counts[j.grade] += 1
        self._weights += weight
        if j.grade == "MISS":
            self.combo = 0
        else:
            self.combo += 1
            self.max_combo = max(self.max_combo, self.combo)
            self.score += points * (1 + min(self.combo, 100) // 25)
        for callback in self.on_judgment:
            callback(j)
        return j

    def press(self, lane: int, now: float) -> Judgment | None:
        """One press in ``lane`` at clock time ``now``. Returns the judgment, or None if nothing to hit."""
        if self._t0 is None or not 0 <= lane < LANES:
            return None
        t = self.song_time(now)
        self.update(now)
        for note in self._pending[lane]:
            offset = t - note.t
            if offset < -MISS_AFTER:
                break  # notes are sorted: the rest are later
            if abs(offset) <= MISS_AFTER:
                self._pending[lane].remove(note)
                for grade, window, points, weight in WINDOWS:
                    if abs(offset) <= window:
                        return self._judge(Judgment(note, grade, offset, t), weight, points)
        return None

    def update(self, now: float) -> list[Judgment]:
        """Marks as MISS every note whose window has passed."""
        if self._t0 is None:
            return []
        t = self.song_time(now)
        out = []
        for lane in range(LANES):
            while self._pending[lane] and t - self._pending[lane][0].t > MISS_AFTER:
                note = self._pending[lane].pop(0)
                out.append(self._judge(Judgment(note, "MISS", None, t), 0.0, 0))
        return out

    @property
    def accuracy(self) -> float | None:
        judged = len(self.judgments)
        return None if judged == 0 else self._weights / judged

    def finished(self, now: float) -> bool:
        return self.chart is not None and self._t0 is not None and not any(self._pending) and \
            self.song_time(now) > self.chart.length_s + 1.0

    def upcoming(self, now: float, horizon_s: float) -> list[ChartNote]:
        """Notes not yet judged that are due within ``horizon_s`` (for drawing)."""
        t = self.song_time(now)
        return [n for lane in self._pending for n in lane if n.t - t <= horizon_s]
