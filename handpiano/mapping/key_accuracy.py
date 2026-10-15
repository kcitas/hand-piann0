"""Guided finger → key accuracy measurement (``python -m handpiano --check-keys``).

This is a measurement tool, not a calibration: it requires an existing valid
calibration and never creates one.

Test points are deterministic, defined in PIANO coordinates from the keyboard
geometry (never arbitrary camera coordinates):

* white-key centres spread over the keyboard;
* black-key centres;
* points near borders (either side of a white/white boundary, just inside and
  just below a black key);
* points outside the keyboard (above, below, left, right).

The expected key of a point is ``layout.key_at(point)`` (None outside). Each
point is shown on the video (projected with the same calibration) and its key
highlighted; the person puts ONE index fingertip on it. A sample is taken once
the fingertip has stayed within ``stable_radius`` for ``stable_s`` (after at
least ``min_settle_s`` on that target): its mean camera position goes through
``KeyMapper.map_point`` — the same homography and ``key_at`` used for every
frame. If no stable fingertip appears within ``timeout_s`` the sample is a
``miss`` with a ``miss_reason`` (the dominant problem while waiting:
``no_index``, ``two_hands``, ``unstable`` or ``did_not_move``).

After each sample the fingertip must move at least ``min_move`` away from where
the previous sample was taken before a new one can be recorded (unless the new
target itself is that close). Without this, a finger still resting on the
previous target was recorded again for the next ones (seen in a real run).
Results only mean something if the person really pointed at the targets.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

from handpiano.instrument.keyboard import KeyboardLayout, PianoKey
from handpiano.mapping.key_mapper import KeyMapper
from handpiano.tracking.landmarks import FingerId

# Inside targets
EXACT = "exact"
ADJACENT = "adjacent"
FAR = "far"
MISS = "miss"
# Outside targets
OUTSIDE_CORRECT = "outside_correct"
OUTSIDE_FALSE_POSITIVE = "outside_false_positive"


@dataclass(frozen=True, slots=True)
class ProbePoint:
    sample_id: int
    kind: str  # "white", "black", "border", "outside"
    piano_xy: tuple[float, float]
    expected: PianoKey | None  # None for outside points

    @property
    def label(self) -> str:
        return "FUERA del piano" if self.expected is None else self.expected.name


def default_probe_points(layout: KeyboardLayout) -> tuple[ProbePoint, ...]:
    w, h = layout.size
    black_h = layout.black_keys[0].height if layout.black_keys else 0.0
    front_y = (black_h + h) / 2  # in front of the black keys: only white keys there
    raw: list[tuple[str, tuple[float, float]]] = []
    raw += [("white", k.center) for k in layout.white_keys[::2]]
    raw += [("black", k.center) for k in layout.black_keys[::2]]
    # Near borders: 0.2 white-key widths either side of the E|F and B|C boundaries
    # of the first octave, and just inside / just below a black key.
    for boundary in (3.0, 7.0):
        if boundary < w:
            raw += [("border", (boundary - 0.2, front_y)), ("border", (boundary + 0.2, front_y))]
    if layout.black_keys:
        b = layout.black_keys[len(layout.black_keys) // 2]
        raw += [("border", (b.x + b.width / 2 + 0.05, black_h - 0.3)), ("border", (b.x + b.width / 2 + 0.05, black_h + 0.3))]
    # Outside: a quarter of the keyboard depth above/below, one white key left/right.
    raw += [("outside", (w / 2, -0.25 * h)), ("outside", (w / 2, 1.25 * h)),
            ("outside", (-1.0, h / 2)), ("outside", (w + 1.0, h / 2))]
    return tuple(ProbePoint(i + 1, kind, xy, layout.key_at(*xy)) for i, (kind, xy) in enumerate(raw))


def _empty_diag() -> dict[str, int]:
    return {"no_index": 0, "two_hands": 0, "unstable": 0, "did_not_move": 0}


def keys_adjacent(a: PianoKey, b: PianoKey, eps: float = 1e-9) -> bool:
    """Different keys whose rectangles touch or overlap (white neighbours, a black key and its whites)."""
    if a.midi == b.midi:
        return False
    return (a.x <= b.x + b.width + eps and b.x <= a.x + a.width + eps
            and a.y <= b.y + b.height + eps and b.y <= a.y + a.height + eps)


def categorize(expected: PianoKey | None, detected: PianoKey | None, had_position: bool) -> str:
    if not had_position:
        return MISS
    if expected is None:
        return OUTSIDE_CORRECT if detected is None else OUTSIDE_FALSE_POSITIVE
    if detected is None:
        return MISS
    if detected.midi == expected.midi:
        return EXACT
    return ADJACENT if keys_adjacent(expected, detected) else FAR


@dataclass(frozen=True, slots=True)
class AccuracySample:
    sample_id: int
    kind: str
    expected_key: str | None
    detected_key: str | None
    correct: bool
    error_category: str
    # For misses: the dominant problem while waiting for this target; None otherwise.
    miss_reason: str | None
    # Frames seen while waiting for this target, by problem (diagnostics).
    frames_no_index: int
    frames_two_hands: int
    frames_waiting_move: int
    stability_resets: int
    finger: FingerId | None
    timestamp: str  # wall clock, ISO 8601
    t: float  # monotonic seconds since the test started
    target_piano_xy: tuple[float, float]
    piano_xy: tuple[float, float] | None
    camera_xy: tuple[float, float] | None
    distance_to_expected_center: float | None


@dataclass(frozen=True, slots=True)
class AccuracyConfig:
    prepare_s: float = 4.0
    min_settle_s: float = 1.5  # time on a new target before a sample may be taken
    stable_s: float = 0.6
    stable_radius: float = 0.01  # frame-width units (y scaled by height/width)
    # 10 s proved too short in a real run (the person reached a target just after it timed out).
    timeout_s: float = 15.0
    # Movement required after a sample before the next one (frame-width units, ≈ 15 px at 1280).
    min_move: float = 0.012


@dataclass(frozen=True, slots=True)
class AccuracyReport:
    inside: int
    exact: int
    adjacent: int
    far: int
    miss_inside: int
    outside: int
    outside_correct: int
    outside_false_positive: int
    miss_outside: int

    @property
    def exact_accuracy(self) -> float | None:
        return None if self.inside == 0 else self.exact / self.inside

    @property
    def outside_accuracy(self) -> float | None:
        return None if self.outside == 0 else self.outside_correct / self.outside

    def lines(self) -> list[str]:
        def pct(v: float | None) -> str:
            return "N/A" if v is None else f"{v * 100:.1f}%"

        return [
            "=== KEY ACCURACY ===",
            f"Samples: {self.inside}",
            f"Exact: {self.exact}",
            f"Adjacent: {self.adjacent}",
            f"Far: {self.far}",
            f"Miss: {self.miss_inside}",
            f"Exact accuracy: {pct(self.exact_accuracy)}",
            "=== OUTSIDE DETECTION ===",
            f"Samples: {self.outside}",
            f"Correct: {self.outside_correct}",
            f"False positives: {self.outside_false_positive}",
            f"Miss: {self.miss_outside}",
            f"Outside accuracy: {pct(self.outside_accuracy)}",
        ]


@dataclass(slots=True)
class KeyAccuracyCheck:
    mapper: KeyMapper
    points: tuple[ProbePoint, ...] = ()
    config: AccuracyConfig = field(default_factory=AccuracyConfig)
    aspect: float = 9 / 16
    samples: list[AccuracySample] = field(default_factory=list)
    started_at: float | None = None
    _target_started_at: float | None = None
    _anchor: np.ndarray | None = None
    _anchor_t: float | None = None
    _window: list[tuple[float, float]] = field(default_factory=list)
    _window_finger: FingerId | None = None
    _last_sample_iso: np.ndarray | None = None
    _needs_move: bool = False
    _diag: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.points:
            self.points = default_probe_points(self.mapper.layout)

    # -- schedule ----------------------------------------------------------------

    def start(self, t: float) -> None:
        self.started_at = t
        self.samples = []
        self._target_started_at = t + self.config.prepare_s
        self._last_sample_iso = None
        self._needs_move = False
        self._diag = _empty_diag()
        self._reset_window()

    @property
    def waiting_to_move(self) -> bool:
        return self._needs_move

    def _iso(self, xy: tuple[float, float]) -> np.ndarray:
        return np.array([xy[0], xy[1] * self.aspect])

    @property
    def done(self) -> bool:
        return len(self.samples) >= len(self.points)

    @property
    def current(self) -> ProbePoint | None:
        return None if self.done else self.points[len(self.samples)]

    def preparing(self, t: float) -> bool:
        return self.started_at is not None and self._target_started_at is not None and t < self._target_started_at

    def stability(self, t: float) -> float:
        """0..1 progress of the stable-hold for the current target."""
        if self._anchor_t is None:
            return 0.0
        return min(1.0, (t - self._anchor_t) / self.config.stable_s)

    def time_left(self, t: float) -> float:
        if self._target_started_at is None:
            return 0.0
        return max(0.0, self._target_started_at + self.config.timeout_s - t)

    # -- sampling ----------------------------------------------------------------

    def update(
        self, t: float, finger: FingerId | None, camera_xy: tuple[float, float] | None, two_hands: bool = False
    ) -> AccuracySample | None:
        """Feed the single pointing fingertip (or None). Returns a sample when one is recorded."""
        point = self.current
        if point is None or self.started_at is None or self._target_started_at is None or t < self._target_started_at:
            return None
        if t - self._target_started_at >= self.config.timeout_s:
            return self._record(t, point, None, None)
        if camera_xy is None:
            self._diag["two_hands" if two_hands else "no_index"] += 1
            self._reset_window()
            return None
        p = self._iso(camera_xy)
        if self._needs_move:
            assert self._last_sample_iso is not None
            if np.linalg.norm(p - self._last_sample_iso) < self.config.min_move:
                self._diag["did_not_move"] += 1
                return None
            self._needs_move = False
        if self._anchor is None or finger != self._window_finger or np.linalg.norm(p - self._anchor) > self.config.stable_radius:
            if self._anchor is not None:
                self._diag["unstable"] += 1
            self._anchor, self._anchor_t, self._window, self._window_finger = p, t, [camera_xy], finger
            return None
        self._window.append(camera_xy)
        assert self._anchor_t is not None
        settled = t - self._target_started_at >= self.config.min_settle_s
        if settled and t - self._anchor_t >= self.config.stable_s:
            mean = np.mean(np.asarray(self._window), axis=0)
            return self._record(t, point, finger, (float(mean[0]), float(mean[1])))
        return None

    def capture_now(self, t: float, finger: FingerId | None, camera_xy: tuple[float, float] | None) -> AccuracySample | None:
        point = self.current
        if point is None or camera_xy is None or self.preparing(t) or self.started_at is None:
            return None
        return self._record(t, point, finger, camera_xy)

    def _record(self, t: float, point: ProbePoint, finger: FingerId | None,
                camera_xy: tuple[float, float] | None) -> AccuracySample:
        piano_xy = detected = None
        if camera_xy is not None:
            piano_xy, detected, _ = self.mapper.map_point(camera_xy)
        had_position = piano_xy is not None
        category = categorize(point.expected, detected, had_position)
        distance = None
        if point.expected is not None and piano_xy is not None:
            distance = point.expected.distance_to_center(*piano_xy)
        assert self.started_at is not None
        diag = self._diag
        miss_reason = None
        if category == MISS:
            miss_reason = max(diag, key=lambda k: diag[k]) if any(diag.values()) else "no_index"
        sample = AccuracySample(
            sample_id=point.sample_id,
            kind=point.kind,
            expected_key=None if point.expected is None else point.expected.name,
            detected_key=None if detected is None else detected.name,
            correct=category in (EXACT, OUTSIDE_CORRECT),
            error_category=category,
            miss_reason=miss_reason,
            frames_no_index=diag["no_index"],
            frames_two_hands=diag["two_hands"],
            frames_waiting_move=diag["did_not_move"],
            stability_resets=diag["unstable"],
            finger=finger,
            timestamp=datetime.now().isoformat(timespec="milliseconds"),
            t=t - self.started_at,
            target_piano_xy=point.piano_xy,
            piano_xy=piano_xy,
            camera_xy=camera_xy,
            distance_to_expected_center=distance,
        )
        self.samples.append(sample)
        self._target_started_at = t
        self._diag = _empty_diag()
        self._reset_window()
        self._require_move_after(camera_xy)
        return sample

    def _require_move_after(self, camera_xy: tuple[float, float] | None) -> None:
        """The next sample needs the finger to move, unless the next target is about where it is."""
        self._needs_move = False
        self._last_sample_iso = None
        nxt = self.current
        calibration = self.mapper.calibration
        if camera_xy is None or nxt is None or calibration is None:
            return
        here = self._iso(camera_xy)
        tx, ty = (float(v) for v in calibration.to_camera([nxt.piano_xy])[0])
        if np.linalg.norm(self._iso((tx, ty)) - here) >= 2 * self.config.min_move:
            self._needs_move = True
            self._last_sample_iso = here

    def _reset_window(self) -> None:
        self._anchor = self._anchor_t = self._window_finger = None
        self._window = []

    # -- results -----------------------------------------------------------------

    def report(self) -> AccuracyReport:
        inside = [s for s in self.samples if s.expected_key is not None]
        outside = [s for s in self.samples if s.expected_key is None]

        def count(group: list[AccuracySample], category: str) -> int:
            return sum(1 for s in group if s.error_category == category)

        return AccuracyReport(
            inside=len(inside),
            exact=count(inside, EXACT),
            adjacent=count(inside, ADJACENT),
            far=count(inside, FAR),
            miss_inside=count(inside, MISS),
            outside=len(outside),
            outside_correct=count(outside, OUTSIDE_CORRECT),
            outside_false_positive=count(outside, OUTSIDE_FALSE_POSITIVE),
            miss_outside=count(outside, MISS),
        )

    def write_csv(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "sample_id", "kind", "expected_key", "detected_key", "correct", "error_category", "miss_reason",
                "frames_no_index", "frames_two_hands", "frames_waiting_move", "stability_resets", "finger",
                "timestamp", "t_s", "target_piano_x", "target_piano_y", "piano_x", "piano_y", "camera_x", "camera_y",
                "distance_to_expected_center",
            ])
            for s in self.samples:
                def opt(v):
                    return "" if v is None else f"{v:.4f}"

                writer.writerow([
                    s.sample_id, s.kind, s.expected_key or "", s.detected_key or "", s.correct, s.error_category,
                    s.miss_reason or "", s.frames_no_index, s.frames_two_hands, s.frames_waiting_move,
                    s.stability_resets, "" if s.finger is None else s.finger.name, s.timestamp, f"{s.t:.3f}",
                    f"{s.target_piano_xy[0]:.4f}", f"{s.target_piano_xy[1]:.4f}",
                    opt(None if s.piano_xy is None else s.piano_xy[0]), opt(None if s.piano_xy is None else s.piano_xy[1]),
                    opt(None if s.camera_xy is None else s.camera_xy[0]), opt(None if s.camera_xy is None else s.camera_xy[1]),
                    opt(s.distance_to_expected_center),
                ])
