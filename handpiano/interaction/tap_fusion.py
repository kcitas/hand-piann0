"""Microphone + camera: the microphone says WHEN a finger hit the table, the camera says WHICH finger and key.

For every tap onset at ``t`` (from ``MicTapInput``), once camera frames up to
``t + decide_after_s`` exist, each tracked finger is scored on the detector's own
signals (``FingerPress.depth`` = press signal against the finger's rest, and
``velocity`` = its rate of change, hand sizes per second):

    gain  = max depth in [t − 0.06, t + 0.08]  −  min depth in [t − 0.30, t − 0.06]
    speed = max velocity in [t − 0.20, t + 0.08]

A finger is a candidate if it was tracked, had a key under it and ``gain`` ≥
``min_gain``. The best score (gain + speed × speed_weight) wins; fingers of the
same tap scoring ≥ ``chord_fraction`` of the best are played too (chords). The
key is the one under the finger BEFORE the motion (at t − 0.12 s), like the
detector's latched key. An onset with no candidate plays nothing and is reported.

Release: the finger is held until its depth falls back below
``rest + release_fraction × gain``, or it is no longer tracked, or ``max_hold_s``.

The camera-only detector keeps running unchanged (its thresholds are not used
here); in microphone mode its events are simply not used for sound.

All parameters are PROVISIONAL (physics and first principles), to be measured on
recordings: they are fusion parameters, not the detector's thresholds.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

from handpiano.audio.tap_input import TapOnset
from handpiano.instrument.keyboard import PianoKey
from handpiano.interaction.finger_presses import PressFrame
from handpiano.interaction.press_detector import PressEvent
from handpiano.tracking.landmarks import PLAYING_FINGERS, FingerId


@dataclass(frozen=True, slots=True)
class FusionConfig:
    decide_after_s: float = 0.08  # wait for the camera frames that show the end of the downstroke
    min_gain: float = 0.05  # same units as the detector's depth (hand sizes)
    speed_weight: float = 0.02
    chord_fraction: float = 0.6
    max_fingers: int = 4
    release_fraction: float = 0.4
    max_hold_s: float = 2.0
    history_s: float = 1.0


@dataclass(frozen=True, slots=True)
class _Sample:
    t: float
    depth: float | None
    velocity: float | None
    key: PianoKey | None
    tracked: bool


@dataclass(slots=True)
class _Held:
    key: PianoKey
    rest: float
    gain: float
    since: float


@dataclass
class FusionStats:
    onsets: int = 0
    played: int = 0  # fingers played (a chord counts each finger)
    unmatched: int = 0  # taps with no moving finger under the camera
    last_unmatched: list[float] = field(default_factory=list)


class TapFusion:
    def __init__(self, config: FusionConfig | None = None) -> None:
        self.config = config or FusionConfig()
        self._history: dict[FingerId, deque[_Sample]] = {f: deque() for f in PLAYING_FINGERS}
        self._pending: list[TapOnset] = []
        self._held: dict[FingerId, _Held] = {}
        self.stats = FusionStats()

    # -- inputs ---------------------------------------------------------------------------

    def add_onsets(self, onsets: list[TapOnset]) -> None:
        self._pending.extend(onsets)
        self.stats.onsets += len(onsets)

    def on_frame(self, t: float, presses: PressFrame) -> list[PressEvent]:
        """Called once per camera frame (tracking thread). Returns press/release events to play."""
        cfg = self.config
        for finger in PLAYING_FINGERS:
            fp = presses.fingers[finger]
            sig = presses.signals[finger]
            history = self._history[finger]
            history.append(_Sample(t, fp.depth, fp.velocity, sig.stable_key, sig.tracked))
            while history and t - history[0].t > cfg.history_s:
                history.popleft()
        events = self._releases(t)
        ready = [o for o in self._pending if t >= o.t + cfg.decide_after_s]
        self._pending = [o for o in self._pending if t < o.t + cfg.decide_after_s]
        for onset in ready:
            events.extend(self._decide(onset))
        return events

    # -- decisions ------------------------------------------------------------------------

    def _window(self, finger: FingerId, a: float, b: float) -> list[_Sample]:
        return [s for s in self._history[finger] if a <= s.t <= b]

    def _decide(self, onset: TapOnset) -> list[PressEvent]:
        cfg = self.config
        t = onset.t
        scored: list[tuple[float, FingerId, PianoKey, float, float]] = []
        for finger in PLAYING_FINGERS:
            if finger in self._held:
                continue
            after = [s.depth for s in self._window(finger, t - 0.06, t + 0.08) if s.tracked and s.depth is not None]
            before = [s.depth for s in self._window(finger, t - 0.30, t - 0.06) if s.tracked and s.depth is not None]
            if not after or not before:
                continue
            rest = min(before)
            gain = max(after) - rest
            speeds = [s.velocity for s in self._window(finger, t - 0.20, t + 0.08)
                      if s.tracked and s.velocity is not None]
            speed = max(speeds) if speeds else 0.0
            key = self._key_before(finger, t - 0.12)
            if key is None or gain < cfg.min_gain:
                continue
            scored.append((gain + cfg.speed_weight * max(0.0, speed), finger, key, rest, gain))
        if not scored:
            self.stats.unmatched += 1
            self.stats.last_unmatched = (self.stats.last_unmatched + [t])[-20:]
            return []
        scored.sort(key=lambda s: -s[0])
        best = scored[0][0]
        chosen = [s for s in scored if s[0] >= cfg.chord_fraction * best][: cfg.max_fingers]
        events = []
        for _, finger, key, rest, gain in chosen:
            self._held[finger] = _Held(key, rest, gain, t)
            events.append(PressEvent("press", finger, t, key, "mic", gain, None))
            self.stats.played += 1
        return events

    def _key_before(self, finger: FingerId, t: float) -> PianoKey | None:
        candidates = [s for s in self._history[finger] if s.t <= t and s.tracked and s.key is not None]
        if candidates:
            return candidates[-1].key
        later = [s for s in self._history[finger] if s.tracked and s.key is not None]
        return later[0].key if later else None

    def _releases(self, t: float) -> list[PressEvent]:
        cfg = self.config
        events = []
        for finger, held in list(self._held.items()):
            last = self._history[finger][-1] if self._history[finger] else None
            lifted = last is not None and last.tracked and last.depth is not None and \
                last.depth < held.rest + cfg.release_fraction * held.gain
            lost = last is None or not last.tracked
            if lifted or lost or t - held.since > cfg.max_hold_s:
                reason = "lifted" if lifted else "tracking_lost" if lost else "max_hold"
                del self._held[finger]
                events.append(PressEvent("release", finger, t, held.key, reason, None, None))
        return events

    # -- state ----------------------------------------------------------------------------

    @property
    def held_keys(self) -> dict[int, tuple[FingerId, ...]]:
        """MIDI → fingers holding it (the same shape as ``PressFrame.pressed_keys``)."""
        out: dict[int, list[FingerId]] = {}
        for finger, held in self._held.items():
            out.setdefault(held.key.midi, []).append(finger)
        return {m: tuple(f) for m, f in out.items()}

    def describe_unmatched(self) -> Any:
        return list(self.stats.last_unmatched)
