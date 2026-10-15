"""Key hysteresis: stop a fingertip near a boundary from flickering between two keys.

A finger keeps its current key while its point stays inside that key's
rectangle grown by ``margin`` (white-key widths); only then does it switch to
whatever ``key_at`` says. A point outside the keyboard always gives None: the
margin never extends the keyboard, so outside positions are rejected explicitly,
never snapped to the nearest key.

The margin is a trade-off: larger values stop more flicker but delay a real
move to the neighbouring key by the same distance. The default (0.08 white-key
widths ≈ 4 px with a 720 px wide keyboard) is about 2× the smoothed fingertip
jitter measured on this machine (≈ 1.8 px median).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from handpiano.instrument.keyboard import KeyboardLayout, PianoKey
from handpiano.tracking.landmarks import FingerId


@dataclass(slots=True)
class StableKeys:
    layout: KeyboardLayout
    margin: float = 0.08
    _current: dict[FingerId, PianoKey | None] = field(default_factory=dict)

    def reset(self, finger: FingerId) -> None:
        self._current.pop(finger, None)

    def update(self, finger: FingerId, piano_xy: tuple[float, float] | None) -> PianoKey | None:
        if piano_xy is None:
            self._current.pop(finger, None)
            return None
        x, y = piano_xy
        if not self.layout.contains(x, y):
            self._current[finger] = None
            return None
        raw = self.layout.key_at(x, y)
        current = self._current.get(finger)
        if current is not None and raw is not None and raw.midi != current.midi:
            m = self.margin
            if current.x - m <= x < current.x + current.width + m and current.y - m <= y < current.y + current.height + m:
                return current  # still within the tolerance band of the key it was on
        self._current[finger] = raw
        return raw
