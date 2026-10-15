"""Interactive capture of the four keyboard corners with a fingertip.

The user points an index fingertip at each corner in turn. A corner is
captured when the fingertip stays within ``dwell_radius`` for ``dwell_s``
seconds (its average position over that time is used), or immediately on an
explicit request (Space key). After a capture, the next dwell only starts once
the fingertip has moved at least ``min_move`` away from the last corner, so one
long pause cannot capture two corners at the same place. A corner closer than
``MIN_CORNER_DISTANCE`` to an existing one is rejected immediately (with a
message) instead of failing only after the fourth corner.

Distances are measured in "frame-width units": x as is, y multiplied by
height/width, so a radius means the same on screen horizontally and vertically.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from handpiano.mapping.calibration import CORNER_NAMES, MIN_CORNER_DISTANCE


@dataclass(frozen=True, slots=True)
class SessionConfig:
    dwell_s: float = 1.2
    dwell_radius: float = 0.012  # ≈ 15 px on a 1280-wide frame
    min_move: float = 0.05


@dataclass(slots=True)
class CalibrationSession:
    config: SessionConfig = field(default_factory=SessionConfig)
    # Frame height / width, to measure distances isotropically.
    aspect: float = 9 / 16
    points: list[tuple[float, float]] = field(default_factory=list)
    _anchor: np.ndarray | None = None
    _anchor_t: float | None = None
    _samples: list[np.ndarray] = field(default_factory=list)
    _armed: bool = True
    # Why the last capture attempt was refused (shown to the user); None after a success.
    rejection: str | None = None

    @property
    def done(self) -> bool:
        return len(self.points) == 4

    @property
    def step(self) -> int:
        """1-based index of the corner being captured (5 when done)."""
        return len(self.points) + 1

    @property
    def corner_name(self) -> str | None:
        return None if self.done else CORNER_NAMES[len(self.points)]

    @property
    def waiting_to_move_away(self) -> bool:
        return not self._armed

    def _iso(self, xy) -> np.ndarray:
        return np.array([xy[0], xy[1] * self.aspect], dtype=np.float64)

    def dwell_progress(self, t: float) -> float:
        if self._anchor_t is None or not self._armed:
            return 0.0
        return min(1.0, (t - self._anchor_t) / self.config.dwell_s)

    def update(self, t: float, tip: tuple[float, float] | None) -> tuple[float, float] | None:
        """Feed the current fingertip (normalized camera coords). Returns a corner when one is captured."""
        if self.done:
            return None
        if tip is None:
            self._reset_dwell()
            return None
        p = self._iso(tip)
        if not self._armed:
            if not self.points or all(
                np.linalg.norm(p - self._iso(q)) >= self.config.min_move for q in self.points
            ):
                self._armed = True
            else:
                return None
        if self._anchor is None or np.linalg.norm(p - self._anchor) > self.config.dwell_radius:
            self._anchor, self._anchor_t, self._samples = p, t, [np.asarray(tip, dtype=np.float64)]
            return None
        self._samples.append(np.asarray(tip, dtype=np.float64))
        assert self._anchor_t is not None
        if t - self._anchor_t >= self.config.dwell_s:
            mean = np.mean(self._samples, axis=0)
            return self._capture((float(mean[0]), float(mean[1])))
        return None

    def capture_now(self, tip: tuple[float, float] | None) -> tuple[float, float] | None:
        if self.done or tip is None:
            return None
        return self._capture((float(tip[0]), float(tip[1])))

    def undo(self) -> None:
        if self.points:
            self.points.pop()
        self._armed = True
        self._reset_dwell()

    def _capture(self, point: tuple[float, float]) -> tuple[float, float] | None:
        # Same metric as validate_corners, so an accepted corner never fails validation for this reason.
        for i, existing in enumerate(self.points, start=1):
            if np.hypot(point[0] - existing[0], point[1] - existing[1]) < MIN_CORNER_DISTANCE:
                self.rejection = f"Demasiado cerca de la esquina {i}. Lleva el dedo a la esquina {self.corner_name}."
                self._armed = False
                self._reset_dwell()
                return None
        self.rejection = None
        self.points.append(point)
        self._armed = False
        self._reset_dwell()
        return point

    def _reset_dwell(self) -> None:
        self._anchor = self._anchor_t = None
        self._samples = []
