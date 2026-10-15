"""Hand placement guide: is each hand where the piano can be played? Tells the player what to fix.

Measured on the recordings of 2026-10-13: with both hands inside the image and the fingertips on
the drawn keyboard (tocar_2) the camera tracked the hands 80–100 % of the time and detected 119
notes in 30 s; with the hands low (dedos_1: wrists at the bottom edge, fingertips below the
keyboard) it tracked them 27–50 % of the time and 19 notes were detected in 102 s. Placement
decided more than any detector setting, and nothing on screen said it was wrong.

Per hand, in normalized image coordinates (the processed, possibly mirrored, frame):
* not seen for ``lost_s``                                → "No veo la mano …"
* wrist below ``wrist_max_y`` (leaving the image)         → "Sube la mano …"
* fingertips (median of the 4 playing fingers) below the keyboard by more than ``margin`` → "Sube …"
* fingertips above it by more than ``margin``             → "Baja …"
* fingertips left/right of it by more than ``margin``     → "Mueve … hacia el centro"
A message must persist ``hold_s`` before it replaces the current one (no flicker).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from handpiano.tracking.finger_tracker import TrackingState, TrackStatus
from handpiano.tracking.landmarks import Finger, Handedness

HAND_NAME = {Handedness.LEFT: "izquierda", Handedness.RIGHT: "derecha"}
TIPS = [f.tip for f in (Finger.INDEX, Finger.MIDDLE, Finger.RING, Finger.PINKY)]


@dataclass(frozen=True, slots=True)
class PlacementConfig:
    wrist_max_y: float = 0.92
    margin: float = 0.06
    lost_s: float = 0.5
    hold_s: float = 0.4


@dataclass(frozen=True, slots=True)
class Placement:
    ok: bool
    messages: tuple[str, ...]

    @property
    def text(self) -> str:
        return "Manos bien colocadas" if self.ok else " · ".join(self.messages)


def keyboard_box(camera_points) -> tuple[float, float, float, float]:  # type: ignore[no-untyped-def]
    """(left, top, right, bottom) of the keyboard drawn on the image, from its 4 corners."""
    pts = np.asarray(camera_points, dtype=np.float64)
    return float(pts[:, 0].min()), float(pts[:, 1].min()), float(pts[:, 0].max()), float(pts[:, 1].max())


def assess_hand(hand: Handedness, landmarks: np.ndarray, box: tuple[float, float, float, float],
                config: PlacementConfig) -> str | None:
    left, top, right, bottom = box
    name = HAND_NAME[hand]
    wrist_y = float(landmarks[0, 1])
    tips = landmarks[TIPS, :2]
    tip_x, tip_y = float(np.median(tips[:, 0])), float(np.median(tips[:, 1]))
    if wrist_y > config.wrist_max_y:
        return f"Sube la mano {name}: la muñeca se sale de la imagen"
    if tip_y > bottom + config.margin:
        return f"Sube la mano {name}: las puntas están por debajo del piano"
    if tip_y < top - config.margin:
        return f"Baja la mano {name}: las puntas están por encima del piano"
    if tip_x < left - config.margin or tip_x > right + config.margin:
        return f"Mueve la mano {name} hacia el piano"
    return None


@dataclass
class PlacementGuide:
    config: PlacementConfig = field(default_factory=PlacementConfig)

    def __post_init__(self) -> None:
        self._last_seen: dict[Handedness, float] = {}
        self._shown = Placement(False, ("Coloca las dos manos sobre el piano",))
        self._candidate: Placement | None = None
        self._candidate_since = 0.0

    def update(self, t: float, state: TrackingState, box: tuple[float, float, float, float] | None) -> Placement:
        """``t``: seconds (any monotonic clock); ``box``: keyboard box, None if there is no keyboard."""
        if box is None:
            return self._show(t, Placement(False, ("Sin piano: calibra o usa el piano en pantalla",)))
        messages = []
        for hand in (Handedness.LEFT, Handedness.RIGHT):
            hs = state.hands[hand]
            if hs.status is TrackStatus.TRACKED and hs.landmarks is not None:
                self._last_seen[hand] = t
                problem = assess_hand(hand, hs.landmarks, box, self.config)
                if problem:
                    messages.append(problem)
            elif t - self._last_seen.get(hand, -1e9) > self.config.lost_s:
                messages.append(f"No veo la mano {HAND_NAME[hand]}: ponla entera dentro de la imagen")
        return self._show(t, Placement(not messages, tuple(messages)))

    def _show(self, t: float, current: Placement) -> Placement:
        if current == self._shown:
            self._candidate = None
            return self._shown
        if current != self._candidate:
            self._candidate, self._candidate_since = current, t
        elif t - self._candidate_since >= self.config.hold_s:
            self._shown, self._candidate = current, None
        return self._shown
