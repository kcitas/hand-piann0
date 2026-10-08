"""Test doubles that stand in for the physical camera and MediaPipe."""

from __future__ import annotations

import time

import numpy as np

from handpiano.tracking.landmarks import NUM_LANDMARKS, Handedness, HandObservation


class FakeCapture:
    """Mimics cv2.VideoCapture. Delivers fixed-size BGR frames regardless of what is requested,
    like a real webcam that ignores unsupported modes."""

    def __init__(
        self,
        opened: bool = True,
        width: int = 640,
        height: int = 480,
        fps: float = 30.0,
        frames: int | None = None,
        accepts: frozenset[int] = frozenset(),
        delay_s: float = 0.0,
    ) -> None:
        self._opened = opened
        self._width = width
        self._height = height
        self._fps = fps
        self._remaining = frames
        self._accepts = accepts
        self._delay_s = delay_s
        self.set_calls: dict[int, float] = {}
        self.released = False

    def isOpened(self) -> bool:  # noqa: N802
        return self._opened

    def read(self):
        if self._delay_s:
            time.sleep(self._delay_s)  # emulate the camera's frame interval
        if self._remaining is not None:
            if self._remaining <= 0:
                return False, None
            self._remaining -= 1
        frame = np.zeros((self._height, self._width, 3), np.uint8)
        frame[:, : self._width // 2, 0] = 255  # left half blue (BGR)
        return True, frame

    def set(self, prop: int, value: float) -> bool:
        self.set_calls[prop] = value
        return prop in self._accepts

    def get(self, prop: int) -> float:
        import cv2

        return self._fps if prop == cv2.CAP_PROP_FPS else 0.0

    def release(self) -> None:
        self.released = True


def make_hand(
    handedness: Handedness,
    center: tuple[float, float] = (0.5, 0.5),
    score: float = 0.95,
    scale: float = 0.1,
) -> HandObservation:
    """A geometrically plausible open hand: wrist at the bottom, fingers spreading upward."""
    cx, cy = center
    pts = np.zeros((NUM_LANDMARKS, 3), np.float32)
    pts[0] = (cx, cy + scale, 0)
    # Mirrored view: the thumb of the user's right hand points to the image left.
    side = -1 if handedness is Handedness.RIGHT else 1
    for j in range(4):  # thumb 1..4
        pts[1 + j] = (cx + side * scale * (0.3 + 0.2 * j), cy + scale * (0.6 - 0.15 * j), 0)
    for f in range(4):  # index, middle, ring, pinky: four columns spreading away from the thumb
        x = cx + side * scale * (0.15 - 0.3 * f)
        for j in range(4):  # mcp, pip, dip, tip going up
            pts[5 + 4 * f + j] = (x, cy + scale * (0.3 - 0.35 * j), -0.01 * j)
    return HandObservation(handedness=handedness, score=score, landmarks=pts)


def shifted(obs: HandObservation, dx: float = 0.0, dy: float = 0.0, label: Handedness | None = None):
    pts = obs.landmarks.copy()
    pts[:, 0] += dx
    pts[:, 1] += dy
    return HandObservation(handedness=label or obs.handedness, score=obs.score, landmarks=pts)
