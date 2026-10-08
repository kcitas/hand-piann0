"""Hand landmark model (MediaPipe's 21-point topology) and finger identities."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import Any

import numpy as np

NUM_LANDMARKS = 21


class Landmark(IntEnum):
    WRIST = 0
    THUMB_CMC = 1
    THUMB_MCP = 2
    THUMB_IP = 3
    THUMB_TIP = 4
    INDEX_MCP = 5
    INDEX_PIP = 6
    INDEX_DIP = 7
    INDEX_TIP = 8
    MIDDLE_MCP = 9
    MIDDLE_PIP = 10
    MIDDLE_DIP = 11
    MIDDLE_TIP = 12
    RING_MCP = 13
    RING_PIP = 14
    RING_DIP = 15
    RING_TIP = 16
    PINKY_MCP = 17
    PINKY_PIP = 18
    PINKY_DIP = 19
    PINKY_TIP = 20


class Finger(Enum):
    THUMB = (Landmark.THUMB_CMC, Landmark.THUMB_TIP)
    INDEX = (Landmark.INDEX_MCP, Landmark.INDEX_TIP)
    MIDDLE = (Landmark.MIDDLE_MCP, Landmark.MIDDLE_TIP)
    RING = (Landmark.RING_MCP, Landmark.RING_TIP)
    PINKY = (Landmark.PINKY_MCP, Landmark.PINKY_TIP)

    @property
    def base(self) -> Landmark:
        return self.value[0]

    @property
    def tip(self) -> Landmark:
        return self.value[1]

    @property
    def joints(self) -> tuple[Landmark, ...]:
        """Base → tip, 4 landmarks."""
        return tuple(Landmark(i) for i in range(self.base, self.tip + 1))


class Handedness(Enum):
    """The user's physical hand (estimated), never MediaPipe's raw label.

    See ``observations_from_result`` for how the raw label is converted.
    """

    LEFT = "Left"
    RIGHT = "Right"

    @property
    def opposite(self) -> Handedness:
        return Handedness.RIGHT if self is Handedness.LEFT else Handedness.LEFT

    @property
    def short(self) -> str:
        return "L" if self is Handedness.LEFT else "R"


class FingerId(Enum):
    """The eight fingers used to play. Thumbs are tracked as landmarks but not as playing fingers."""

    LEFT_INDEX = (Handedness.LEFT, Finger.INDEX)
    LEFT_MIDDLE = (Handedness.LEFT, Finger.MIDDLE)
    LEFT_RING = (Handedness.LEFT, Finger.RING)
    LEFT_PINKY = (Handedness.LEFT, Finger.PINKY)
    RIGHT_INDEX = (Handedness.RIGHT, Finger.INDEX)
    RIGHT_MIDDLE = (Handedness.RIGHT, Finger.MIDDLE)
    RIGHT_RING = (Handedness.RIGHT, Finger.RING)
    RIGHT_PINKY = (Handedness.RIGHT, Finger.PINKY)

    @property
    def hand(self) -> Handedness:
        return self.value[0]

    @property
    def finger(self) -> Finger:
        return self.value[1]

    @property
    def short(self) -> str:
        return f"{self.hand.short}-{self.finger.name[:3]}"


PLAYING_FINGERS: tuple[FingerId, ...] = tuple(FingerId)

HAND_CONNECTIONS: tuple[tuple[int, int], ...] = (
    *((Landmark.WRIST, f.base) for f in Finger),
    *((a, b) for f in Finger for a, b in zip(f.joints, f.joints[1:], strict=False)),
    (Landmark.INDEX_MCP, Landmark.MIDDLE_MCP),
    (Landmark.MIDDLE_MCP, Landmark.RING_MCP),
    (Landmark.RING_MCP, Landmark.PINKY_MCP),
)


@dataclass(frozen=True, slots=True)
class HandObservation:
    """One hand as reported by the detector for one frame (unsmoothed).

    ``landmarks``: (21, 3) float array. x, y normalized to [0, 1] of the image
    width/height (can slightly exceed it near borders); z is relative depth with
    the wrist as origin, roughly in the same scale as x. Smaller z = closer to camera.
    """

    # The user's physical hand as estimated from this frame alone (already converted
    # from MediaPipe's label; no temporal identity applied yet).
    handedness: Handedness
    # MediaPipe's handedness classification score: how sure it is about left/right
    # on this frame. It is the only per-hand score the Tasks API exposes.
    score: float
    landmarks: np.ndarray

    @property
    def wrist(self) -> np.ndarray:
        return self.landmarks[Landmark.WRIST, :2]


def observations_from_result(result: Any, mirrored_input: bool) -> list[HandObservation]:
    """Converts a MediaPipe ``HandLandmarkerResult`` into plain observations.

    MediaPipe Tasks labels the hand by how it LOOKS in the image it receives. A
    mirrored image of a right hand looks like a left hand, so when HandPiano feeds
    mirrored frames (selfie view, the default) the label is the opposite of the
    user's physical hand and is inverted here. With unmirrored frames it is used as is.

    Validated on 2026-10-08 with the real camera in mirrored view: both hands
    raised, palms to the camera, uncrossed — the hand on the image's left (the
    user's left) came back as "Right" (score 0.95) and the other as "Left".
    """
    observations: list[HandObservation] = []
    for landmarks, categories in zip(result.hand_landmarks, result.handedness, strict=False):
        if len(landmarks) != NUM_LANDMARKS or not categories:
            continue
        best = max(categories, key=lambda c: c.score)
        try:
            label = Handedness(best.category_name)
        except ValueError:
            continue
        handedness = label.opposite if mirrored_input else label
        points = np.array([(p.x, p.y, p.z) for p in landmarks], dtype=np.float32)
        observations.append(HandObservation(handedness=handedness, score=float(best.score), landmarks=points))
    return observations
