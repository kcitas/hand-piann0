"""Hand-relative press signals for one finger.

Every signal is measured from the finger's own knuckle (MCP) and divided by
the hand's size, so moving or scaling the whole hand does not change it, and it
grows when the finger performs a press. Which one actually tracks a press in a
given camera setup is an empirical question (see ``scripts/analyze_trace.py``);
none is a physical measurement.

* ``tip_y``  image-vertical offset of the tip below its knuckle. A tap moves the
  tip down in the image both with palms down (camera in front, above) and with
  fingers up facing the camera (the finger folds).
* ``flex``   negative projected finger length: a finger that bends towards the
  camera or the surface looks shorter.
* ``tip_z``  MediaPipe's relative depth of the tip with respect to the knuckle,
  negated so that "towards the camera" is positive. Relative only, unitless.

Coordinates are made isotropic first ("frame-width units": y multiplied by
height/width) so that x, y and z share a scale.
"""

from __future__ import annotations

import math

import numpy as np

from handpiano.tracking.landmarks import Finger, Landmark

FEATURES = ("tip_y", "flex", "tip_z")


def hand_scale(landmarks: np.ndarray, aspect: float) -> float:
    """Wrist → middle-finger knuckle distance in frame-width units (≈ palm length)."""
    w = landmarks[Landmark.WRIST]
    m = landmarks[Landmark.MIDDLE_MCP]
    return math.hypot(float(m[0] - w[0]), float(m[1] - w[1]) * aspect)


def press_features(landmarks: np.ndarray, finger: Finger, aspect: float) -> dict[str, float] | None:
    """All candidate press signals for ``finger``, or None if the hand size is degenerate."""
    scale = hand_scale(landmarks, aspect)
    if scale < 1e-6:
        return None
    tip = landmarks[finger.tip]
    mcp = landmarks[finger.base]
    dx = float(tip[0] - mcp[0])
    dy = float(tip[1] - mcp[1]) * aspect
    dz = float(tip[2] - mcp[2])
    return {
        "tip_y": dy / scale,
        "flex": -math.hypot(dx, dy) / scale,
        "tip_z": -dz / scale,
    }
