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
* ``bend``   how much the finger is bent at its middle joint (PIP), in the image:
  (180° − angle knuckle–PIP–tip) / 90°. 0 = straight, 1 = bent 90°. Recorded for
  the finger-note mode (hands facing the camera, a note = bending a finger) to compare
  with ``flex``, which that mode uses: a finger bending towards the camera keeps its
  joints nearly in line in the image, so this angle changes late.
  Not part of ``FEATURES`` (the candidates benchmarked for the drawn piano).

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
        "bend": _bend(landmarks, finger, aspect),
    }


def _bend(landmarks: np.ndarray, finger: Finger, aspect: float) -> float:
    mcp, pip, _, tip = (np.array([landmarks[j][0], landmarks[j][1] * aspect]) for j in finger.joints)
    u, v = mcp - pip, tip - pip
    if float(np.linalg.norm(u)) < 1e-9 or float(np.linalg.norm(v)) < 1e-9:
        return 0.0
    angle = math.degrees(math.atan2(abs(float(u[0] * v[1] - u[1] * v[0])), float(np.dot(u, v))))
    return (180.0 - angle) / 90.0
