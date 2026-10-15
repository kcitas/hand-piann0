"""Candidate press features computed from the 21 landmarks of one hand (research only).

The production detector uses ``press_features`` (tip_y by default); this module
adds features that need the whole hand, so they can be benchmarked offline on
v4 traces (which store the landmarks) and on PianoVAM. Nothing here feeds the
detector.

Coordinates
-----------
* image landmarks: x, y in [0, 1] of the image width/height, z MediaPipe's relative
  depth (wrist origin, ~x scale). Made isotropic first: (x, y·height/width, z).
* world landmarks: metres, MediaPipe's ``hand_world_landmarks`` (origin near the
  hand's centre). Features from them carry the ``_w`` suffix.

Every distance is divided by the palm length (wrist → middle MCP) of the same
coordinates, so features are unitless and hand-size invariant; angles are
degrees and invariant to hand position, size and in-plane rotation.

Features (all oriented so that a press makes them GROW)
-------------------------------------------------------
``tip_y``, ``flex``, ``tip_z``     as in ``press_features`` (identical values).
``mcp_bend``, ``pip_bend``, ``dip_bend``
    bend at each joint, 180° − interior angle (0 = straight). Image coordinates
    including z, which is noisy: the 2D projection dominates.
``palm_dist``   −(tip → palm centre distance): the tip approaching the palm.
``*_w``         the same bends / palm distance on world landmarks.
"""

from __future__ import annotations

import math

import numpy as np

from handpiano.interaction.press_features import press_features
from handpiano.tracking.landmarks import Finger, Landmark

PALM = (Landmark.WRIST, Landmark.INDEX_MCP, Landmark.MIDDLE_MCP, Landmark.RING_MCP, Landmark.PINKY_MCP)
IMAGE_FEATURES = ("tip_y", "flex", "tip_z", "mcp_bend", "pip_bend", "dip_bend", "palm_dist")
WORLD_FEATURES = ("mcp_bend_w", "pip_bend_w", "dip_bend_w", "palm_dist_w")
ALL_FEATURES = IMAGE_FEATURES + WORLD_FEATURES


def bend_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """180° − angle abc. NaN if a segment is degenerate."""
    u, v = a - b, c - b
    nu, nv = float(np.linalg.norm(u)), float(np.linalg.norm(v))
    if nu < 1e-9 or nv < 1e-9:
        return float("nan")
    # atan2 instead of acos: accurate near 0° and 180° (a nearly straight finger).
    return 180.0 - math.degrees(math.atan2(float(np.linalg.norm(np.cross(u, v))), float(np.dot(u, v))))


def _shape_features(points: np.ndarray, finger: Finger, suffix: str) -> dict[str, float]:
    """Bends and palm distance on isotropic 3D points (21, 3)."""
    scale = float(np.linalg.norm(points[Landmark.MIDDLE_MCP] - points[Landmark.WRIST]))
    if not np.isfinite(scale) or scale < 1e-9:
        return {f"{n}{suffix}": float("nan") for n in ("mcp_bend", "pip_bend", "dip_bend", "palm_dist")}
    mcp, pip, dip, tip = (points[j] for j in finger.joints)
    palm = points[list(PALM)].mean(axis=0)
    return {
        f"mcp_bend{suffix}": bend_deg(points[Landmark.WRIST], mcp, pip),
        f"pip_bend{suffix}": bend_deg(mcp, pip, dip),
        f"dip_bend{suffix}": bend_deg(pip, dip, tip),
        f"palm_dist{suffix}": -float(np.linalg.norm(tip - palm)) / scale,
    }


def landmark_features(landmarks: np.ndarray | None, finger: Finger, aspect: float,
                      world: np.ndarray | None = None) -> dict[str, float]:
    """Every feature in ``ALL_FEATURES`` (NaN where not computable).

    ``landmarks`` / ``world``: (21, 3) arrays or None; non-finite values give NaN.
    """
    out = {f: float("nan") for f in ALL_FEATURES}
    if landmarks is not None and np.all(np.isfinite(landmarks)):
        base = press_features(landmarks, finger, aspect)
        if base is not None:
            out.update(base)
            iso = np.asarray(landmarks, dtype=np.float64) * np.array([1.0, aspect, 1.0])
            out.update(_shape_features(iso, finger, ""))
    if world is not None and np.all(np.isfinite(world)):
        out.update(_shape_features(np.asarray(world, dtype=np.float64), finger, "_w"))
    return out
