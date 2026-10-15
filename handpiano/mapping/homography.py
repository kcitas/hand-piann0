"""2-D projective transformation (homography) from four point correspondences. NumPy only.

A homography maps any convex quadrilateral onto any other, which is what a
camera does to a flat rectangle seen at an angle (perspective). An affine
transform cannot: it keeps parallel lines parallel, so it cannot undo
perspective. The mapping is exact only for points ON the plane defined by the
four reference points; a point above or below that plane (a fingertip in the
air, for example) is projected with a parallax error that grows with its
distance from the plane. Nothing here is 3-D.
"""

from __future__ import annotations

import numpy as np


class DegenerateHomographyError(ValueError):
    """The four points do not define a valid projective transformation."""


def homography_from_points(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """3×3 matrix H with H · [x, y, 1]ᵀ ∝ [u, v, 1]ᵀ for the 4 pairs (src → dst). H[2, 2] = 1."""
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if src.shape != (4, 2) or dst.shape != (4, 2):
        raise DegenerateHomographyError("exactly four 2-D points are required on each side")
    if not (np.all(np.isfinite(src)) and np.all(np.isfinite(dst))):
        raise DegenerateHomographyError("points must be finite")
    a = np.zeros((8, 8))
    b = np.zeros(8)
    for i, ((x, y), (u, v)) in enumerate(zip(src, dst, strict=True)):
        a[2 * i] = (x, y, 1, 0, 0, 0, -u * x, -u * y)
        a[2 * i + 1] = (0, 0, 0, x, y, 1, -v * x, -v * y)
        b[2 * i], b[2 * i + 1] = u, v
    if np.linalg.cond(a) > 1e10:
        raise DegenerateHomographyError("points are (nearly) collinear or repeated")
    h = np.linalg.solve(a, b)
    return np.append(h, 1.0).reshape(3, 3)


def apply_homography(h: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Maps (n, 2) points. Points that map to infinity come back as NaN."""
    pts = np.atleast_2d(np.asarray(points, dtype=np.float64))
    homogeneous = np.hstack([pts, np.ones((len(pts), 1))]) @ h.T
    w = homogeneous[:, 2:3]
    with np.errstate(divide="ignore", invalid="ignore"):
        out = homogeneous[:, :2] / w
    out[np.abs(w[:, 0]) < 1e-12] = np.nan
    return out
