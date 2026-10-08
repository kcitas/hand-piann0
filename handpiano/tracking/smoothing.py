"""Landmark smoothing.

The One Euro Filter (Casiez, Roussel & Vogel, CHI 2012) is a low-pass filter
whose cutoff frequency rises with speed: when the finger is still, a low cutoff
removes jitter; when it moves fast, the cutoff rises and lag shrinks. That is
the trade-off a musical controller needs (stable hover, responsive strike).

Units: positions are normalized image coordinates, so speeds are
"image widths per second". Parameters live in ``SmoothingConfig``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import numpy as np


class SmoothingKind(Enum):
    NONE = "none"
    ONE_EURO = "one_euro"


@dataclass(frozen=True, slots=True)
class SmoothingConfig:
    kind: SmoothingKind = SmoothingKind.ONE_EURO
    # Cutoff (Hz) at rest. Lower = steadier but laggier when slow.
    min_cutoff: float = 1.5
    # How fast the cutoff rises with speed. Higher = less lag on fast moves.
    beta: float = 10.0
    # Cutoff (Hz) for the speed estimate itself.
    d_cutoff: float = 1.0
    # A gap longer than this (e.g. hand lost) restarts the filter instead of
    # interpolating across stale data.
    max_gap_s: float = 0.25


def _alpha(cutoff: np.ndarray | float, dt: float) -> np.ndarray | float:
    tau = 1.0 / (2.0 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


class OneEuroFilter:
    """Vectorized One Euro Filter: filters every element of an array independently."""

    def __init__(self, config: SmoothingConfig) -> None:
        self.config = config
        self._x: np.ndarray | None = None
        self._dx: np.ndarray | None = None
        self._t: float | None = None

    def reset(self) -> None:
        self._x = self._dx = self._t = None

    def __call__(self, x: np.ndarray, t: float) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        cfg = self.config
        if cfg.kind is SmoothingKind.NONE:
            return x.copy()

        if self._x is None or self._t is None or self._dx is None or x.shape != self._x.shape:
            return self._restart(x, t)
        dt = t - self._t
        if dt <= 0:
            return self._x.copy()
        if dt > cfg.max_gap_s:
            return self._restart(x, t)

        dx = (x - self._x) / dt
        a_d = _alpha(cfg.d_cutoff, dt)
        dx_hat = a_d * dx + (1 - a_d) * self._dx
        cutoff = cfg.min_cutoff + cfg.beta * np.abs(dx_hat)
        a = _alpha(cutoff, dt)
        x_hat = a * x + (1 - a) * self._x

        self._x, self._dx, self._t = x_hat, dx_hat, t
        return x_hat.copy()

    def _restart(self, x: np.ndarray, t: float) -> np.ndarray:
        self._x, self._dx, self._t = x.copy(), np.zeros_like(x), t
        return x.copy()
