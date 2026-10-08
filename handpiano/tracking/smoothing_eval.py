"""Reproducible, synthetic evaluation of the landmark smoother.

Used by the tests and by ``scripts/evaluate_smoothing.py``. Signals are 1-D in
normalized image units (1.0 = full image width) sampled at a fixed frame rate,
so results are exactly reproducible. They characterize the filter, not the
whole system: real landmark noise is not white and real hands do not move in
perfect ramps.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from handpiano.tracking.smoothing import OneEuroFilter, SmoothingConfig


def run_one_euro(config: SmoothingConfig, signal: np.ndarray, fps: float) -> np.ndarray:
    f = OneEuroFilter(config)
    return np.array([f(np.array([v]), i / fps)[0] for i, v in enumerate(signal)])


def run_ema(alpha: float, signal: np.ndarray) -> np.ndarray:
    out = np.empty_like(signal)
    y = signal[0]
    for i, v in enumerate(signal):
        y = alpha * v + (1 - alpha) * y
        out[i] = y
    return out


def still_noise(noise_std: float, frames: int = 600, seed: int = 0) -> np.ndarray:
    return 0.5 + np.random.default_rng(seed).normal(0, noise_std, frames)


def settle_frames(response: np.ndarray, start: int, target: float, tolerance: float = 0.1) -> int:
    """Frames after ``start`` until the response is within ``tolerance`` × step of ``target``."""
    step = abs(target - response[start - 1])
    for k in range(start, len(response)):
        if abs(target - response[k]) <= tolerance * step:
            return k - start
    return len(response) - start


def steady_ramp_lag_ms(output: np.ndarray, speed: float, fps: float, settle: int) -> float:
    """Lag (ms) of the filtered ramp behind the true ramp, after ``settle`` frames."""
    t = np.arange(len(output)) / fps
    truth = 0.2 + speed * t
    return float(np.mean(truth[settle:] - output[settle:]) / speed * 1000.0)


def matched_ema_alpha(target_std: float, noisy: np.ndarray, skip: int = 30) -> float:
    """Largest EMA alpha whose output noise is ≤ ``target_std`` (as steady at rest as the reference)."""
    for alpha in np.linspace(0.95, 0.01, 400):
        if np.std(run_ema(alpha, noisy)[skip:]) <= target_std:
            return float(alpha)
    return 0.01


@dataclass(frozen=True, slots=True)
class SmoothingReport:
    fps: float
    noise_std: float
    raw_std: float
    filtered_std: float
    ema_alpha: float
    step_settle_one_euro: int
    step_settle_ema: int
    # speed (widths/s) → steady-state lag (ms)
    ramp_lag_one_euro_ms: dict[float, float]
    ramp_lag_ema_ms: dict[float, float]

    @property
    def noise_reduction(self) -> float:
        return 1.0 - self.filtered_std / self.raw_std


def evaluate(
    config: SmoothingConfig | None = None,
    fps: float = 24.0,
    noise_std: float = 0.0015,
    step: float = 0.1,
    speeds: tuple[float, ...] = (0.1, 0.25, 0.5, 1.0, 2.0),
) -> SmoothingReport:
    config = config or SmoothingConfig()
    noisy = still_noise(noise_std)
    filtered = run_one_euro(config, noisy, fps)
    filtered_std = float(np.std(filtered[30:]))
    alpha = matched_ema_alpha(filtered_std, noisy)

    step_signal = np.concatenate([np.full(30, 0.5), np.full(60, 0.5 + step)])
    settle_one_euro = settle_frames(run_one_euro(config, step_signal, fps), 30, 0.5 + step)
    settle_ema = settle_frames(run_ema(alpha, step_signal), 30, 0.5 + step)

    lag_one_euro: dict[float, float] = {}
    lag_ema: dict[float, float] = {}
    frames = int(fps * 3)
    t = np.arange(frames) / fps
    for speed in speeds:
        ramp = 0.2 + speed * t
        lag_one_euro[speed] = steady_ramp_lag_ms(run_one_euro(config, ramp, fps), speed, fps, settle=int(fps))
        lag_ema[speed] = steady_ramp_lag_ms(run_ema(alpha, ramp), speed, fps, settle=int(fps))

    return SmoothingReport(
        fps=fps,
        noise_std=noise_std,
        raw_std=float(np.std(noisy[30:])),
        filtered_std=filtered_std,
        ema_alpha=alpha,
        step_settle_one_euro=settle_one_euro,
        step_settle_ema=settle_ema,
        ramp_lag_one_euro_ms=lag_one_euro,
        ramp_lag_ema_ms=lag_ema,
    )
