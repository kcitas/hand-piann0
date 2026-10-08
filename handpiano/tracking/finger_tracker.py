"""Temporal tracking of both hands and the eight playing fingers.

Per frame: detector observations → stable hand identity → outlier rejection →
One Euro smoothing → per-finger position, velocity and jitter.

Tracking status is deliberately limited to what vision can tell us:

* ``TRACKED`` — fresh, accepted landmarks this frame.
* ``HELD``    — no usable landmarks this frame (missed detection or rejected
  outlier), last good position kept for at most ``lost_grace_s``.
* ``LOST``    — no usable data for longer than that.

Musical states (hover, down, release…) belong to the interaction layer.

Jitter
------
For each fingertip, over the last ``jitter_window`` TRACKED frames, a straight
line (constant velocity) is least-squares fitted to x(t) and to y(t)
separately. Jitter is the RMS of the deviations from those lines, combined as
``sqrt(mean(dx_px² + dy_px²))``. The tracker works in normalized coordinates, so
it stores the per-axis mean squares and the UI converts them to pixels with the
actual frame width and height (see ``jitter_px``).

Removing a linear trend means steady motion is not counted as jitter, but
acceleration (a strike, a direction change) is. It is computed twice:
``raw`` on MediaPipe's landmarks before smoothing and ``smoothed`` after the
One Euro Filter. Neither is the physical tremor of the hand: raw jitter is
detector noise plus real micro-movement; smoothed jitter is the residual
variability that reaches the instrument.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from handpiano.tracking.hand_identity import HandIdentityResolver, IdentityConfig
from handpiano.tracking.landmarks import PLAYING_FINGERS, FingerId, Handedness, HandObservation
from handpiano.tracking.smoothing import OneEuroFilter, SmoothingConfig


class TrackStatus(Enum):
    LOST = "LOST"
    TRACKED = "TRACKED"
    HELD = "HELD"


@dataclass(frozen=True, slots=True)
class FingerTrackerConfig:
    # Max time (s) to keep showing the last good position after detection stops.
    lost_grace_s: float = 0.12
    # A frame whose median landmark speed exceeds this (normalized units / s) is
    # treated as a detector glitch. ~8 image widths per second is beyond a real hand.
    max_speed: float = 8.0
    # After this many consecutive rejections the new position is accepted
    # (the hand really did move, e.g. after a re-detection).
    max_consecutive_outliers: int = 2
    # TRACKED frames used for jitter (~0.6 s at 24 fps). Jitter is None until full.
    jitter_window: int = 15
    smoothing: SmoothingConfig = field(default_factory=SmoothingConfig)
    identity: IdentityConfig = field(default_factory=IdentityConfig)


# Per-axis mean squared deviation (x, y) in normalized units².
JitterMs = tuple[float, float]


def jitter_px(mean_squares: JitterMs | None, width: int, height: int) -> float | None:
    """RMS jitter in pixels of a ``width`` × ``height`` frame."""
    if mean_squares is None:
        return None
    msx, msy = mean_squares
    return float(np.sqrt(msx * width**2 + msy * height**2))


@dataclass(frozen=True, slots=True)
class HandState:
    hand: Handedness
    status: TrackStatus
    # Smoothed (21, 3) landmarks, read-only; None when LOST.
    landmarks: np.ndarray | None
    # Handedness score of the observation in use; None when LOST.
    score: float | None
    # Label MediaPipe gave this frame, before identity resolution. None if not detected this frame.
    detector_label: Handedness | None
    # Outliers rejected for this hand since the pipeline started.
    rejected_outliers: int

    @property
    def label_overridden(self) -> bool:
        """True if temporal identity kept a label different from MediaPipe's for this frame."""
        return self.detector_label is not None and self.detector_label is not self.hand


@dataclass(frozen=True, slots=True)
class FingerState:
    finger: FingerId
    status: TrackStatus
    # Smoothed tip (x, y, z), read-only; None when LOST.
    tip: np.ndarray | None
    # Smoothed tip velocity (x, y, z) in normalized units per second; None when not computable.
    velocity: np.ndarray | None
    raw_jitter: JitterMs | None
    smoothed_jitter: JitterMs | None


@dataclass(frozen=True, slots=True)
class TrackingState:
    hands: dict[Handedness, HandState]
    fingers: dict[FingerId, FingerState]

    @property
    def hands_tracked(self) -> int:
        return sum(1 for h in self.hands.values() if h.status is TrackStatus.TRACKED)


def _read_only(a: np.ndarray) -> np.ndarray:
    a.setflags(write=False)
    return a


def _detrended_mean_squares(t: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Mean squared residual of a per-column least-squares line fit. y: (n, k) → (k,)."""
    tc = t - t.mean()
    yc = y - y.mean(axis=0)
    denom = float(np.dot(tc, tc))
    slope = (tc @ yc) / denom if denom > 0 else np.zeros(y.shape[1])
    residual = yc - np.outer(tc, slope)
    return np.mean(residual**2, axis=0)


class _HandTrack:
    def __init__(self, hand: Handedness, config: FingerTrackerConfig) -> None:
        self.hand = hand
        self.config = config
        self.fingers = tuple(f for f in PLAYING_FINGERS if f.hand is hand)
        self._tip_indices = [f.finger.tip for f in self.fingers]
        self.filter = OneEuroFilter(config.smoothing)
        self.smoothed: np.ndarray | None = None
        self.last_raw: np.ndarray | None = None
        self.last_good_t: float | None = None
        self.score: float | None = None
        self.consecutive_outliers = 0
        self.rejected_outliers = 0
        self.prev_tips: np.ndarray | None = None
        self.prev_t: float | None = None
        self.velocity: np.ndarray | None = None
        # (t, raw tips (4, 2), smoothed tips (4, 2)) for jitter.
        self.history: deque[tuple[float, np.ndarray, np.ndarray]] = deque(maxlen=config.jitter_window)

    def update(self, obs: HandObservation | None, t: float) -> HandState:
        detector_label = None if obs is None else obs.handedness
        if obs is not None and self._is_outlier(obs, t):
            obs = None

        if obs is None:
            if self.last_good_t is not None and t - self.last_good_t <= self.config.lost_grace_s:
                return self._state(TrackStatus.HELD, detector_label)
            self._reset()
            return self._state(TrackStatus.LOST, detector_label)

        raw = obs.landmarks.astype(np.float64)
        smoothed = _read_only(self.filter(raw, t))
        self._update_velocity(smoothed, t)
        self.history.append((t, raw[self._tip_indices, :2], smoothed[self._tip_indices, :2]))
        self.smoothed, self.last_raw, self.last_good_t, self.score = smoothed, raw, t, obs.score
        return self._state(TrackStatus.TRACKED, detector_label)

    def _is_outlier(self, obs: HandObservation, t: float) -> bool:
        if self.last_raw is None or self.last_good_t is None:
            return False
        dt = t - self.last_good_t
        if dt <= 0:
            return False
        displacement = np.linalg.norm(obs.landmarks[:, :2] - self.last_raw[:, :2], axis=1)
        speed = float(np.median(displacement)) / dt
        if speed <= self.config.max_speed:
            self.consecutive_outliers = 0
            return False
        self.consecutive_outliers += 1
        if self.consecutive_outliers > self.config.max_consecutive_outliers:
            # Persistent: the hand really is somewhere else now. Start fresh.
            self.consecutive_outliers = 0
            self.filter.reset()
            self.prev_tips = self.prev_t = self.velocity = None
            self.history.clear()
            return False
        self.rejected_outliers += 1
        return True

    def _update_velocity(self, smoothed: np.ndarray, t: float) -> None:
        tips = smoothed[self._tip_indices]
        if self.prev_tips is not None and self.prev_t is not None and t > self.prev_t:
            self.velocity = (tips - self.prev_tips) / (t - self.prev_t)
        else:
            self.velocity = None
        self.prev_tips, self.prev_t = tips, t

    def _reset(self) -> None:
        self.filter.reset()
        self.smoothed = self.last_raw = self.last_good_t = self.score = None
        self.prev_tips = self.prev_t = self.velocity = None
        self.consecutive_outliers = 0
        self.history.clear()

    def _state(self, status: TrackStatus, detector_label: Handedness | None) -> HandState:
        return HandState(
            hand=self.hand,
            status=status,
            landmarks=self.smoothed,
            score=self.score,
            detector_label=detector_label,
            rejected_outliers=self.rejected_outliers,
        )

    def _jitter(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Per-finger (msx, msy) for raw and smoothed tips, or None until the window is full."""
        if len(self.history) < self.config.jitter_window:
            return None
        t = np.array([h[0] for h in self.history])
        raw = np.stack([h[1] for h in self.history]).reshape(len(t), -1)
        smooth = np.stack([h[2] for h in self.history]).reshape(len(t), -1)
        return (
            _detrended_mean_squares(t, raw).reshape(-1, 2),
            _detrended_mean_squares(t, smooth).reshape(-1, 2),
        )

    def finger_states(self, status: TrackStatus) -> dict[FingerId, FingerState]:
        jitter = self._jitter()
        states: dict[FingerId, FingerState] = {}
        for i, finger in enumerate(self.fingers):
            tip = None if self.smoothed is None else _read_only(self.smoothed[finger.finger.tip].copy())
            velocity = None
            if self.velocity is not None and status is TrackStatus.TRACKED:
                velocity = _read_only(self.velocity[i].copy())
            raw_j = smooth_j = None
            if jitter is not None:
                raw_j = (float(jitter[0][i, 0]), float(jitter[0][i, 1]))
                smooth_j = (float(jitter[1][i, 0]), float(jitter[1][i, 1]))
            states[finger] = FingerState(
                finger=finger, status=status, tip=tip, velocity=velocity, raw_jitter=raw_j, smoothed_jitter=smooth_j
            )
        return states


class FingerTracker:
    def __init__(self, config: FingerTrackerConfig | None = None) -> None:
        self.config = config or FingerTrackerConfig()
        self._identity = HandIdentityResolver(self.config.identity)
        self._hands = {h: _HandTrack(h, self.config) for h in Handedness}

    def reset(self) -> None:
        self._identity.reset()
        self._hands = {h: _HandTrack(h, self.config) for h in Handedness}

    def update(self, observations: list[HandObservation], t: float) -> TrackingState:
        assigned = self._identity.resolve(observations, t)
        hands: dict[Handedness, HandState] = {}
        fingers: dict[FingerId, FingerState] = {}
        for hand, track in self._hands.items():
            state = track.update(assigned.get(hand), t)
            hands[hand] = state
            fingers.update(track.finger_states(state.status))
        return TrackingState(hands=hands, fingers=fingers)
