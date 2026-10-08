"""Temporal tracking of both hands and the eight playing fingers.

Per frame: detector observations → stable hand identity → outlier rejection →
One Euro smoothing → per-finger position, velocity and jitter.

Tracking status is deliberately limited to what vision can tell us:

* ``TRACKED`` — fresh, accepted landmarks this frame.
* ``HELD``    — no usable landmarks this frame (missed detection or rejected
  outlier), last good position kept for at most ``lost_grace_s``.
* ``LOST``    — no usable data for longer than that.

Musical states (hover, down, release…) belong to the interaction layer.
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
    # Samples used to estimate jitter (RMS of raw − smoothed).
    jitter_window: int = 30
    smoothing: SmoothingConfig = field(default_factory=SmoothingConfig)
    identity: IdentityConfig = field(default_factory=IdentityConfig)


@dataclass(frozen=True, slots=True)
class HandState:
    hand: Handedness
    status: TrackStatus
    # Smoothed (21, 3) landmarks; None when LOST.
    landmarks: np.ndarray | None
    score: float | None
    rejected_outliers: int


@dataclass(frozen=True, slots=True)
class FingerState:
    finger: FingerId
    status: TrackStatus
    # Smoothed tip (x, y, z); None when LOST.
    tip: np.ndarray | None
    # Smoothed tip velocity (x, y, z) per second; None when not computable.
    velocity: np.ndarray | None
    # RMS distance between raw and smoothed tip over recent frames (normalized units).
    jitter: float | None


@dataclass(frozen=True, slots=True)
class TrackingState:
    hands: dict[Handedness, HandState]
    fingers: dict[FingerId, FingerState]

    @property
    def hands_tracked(self) -> int:
        return sum(1 for h in self.hands.values() if h.status is TrackStatus.TRACKED)


class _HandTrack:
    def __init__(self, hand: Handedness, config: FingerTrackerConfig) -> None:
        self.hand = hand
        self.config = config
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
        self.residuals: dict[FingerId, deque[float]] = {
            f: deque(maxlen=config.jitter_window) for f in PLAYING_FINGERS if f.hand is hand
        }

    def update(self, obs: HandObservation | None, t: float) -> HandState:
        if obs is not None and self._is_outlier(obs, t):
            obs = None

        if obs is None:
            if self.last_good_t is not None and t - self.last_good_t <= self.config.lost_grace_s:
                return self._state(TrackStatus.HELD)
            self._reset()
            return self._state(TrackStatus.LOST)

        raw = obs.landmarks.astype(np.float64)
        smoothed = self.filter(raw, t)
        self._update_velocity(smoothed, t)
        for finger, residuals in self.residuals.items():
            tip = finger.finger.tip
            residuals.append(float(np.linalg.norm(raw[tip, :2] - smoothed[tip, :2])))
        self.smoothed, self.last_raw, self.last_good_t, self.score = smoothed, raw, t, obs.score
        return self._state(TrackStatus.TRACKED)

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
            return False
        self.rejected_outliers += 1
        return True

    def _update_velocity(self, smoothed: np.ndarray, t: float) -> None:
        tips = smoothed[[f.finger.tip for f in self.residuals]]
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
        for residuals in self.residuals.values():
            residuals.clear()

    def _state(self, status: TrackStatus) -> HandState:
        return HandState(
            hand=self.hand,
            status=status,
            landmarks=self.smoothed,
            score=self.score,
            rejected_outliers=self.rejected_outliers,
        )

    def finger_states(self, status: TrackStatus) -> dict[FingerId, FingerState]:
        states: dict[FingerId, FingerState] = {}
        for i, (finger, residuals) in enumerate(self.residuals.items()):
            tip = None if self.smoothed is None else self.smoothed[finger.finger.tip].copy()
            velocity = None
            if self.velocity is not None and status is TrackStatus.TRACKED:
                velocity = self.velocity[i].copy()
            jitter = float(np.sqrt(np.mean(np.square(residuals)))) if len(residuals) >= 5 else None
            states[finger] = FingerState(finger=finger, status=status, tip=tip, velocity=velocity, jitter=jitter)
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
