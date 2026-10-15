"""Finger → key mapping (hover only: no presses, no sound).

For each of the eight playing fingers:

    smoothed fingertip (camera coords) → calibration homography → piano coords → key_at()

The key under a finger depends only on where the finger is; no finger has a
fixed note. Fingers are mapped independently, so two fingers (of the same or
different hands) can report the same key; ``keys_under`` groups them without
inventing extra notes.

Without a valid calibration nothing is mapped: ``calibrated`` is False and
every finger's key is None.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from handpiano.instrument.keyboard import KeyboardLayout, PianoKey
from handpiano.mapping.calibration import PianoCalibration
from handpiano.tracking.finger_tracker import TrackingState, TrackStatus
from handpiano.tracking.landmarks import PLAYING_FINGERS, FingerId


@dataclass(frozen=True, slots=True)
class FingerKey:
    finger: FingerId
    # TRACKED / HELD / LOST, from the tracker (HELD = last known position).
    status: TrackStatus
    # Smoothed fingertip in normalized camera coordinates; None when LOST.
    camera_xy: tuple[float, float] | None
    # Same point in piano coordinates; None when LOST or uncalibrated.
    piano_xy: tuple[float, float] | None
    key: PianoKey | None
    # Distance (white-key widths) from the fingertip to the geometric centre of
    # ``key``. A geometric distance, not an error: no target is known here.
    distance_to_center: float | None

    @property
    def has_position(self) -> bool:
        return self.camera_xy is not None

    @property
    def inside(self) -> bool | None:
        """True/False once calibrated and the finger has a position; None otherwise."""
        if self.piano_xy is None:
            return None
        return self.key is not None


@dataclass(frozen=True, slots=True)
class MappingState:
    calibrated: bool
    fingers: dict[FingerId, FingerKey]

    @property
    def with_position(self) -> int:
        return sum(1 for f in self.fingers.values() if f.has_position)

    @property
    def inside(self) -> int | None:
        return sum(1 for f in self.fingers.values() if f.inside) if self.calibrated else None

    @property
    def outside(self) -> int | None:
        return sum(1 for f in self.fingers.values() if f.inside is False) if self.calibrated else None

    @property
    def keys_under(self) -> dict[int, tuple[FingerId, ...]]:
        """MIDI → fingers over that key, in finger order. One entry per key, however many fingers."""
        result: dict[int, list[FingerId]] = {}
        for finger in PLAYING_FINGERS:
            fk = self.fingers[finger]
            if fk.key is not None:
                result.setdefault(fk.key.midi, []).append(finger)
        return {midi: tuple(fingers) for midi, fingers in result.items()}


class KeyMapper:
    def __init__(self, layout: KeyboardLayout, calibration: PianoCalibration | None = None) -> None:
        self.layout = layout
        self._calibration = calibration

    @property
    def calibration(self) -> PianoCalibration | None:
        return self._calibration

    @calibration.setter
    def calibration(self, value: PianoCalibration | None) -> None:
        # A single reference swap: the tracking thread sees either the old or the new one.
        self._calibration = value

    def map_points(
        self, camera_points: np.ndarray, calibration: PianoCalibration | None = None
    ) -> list[tuple[tuple[float, float] | None, PianoKey | None, float | None]]:
        """(n, 2) camera points → [(piano point, key, distance to that key's centre)].

        The single implementation of camera → homography → key_at, used for every
        frame (all fingers in one homography call) and by the accuracy test.
        All None without a calibration.
        """
        calibration = calibration if calibration is not None else self._calibration
        if calibration is None:
            return [(None, None, None)] * len(camera_points)
        out: list[tuple[tuple[float, float] | None, PianoKey | None, float | None]] = []
        for point in calibration.to_piano(np.asarray(camera_points, dtype=np.float64)):
            px, py = float(point[0]), float(point[1])
            key = self.layout.key_at(px, py)
            out.append(((px, py), key, None if key is None else key.distance_to_center(px, py)))
        return out

    def map_point(
        self, camera_xy: tuple[float, float], calibration: PianoCalibration | None = None
    ) -> tuple[tuple[float, float] | None, PianoKey | None, float | None]:
        return self.map_points(np.array([camera_xy], dtype=np.float64), calibration)[0]

    def map(self, state: TrackingState) -> MappingState:
        calibration = self._calibration  # read once per frame
        tips_by_finger = {f: tip[:2] for f in PLAYING_FINGERS if (tip := state.fingers[f].tip) is not None}
        with_tip = list(tips_by_finger)
        mapped = {}
        if calibration is not None and with_tip:
            tips = np.array([tips_by_finger[f] for f in with_tip], dtype=np.float64)
            mapped = dict(zip(with_tip, self.map_points(tips, calibration), strict=True))
        result: dict[FingerId, FingerKey] = {}
        for finger in PLAYING_FINGERS:
            fs = state.fingers[finger]
            camera_xy = None if fs.tip is None else (float(fs.tip[0]), float(fs.tip[1]))
            piano_xy, key, distance = mapped.get(finger, (None, None, None))
            result[finger] = FingerKey(finger, fs.status, camera_xy, piano_xy, key, distance)
        return MappingState(calibrated=calibration is not None, fingers=result)
