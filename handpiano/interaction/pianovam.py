"""Adapter for the PianoVAM dataset (Kim et al., ISMIR 2025; CC BY-NC-SA 4.0).

PianoVAM pairs top-view piano videos with Disklavier MIDI. Per recording we use:

* ``Handskeleton/<id>.json``: per video frame, {"Left": [[x, y, z] × 21] | null,
  "Right": ...}: MediaPipe-format normalized landmarks.
* ``Fingering_GT/<id>.tsv``: per note: onset (s), offsets, MIDI note, velocity,
  hand (L/R, the player's physical hand), finger (1 = thumb … 5 = pinky);
  manually annotated (11 recordings, 150 notes each).
* ``metadata.json``: duration and keyboard corners (pixels).

Two facts were checked on the data rather than assumed (2026-10-11, 3 recordings):

* the JSON hand labels are the OPPOSITE of the physical hand (the fingertip of
  the annotated finger lies 7–16 px from its key with labels swapped, ~300 px
  without); ``detect_label_swap`` measures this for every recording;
* video and MIDI agree within ~±0.1 s (best alignment lag between −0.1 and +0.1 s).

The camera looks DOWN at the keyboard, while HandPiano's webcam looks at the
hands from the front, so signal directions and magnitudes do not transfer
directly; what transfers is the question "does a signal single out the
pressing finger from its neighbours".
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from handpiano.interaction.benchmark import Event, Series
from handpiano.interaction.landmark_features import IMAGE_FEATURES, landmark_features
from handpiano.tracking.landmarks import Finger

FINGER_BY_NUMBER = {"2": Finger.INDEX, "3": Finger.MIDDLE, "4": Finger.RING, "5": Finger.PINKY}  # thumbs not used
TIP_BY_NUMBER = {"1": 4, "2": 8, "3": 12, "4": 16, "5": 20}
_BLACK = {1, 3, 6, 8, 10}
VIDEO_WIDTH = 1920.0
VIDEO_ASPECT = 1080 / 1920


@dataclass(frozen=True, slots=True)
class PianoVamRecording:
    record_id: str
    frames: dict[str, dict[str, list[list[float]] | None]]
    notes: list[dict[str, str]]
    duration_s: float
    left_top: tuple[float, float]
    right_top: tuple[float, float]

    @property
    def fps(self) -> float:
        return len(self.frames) / self.duration_s


def parse_duration(text: str) -> float:
    """'0 days 00:02:57.835.000000' → 177.835 (PianoVAM's format has two dots)."""
    hms = text.split()[-1]
    h, m, s = hms.split(":")
    parts = s.split(".")
    return int(h) * 3600 + int(m) * 60 + float(parts[0] + "." + (parts[1] if len(parts) > 1 else "0"))


def load_recording(handskeleton: Path, fingering: Path, metadata: Path) -> PianoVamRecording:
    record_id = handskeleton.stem
    meta_all = json.loads(metadata.read_text(encoding="utf-8"))
    meta = next(v for v in meta_all.values() if v.get("record_time") == record_id)
    with fingering.open(newline="", encoding="utf-8") as f:
        notes = list(csv.DictReader(f, delimiter="\t"))

    def point(name: str) -> tuple[float, float]:
        x, y = (float(v) for v in meta[name].split(","))
        return x, y

    return PianoVamRecording(
        record_id=record_id,
        frames=json.loads(handskeleton.read_text(encoding="utf-8")),
        notes=notes,
        duration_s=parse_duration(meta["duration"]),
        left_top=point("Point_LT"),
        right_top=point("Point_RT"),
    )


def key_x(note: int, left_top: tuple[float, float], right_top: tuple[float, float]) -> float:
    """Pixel x of a MIDI note on an 88-key keyboard (A0 = 21) spanning left_top → right_top."""
    idx = sum(1 for n in range(21, note) if n % 12 not in _BLACK) + (0.5 if note % 12 not in _BLACK else 0.0)
    return left_top[0] + idx / 52 * (right_top[0] - left_top[0])


def label_errors(rec: PianoVamRecording) -> dict[bool, float]:
    """Median |annotated fingertip x − key x| (px) with the JSON labels kept (False) or swapped (True)."""
    out: dict[bool, float] = {}
    for swapped in (False, True):
        errors = []
        for n in rec.notes:
            frame = rec.frames.get(str(round(float(n["onset"]) * rec.fps))) or {}
            label = {"L": "Left", "R": "Right"}[n["hand"]]
            if swapped:
                label = "Right" if label == "Left" else "Left"
            hand = frame.get(label)
            if hand:
                tip = hand[TIP_BY_NUMBER[n["finger"]]][0] * VIDEO_WIDTH
                errors.append(abs(tip - key_x(int(n["note"]), rec.left_top, rec.right_top)))
        out[swapped] = float(np.median(errors)) if errors else float("nan")
    return out


def detect_label_swap(rec: PianoVamRecording) -> bool:
    """True if the JSON 'Left'/'Right' must be swapped to get the physical hand (measured, not assumed)."""
    e = label_errors(rec)
    return bool(e[True] < e[False])


def series_from_recording(rec: PianoVamRecording, swapped: bool) -> dict[str, Series]:
    """Per physical finger (LEFT_INDEX…RIGHT_PINKY): HandPiano's press features and the
    landmark features (joint bends, palm distance; image coordinates only: no world landmarks)."""
    n = len(rec.frames)
    t = np.arange(n) / rec.fps
    out: dict[str, Series] = {}
    for hand_name, physical in (("Left", "LEFT"), ("Right", "RIGHT")):
        label = ("Right" if hand_name == "Left" else "Left") if swapped else hand_name
        values = {finger: {f: np.full(n, np.nan) for f in IMAGE_FEATURES} for finger in FINGER_BY_NUMBER.values()}
        for i in range(n):
            hand = (rec.frames.get(str(i)) or {}).get(label)
            if not hand:
                continue
            landmarks = np.asarray(hand, dtype=np.float64)
            for finger in FINGER_BY_NUMBER.values():
                feats = landmark_features(landmarks, finger, VIDEO_ASPECT)
                for f in IMAGE_FEATURES:
                    values[finger][f][i] = feats[f]
        for finger, v in values.items():
            out[f"{physical}_{finger.name}"] = Series(t, v)
    return out


def events_from_recording(rec: PianoVamRecording, isolation_s: float = 0.25) -> tuple[list[Event], list[Event]]:
    """(all non-thumb onsets, isolated ones: no other onset of the same hand within ±isolation_s)."""
    all_events: list[Event] = []
    times_by_hand: dict[str, list[float]] = {}
    for n in rec.notes:
        times_by_hand.setdefault(n["hand"], []).append(float(n["onset"]))
    isolated: list[Event] = []
    for n in rec.notes:
        if n["finger"] not in FINGER_BY_NUMBER:
            continue
        hand = "LEFT" if n["hand"] == "L" else "RIGHT"
        ev = Event(f"{hand}_{FINGER_BY_NUMBER[n['finger']].name}", float(n["onset"]))
        all_events.append(ev)
        others = [t for t in times_by_hand[n["hand"]] if t != ev.t and abs(t - ev.t) <= isolation_s]
        if not others:
            isolated.append(ev)
    return all_events, isolated
