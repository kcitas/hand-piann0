"""Trace format v4: the 21 landmarks of each hand, beside the per-finger trace.

A v4 recording ``NAME.csv`` is a set of files (none overwrites an existing one):

* ``NAME.csv``            the per-finger trace (same columns as v2/v3 plus
                          ``seq, hand_status, common_mode, baseline`` at the end;
                          older readers ignore them, newer readers accept their absence).
* ``NAME.landmarks.csv``  one row per hand per processed frame (both hands, always:
                          a lost hand is an explicit row with empty coordinates).
* ``NAME.meta.json``      format version, clock, camera size, mirroring, protocol,
                          detector configuration, column conventions.
* ``NAME.labels.csv``     independent labels (``labels.py``), only if a source
                          such as the physical keyboard produced any.

Landmark columns (``LANDMARK_COLUMNS``)
---------------------------------------
``t``                  seconds, ``time.perf_counter()`` of frame capture (monotonic,
                       arbitrary origin; the same ``t`` as the per-finger trace).
``seq``                processed-frame counter of the pipeline.
``hand``               the PHYSICAL hand after temporal identity (LEFT/RIGHT).
``status``             TRACKED (fresh landmarks), HELD (no fresh data, last state kept
                       for a grace period), LOST (no data).
``detected``           True if MediaPipe reported a hand assigned to this one this frame.
``detector_label``     the physical hand implied by MediaPipe's label this frame (empty
                       if not detected); differs from ``hand`` when identity overrode it.
``outlier_rejected``   True if this frame's detection was rejected as a jump.
``handedness_score``   MediaPipe's left/right classification score of the observation
                       in use; the ONLY per-hand score the Tasks API exposes. Not a
                       landmark-quality measure.
``rejected_outliers``  cumulative outliers rejected for this hand.
``raw_{x,y,z}{i}``     MediaPipe image landmarks of this frame's detection, unsmoothed
                       (x, y: fraction of image width / height, may slightly exceed
                       [0, 1]; z: relative depth, wrist origin, ≈ x scale, smaller =
                       closer to the camera). Empty if not detected. In the camera
                       view as processed: mirrored if ``meta.mirrored``.
``sm_{x,y,z}{i}``      the One Euro-smoothed landmarks the detector used. Empty when LOST;
                       during HELD they are the last smoothed values (stale).
``w_{x,y,z}{i}``       ``hand_world_landmarks`` (metres, origin near the hand centre).
``vis{i}``, ``pres{i}``  per-landmark visibility / presence if the model provides them;
                       empty otherwise (never filled with defaults).

``i`` is MediaPipe's landmark index (0 = wrist … 20 = pinky tip, see ``Landmark``).
Size: about 2 KB per hand per frame (~15 MB for 3 minutes at 24 fps).
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from handpiano.interaction.benchmark import Series
from handpiano.interaction.landmark_features import ALL_FEATURES, landmark_features
from handpiano.tracking.finger_tracker import HandState
from handpiano.tracking.landmarks import NUM_LANDMARKS, PLAYING_FINGERS, Handedness

FORMAT_VERSION = 4
_AXES = ("x", "y", "z")
_HEAD = ("t", "seq", "hand", "status", "detected", "detector_label", "outlier_rejected", "handedness_score",
         "rejected_outliers")


def _coord_columns(prefix: str) -> tuple[str, ...]:
    return tuple(f"{prefix}_{a}{i}" for i in range(NUM_LANDMARKS) for a in _AXES)


LANDMARK_COLUMNS = (
    *_HEAD, *_coord_columns("raw"), *_coord_columns("sm"), *_coord_columns("w"),
    *(f"vis{i}" for i in range(NUM_LANDMARKS)), *(f"pres{i}" for i in range(NUM_LANDMARKS)),
)
# Image coordinates outside this range are treated as invalid (MediaPipe extrapolates a
# little beyond the border; values far outside are corrupt data, not a hand).
VALID_IMAGE_RANGE = (-1.0, 2.0)


def sidecar_paths(path: Path) -> dict[str, Path]:
    """All files of a v4 recording named after ``path`` (the per-finger CSV)."""
    stem = path.with_suffix("") if path.suffix == ".csv" else path
    return {
        "trace": path,
        "landmarks": stem.with_name(stem.name + ".landmarks.csv"),
        "meta": stem.with_name(stem.name + ".meta.json"),
        "labels": stem.with_name(stem.name + ".labels.csv"),
    }


@dataclass(frozen=True, slots=True)
class HandSample:
    """One hand at one processed frame, as recorded."""

    t: float
    seq: int
    hand: Handedness
    status: str
    detected: bool
    detector_label: str
    outlier_rejected: bool
    handedness_score: float | None
    rejected_outliers: int
    raw: np.ndarray | None  # (21, 3)
    smoothed: np.ndarray | None  # (21, 3)
    world: np.ndarray | None  # (21, 3)
    visibility: np.ndarray | None  # (21,)
    presence: np.ndarray | None  # (21,)

    @property
    def tracked(self) -> bool:
        return self.status == "TRACKED"


def _copy(a: np.ndarray | None) -> np.ndarray | None:
    return None if a is None else np.array(a, dtype=np.float32, copy=True)


def sample_from_state(t: float, seq: int, state: HandState) -> HandSample:
    """Copies what is needed from the tracker's hand state (cheap; runs on the tracking thread)."""
    obs = state.observation
    return HandSample(
        t=t, seq=seq, hand=state.hand, status=state.status.name, detected=obs is not None,
        detector_label="" if state.detector_label is None else state.detector_label.name,
        outlier_rejected=state.outlier_rejected,
        handedness_score=state.score if state.score is not None else (None if obs is None else obs.score),
        rejected_outliers=state.rejected_outliers,
        raw=None if obs is None else _copy(obs.landmarks),
        smoothed=_copy(state.landmarks),
        world=None if obs is None else _copy(obs.world_landmarks),
        visibility=None if obs is None else _copy(obs.visibility),
        presence=None if obs is None else _copy(obs.presence),
    )


def _cells(a: np.ndarray | None, n: int, fmt: str) -> list[str]:
    if a is None:
        return [""] * n
    return [format(float(v), fmt) for v in np.asarray(a).reshape(-1)]


def _row(s: HandSample) -> list[str]:
    n3 = NUM_LANDMARKS * 3
    return [
        f"{s.t:.4f}", str(s.seq), s.hand.name, s.status, str(s.detected), s.detector_label,
        str(s.outlier_rejected), "" if s.handedness_score is None else f"{s.handedness_score:.4f}",
        str(s.rejected_outliers),
        *_cells(s.raw, n3, ".6f"), *_cells(s.smoothed, n3, ".6f"), *_cells(s.world, n3, ".6f"),
        *_cells(s.visibility, NUM_LANDMARKS, ".4f"), *_cells(s.presence, NUM_LANDMARKS, ".4f"),
    ]


def write_landmarks(path: Path, samples: list[HandSample]) -> None:
    with path.open("x", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(LANDMARK_COLUMNS)
        for s in samples:
            w.writerow(_row(s))


def write_meta(path: Path, meta: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as f:
        json.dump({"format": "handpiano-trace", "version": FORMAT_VERSION, **meta}, f, indent=2, ensure_ascii=False)
        f.write("\n")


def load_meta(path: Path) -> dict[str, Any]:
    meta = json.loads(path.read_text(encoding="utf-8"))
    if meta.get("format") != "handpiano-trace":
        raise ValueError(f"{path}: not a HandPiano trace metadata file")
    return meta


# -- reading ------------------------------------------------------------------------------


def _array(r: dict[str, str], prefix: str) -> np.ndarray | None:
    """(21, 3) array, or None if the block is empty. Non-numeric cells become NaN."""
    cells = [r.get(c, "") for c in _coord_columns(prefix)]
    if all(c == "" for c in cells):
        return None
    values = []
    for c in cells:
        try:
            values.append(float(c))
        except ValueError:
            values.append(float("nan"))
    return np.array(values, dtype=np.float64).reshape(NUM_LANDMARKS, 3)


def _scores(r: dict[str, str], prefix: str) -> np.ndarray | None:
    cells = [r.get(f"{prefix}{i}", "") for i in range(NUM_LANDMARKS)]
    if any(c == "" for c in cells):
        return None
    return np.array([float(c) for c in cells], dtype=np.float64)


def _valid_image(a: np.ndarray | None) -> bool:
    if a is None or not np.all(np.isfinite(a)):
        return False
    lo, hi = VALID_IMAGE_RANGE
    return bool(np.all((a[:, :2] >= lo) & (a[:, :2] <= hi)))


@dataclass(frozen=True, slots=True)
class LandmarkLoad:
    samples: list[HandSample]
    # Human-readable problems found (invalid coordinates set to None, etc.).
    issues: list[str]


def load_landmarks(path: Path) -> LandmarkLoad:
    """Reads a ``.landmarks.csv``. Raises ValueError on a broken file (missing columns,
    bad timestamps); invalid coordinates are dropped (set to None) and reported in ``issues``."""
    samples: list[HandSample] = []
    issues: list[str] = []
    last_t: dict[str, float] = {}
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = set(_HEAD) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        for line, r in enumerate(reader, start=2):
            try:
                t = float(r["t"])
            except ValueError as exc:
                raise ValueError(f"{path}:{line}: bad timestamp {r['t']!r}") from exc
            if not np.isfinite(t):
                raise ValueError(f"{path}:{line}: non-finite timestamp")
            hand = r["hand"]
            if hand not in ("LEFT", "RIGHT"):
                raise ValueError(f"{path}:{line}: bad hand {hand!r}")
            if hand in last_t and t <= last_t[hand]:
                raise ValueError(f"{path}:{line}: timestamp {t} not after {last_t[hand]} for {hand}")
            last_t[hand] = t
            raw, smoothed, world = _array(r, "raw"), _array(r, "sm"), _array(r, "w")
            if raw is not None and not _valid_image(raw):
                issues.append(f"line {line}: invalid raw image landmarks (dropped)")
                raw = None
            if smoothed is not None and not _valid_image(smoothed):
                issues.append(f"line {line}: invalid smoothed landmarks (dropped)")
                smoothed = None
            if world is not None and not np.all(np.isfinite(world)):
                issues.append(f"line {line}: invalid world landmarks (dropped)")
                world = None
            status = r["status"]
            if status == "TRACKED" and smoothed is None:
                issues.append(f"line {line}: TRACKED without valid smoothed landmarks (treated as LOST)")
                status = "LOST"
            score = r.get("handedness_score") or ""
            samples.append(HandSample(
                t=t, seq=int(r["seq"]) if r.get("seq") else -1, hand=Handedness[hand], status=status,
                detected=r.get("detected") == "True", detector_label=r.get("detector_label") or "",
                outlier_rejected=r.get("outlier_rejected") == "True",
                handedness_score=float(score) if score else None,
                rejected_outliers=int(r["rejected_outliers"]) if r.get("rejected_outliers") else 0,
                raw=raw, smoothed=smoothed, world=world,
                visibility=_scores(r, "vis"), presence=_scores(r, "pres"),
            ))
    return LandmarkLoad(samples, issues)


def series_from_landmarks(samples: list[HandSample], aspect: float, source: str = "smoothed",
                          features: tuple[str, ...] = ALL_FEATURES) -> dict[str, Series]:
    """Per playing finger: every landmark feature over time (NaN where unusable).

    ``source="smoothed"`` reproduces what the detector saw (TRACKED frames only);
    ``"raw"`` uses the unsmoothed detection of every frame where it exists (to
    evaluate other filters). World features use the frame's world landmarks.
    """
    if source not in ("smoothed", "raw"):
        raise ValueError(f"source must be 'smoothed' or 'raw', not {source!r}")
    out: dict[str, Series] = {}
    for hand in Handedness:
        hs = [s for s in samples if s.hand is hand]
        t = np.array([s.t for s in hs], dtype=np.float64)
        fingers = [f for f in PLAYING_FINGERS if f.hand is hand]
        values = {f: {name: np.full(len(hs), np.nan) for name in features} for f in fingers}
        for i, s in enumerate(hs):
            pts = (s.smoothed if s.tracked else None) if source == "smoothed" else s.raw
            world = s.world if (s.tracked or source == "raw") else None
            if pts is None and world is None:
                continue
            for f in fingers:
                feats = landmark_features(pts, f.finger, aspect, world)
                for name in features:
                    values[f][name][i] = feats[name]
        for f in fingers:
            out[f.name] = Series(t, values[f])
    return out
