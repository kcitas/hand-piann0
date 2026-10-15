"""Independent press labels: what really happened, kept apart from what the detector said.

A label file (``<trace>.labels.csv``) is never written from detector output or
from the cue schedule: a cue says when a tap was ASKED for, the detector says
what it THOUGHT; neither is evidence that a finger touched anything. Each label
states its ``source`` (how it is known) and ``confidence``.

Columns
-------
``t_start``     seconds, same clock as the trace's ``t`` (``time.perf_counter``).
``t_end``       seconds; empty for point labels (motion_onset, contact, release).
``target``      a finger (``RIGHT_INDEX`` …), a hand (``LEFT`` / ``RIGHT``) or ``ALL``.
``label``       one of ``LABELS``.
``source``      one of ``SOURCES``.
``confidence``  ``confirmed`` or ``uncertain``.
``note``        free text (e.g. ``key=F``).

Labels
------
``rest``                   interval: the target is not pressing nor intending to.
``motion_onset``           point: the finger starts moving towards a press.
``contact``                point: physical contact / press confirmed.
``release``                point: contact ends.
``neighbour_motion``       interval: the target moved involuntarily while ANOTHER finger pressed.
``insufficient_tracking``  interval: the data cannot tell (hand lost, occluded, blurred).
``ambiguous``              interval: something happened that cannot be classified.

Sources
-------
``keyboard``      a physical computer key went down/up (``KEY_FINGERS`` maps keys
                  to fingers by protocol: the finger is assumed, the timing is measured).
``video_review``  a person reviewed a video, frame by frame (only with explicit consent;
                  HandPiano does not save video).
``manual``        a person annotated it otherwise (e.g. live notes); state how in ``note``.
``tracking``      only for ``insufficient_tracking``: the tracker reported the hand LOST/HELD.

``detector`` and ``cue`` are rejected on purpose.
"""

from __future__ import annotations

import csv
import math
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from handpiano.tracking.landmarks import FingerId

LABELS = ("rest", "motion_onset", "contact", "release", "neighbour_motion", "insufficient_tracking", "ambiguous")
POINT_LABELS = frozenset({"motion_onset", "contact", "release"})
SOURCES = ("keyboard", "video_review", "manual", "tracking")
FORBIDDEN_SOURCES = frozenset({"detector", "cue"})
CONFIDENCES = ("confirmed", "uncertain")
LABEL_COLUMNS = ("t_start", "t_end", "target", "label", "source", "confidence", "note")
TARGETS = frozenset({*(f.name for f in FingerId), "LEFT", "RIGHT", "ALL"})

# Home row of a physical keyboard, by protocol. "Ñ" (Spanish layout) and ";" (US layout)
# are the same physical key. The key identifies the finger only if the person follows the protocol.
KEY_FINGERS: dict[str, FingerId] = {
    "A": FingerId.LEFT_PINKY, "S": FingerId.LEFT_RING, "D": FingerId.LEFT_MIDDLE, "F": FingerId.LEFT_INDEX,
    "J": FingerId.RIGHT_INDEX, "K": FingerId.RIGHT_MIDDLE, "L": FingerId.RIGHT_RING,
    "Ñ": FingerId.RIGHT_PINKY, ";": FingerId.RIGHT_PINKY,
}


@dataclass(frozen=True, slots=True)
class Label:
    t_start: float
    t_end: float | None
    target: str
    label: str
    source: str
    confidence: str = "confirmed"
    note: str = ""

    def validate(self) -> None:
        """Raises ValueError if the label breaks the format's rules."""
        if self.source in FORBIDDEN_SOURCES:
            raise ValueError(f"source {self.source!r} is not independent evidence (detector output / cue schedule)")
        if self.label not in LABELS:
            raise ValueError(f"unknown label {self.label!r}")
        if self.source not in SOURCES:
            raise ValueError(f"unknown source {self.source!r}")
        if self.source == "tracking" and self.label != "insufficient_tracking":
            raise ValueError("source 'tracking' only supports 'insufficient_tracking'")
        if self.confidence not in CONFIDENCES:
            raise ValueError(f"unknown confidence {self.confidence!r}")
        if self.target not in TARGETS:
            raise ValueError(f"unknown target {self.target!r}")
        if not _finite(self.t_start):
            raise ValueError("t_start must be a finite number")
        if self.label in POINT_LABELS:
            if self.t_end is not None:
                raise ValueError(f"{self.label!r} is a point label: leave t_end empty")
        elif self.t_end is None or not _finite(self.t_end) or self.t_end < self.t_start:
            raise ValueError(f"{self.label!r} is an interval: t_end must be ≥ t_start")


def _finite(v: float) -> bool:
    return math.isfinite(v)


def write_labels(path: Path, labels: Iterable[Label]) -> None:
    """Writes a NEW label file (refuses to overwrite). Every label is validated first."""
    items = sorted(labels, key=lambda lb: (lb.t_start, lb.target, lb.label))
    for lb in items:
        lb.validate()
    with path.open("x", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(LABEL_COLUMNS)
        for lb in items:
            w.writerow((f"{lb.t_start:.4f}", "" if lb.t_end is None else f"{lb.t_end:.4f}", lb.target, lb.label,
                        lb.source, lb.confidence, lb.note))


def load_labels(path: Path) -> list[Label]:
    """Reads and validates a label file; the error names the offending line."""
    out: list[Label] = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = set(LABEL_COLUMNS) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        for line, r in enumerate(reader, start=2):
            try:
                lb = Label(float(r["t_start"]), float(r["t_end"]) if r["t_end"] else None, r["target"], r["label"],
                           r["source"], r["confidence"] or "confirmed", r.get("note") or "")
                lb.validate()
            except ValueError as exc:
                raise ValueError(f"{path}:{line}: {exc}") from exc
            out.append(lb)
    return out


@dataclass(frozen=True, slots=True)
class KeyEvent:
    t: float  # same clock as the trace (perf_counter seconds), taken when Qt delivered the event
    key: str  # upper-case key text, e.g. "F"
    down: bool


def labels_from_keys(events: Iterable[KeyEvent], end_t: float | None = None) -> list[Label]:
    """Physical key down/up → contact/release labels (source ``keyboard``).

    Keys outside ``KEY_FINGERS`` become ``ambiguous`` (uncertain): a key was hit,
    but by which finger is unknown. A key still down at the end gets no release.
    """
    out: list[Label] = []
    down_at: dict[str, float] = {}
    for e in sorted(events, key=lambda e: e.t):
        finger = KEY_FINGERS.get(e.key)
        note = f"key={e.key}"
        if finger is None:
            if e.down:
                down_at[e.key] = e.t
            elif e.key in down_at:
                out.append(Label(down_at.pop(e.key), e.t, "ALL", "ambiguous", "keyboard", "uncertain",
                                 note + " (key outside the protocol's map)"))
            continue
        if e.down:
            if e.key in down_at:
                continue  # a second down without up (should not happen with auto-repeat filtered)
            down_at[e.key] = e.t
            out.append(Label(e.t, None, finger.name, "contact", "keyboard", "confirmed", note))
        elif e.key in down_at:
            down_at.pop(e.key)
            out.append(Label(e.t, None, finger.name, "release", "keyboard", "confirmed", note))
    for key, t in down_at.items():
        if key not in KEY_FINGERS and end_t is not None:
            out.append(Label(t, max(t, end_t), "ALL", "ambiguous", "keyboard", "uncertain",
                             f"key={key} (outside the map, no release)"))
    return out
