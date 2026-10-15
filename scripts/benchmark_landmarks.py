"""Benchmark of landmark features on a v4 trace, against INDEPENDENT labels.

Usage: python scripts/benchmark_landmarks.py NAME.csv [--source smoothed|raw] [--cues]

Reads ``NAME.landmarks.csv``, ``NAME.meta.json`` and, if present,
``NAME.labels.csv`` (e.g. physical key contacts from ``--trace-protocol keys``).

* With labels: every landmark feature (and its ``:cm`` / ``:vel`` / ``:vel:cm``
  variants) is scored around each confirmed contact (window aligned to the
  contact, like PianoVAM's MIDI onsets), and the detector's recorded presses
  are matched one-to-one to the contacts (hits, misses, false presses, offset).
* Without labels: nothing is evaluated unless ``--cues`` is given; then the cue
  times are used, which are when a tap was ASKED for, not when it happened.

Older traces (v1–v3) have no landmarks: use ``scripts/benchmark_signals.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from handpiano.interaction.benchmark import (
    Event,
    events_from_labels,
    format_scores,
    hand_of_finger,
    match_detections,
    score_signals,
    trace_events,
    variants_for,
)
from handpiano.interaction.labels import load_labels
from handpiano.interaction.landmark_features import ALL_FEATURES
from handpiano.interaction.trace import load_trace
from handpiano.interaction.trace_landmarks import (
    load_landmarks,
    load_meta,
    series_from_landmarks,
    sidecar_paths,
)

CONTACT_PRE = (-0.5, -0.2)
CONTACT_POST = (-0.15, 0.10)


def detector_events(rows: list[dict[str, str]], kind: str = "press") -> list[Event]:
    """The detector's own events recorded live (``press`` or ``release``, any reason)."""
    out: list[Event] = []
    for r in rows:
        for e in (r.get("events") or "").split("|"):
            if e.startswith(f"{kind}:"):
                out.append(Event(r["finger"], float(r["t"])))
    return out


def detector_presses(rows: list[dict[str, str]]) -> list[Event]:
    return detector_events(rows, "press")


def print_matches(title: str, reference: list[Event], detected: list[Event], window: tuple[float, float],
                  names: tuple[str, str, str]) -> None:
    ref_name, det_name, extra_name = names
    print(f"-- {title} (one-to-one, offset window {window[0]:+.1f}…{window[1]:+.1f} s)")
    print(f"  {'finger':<14}{ref_name:>9}{det_name:>9}{'hits':>6}{'missed':>8}{extra_name:>9}{'median offset':>15}")
    for m in match_detections(reference, detected, window):
        off = f"{np.median(m.offsets) * 1000:+.0f} ms" if m.offsets else "N/A"
        print(f"  {m.finger:<14}{m.reference:>9}{m.detected:>9}{m.hits:>6}{m.misses:>8}{m.false_presses:>9}{off:>15}")


def main() -> int:
    args = sys.argv[1:]
    use_cues = "--cues" in args
    source = "smoothed"
    if "--source" in args:
        source = args[args.index("--source") + 1]
        args = [a for i, a in enumerate(args) if a != "--source" and (i == 0 or args[i - 1] != "--source")]
    args = [a for a in args if a != "--cues"]
    if len(args) != 1:
        print(__doc__)
        return 2
    files = sidecar_paths(Path(args[0]))
    if not files["landmarks"].is_file() or not files["meta"].is_file():
        print(f"{args[0]}: no landmarks/metadata (format before v4). Use scripts/benchmark_signals.py.")
        return 1
    meta = load_meta(files["meta"])
    loaded = load_landmarks(files["landmarks"])
    samples = loaded.samples
    if not samples or meta.get("aspect") is None:
        print("No landmark rows.")
        return 1
    tracked = sum(s.tracked for s in samples)
    print(f"{files['landmarks']}: {len(samples)} hand rows, {tracked} TRACKED "
          f"({tracked / len(samples):.0%}); protocol {meta.get('protocol')}; {len(loaded.issues)} data issues")
    for issue in loaded.issues[:10]:
        print(f"  ! {issue}")
    series = series_from_landmarks(samples, float(meta["aspect"]), source=source)
    variants = variants_for(ALL_FEATURES)
    rows = load_trace(files["trace"])
    start_t, end_t = samples[0].t, samples[-1].t

    if files["labels"].is_file():
        labels = load_labels(files["labels"])
        events, background = events_from_labels(labels, start_t, end_t)
        kinds: dict[str, int] = {}
        for lb in labels:
            kinds[lb.label] = kinds.get(lb.label, 0) + 1
        print(f"Independent labels: {kinds}; {len(events)} confirmed finger contacts, "
              f"{len(background)} background times")
        if events:
            scores = score_signals(series, events, background, hand_of_finger, pre=CONTACT_PRE, post=CONTACT_POST,
                                   variants=variants)
            print("\n".join(format_scores(scores, f"-- features around contacts ({source} landmarks)")))
            print_matches("detector presses vs contacts", events, detector_presses(rows), (-0.2, 0.4),
                          ("contacts", "presses", "false"))
            # Releases: a key-up with no detector release near it = a release missed or LATE (the key
            # stays lit / sounding); a detector release with no key-up = a spurious release.
            key_ups = [Event(lb.target, lb.t_start) for lb in labels
                       if lb.label == "release" and lb.confidence == "confirmed" and lb.target in series]
            print_matches("detector releases vs key releases", key_ups, detector_events(rows, "release"),
                          (-0.3, 0.5), ("key-ups", "releases", "spurious"))
        return 0

    print("No independent labels (NAME.labels.csv). Record with --trace-protocol keys, or annotate by hand.")
    if not use_cues:
        return 0
    cue_events, background = trace_events(rows)
    print("WARNING: cue times are when a tap was ASKED for, not when a contact happened.")
    scores = score_signals(series, cue_events, background, hand_of_finger, variants=variants)
    print("\n".join(format_scores(scores, f"-- features around cues ({source} landmarks)")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
