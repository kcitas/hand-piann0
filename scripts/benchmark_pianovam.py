"""Threshold-free comparison of HandPiano's press signals on PianoVAM recordings (local files).

PianoVAM (CC BY-NC-SA 4.0, https://huggingface.co/datasets/PianoVAM/PianoVAM_v1) is 45 GB in
total; this benchmark only needs, per recording, the landmark JSON and the manual fingering TSV:

    BASE=https://huggingface.co/datasets/PianoVAM/PianoVAM_v1/resolve/main
    curl -L -o metadata.json $BASE/metadata.json
    for id in 2024-02-17_21-44-37 2024-04-08_22-49-18 2024-09-05_13-25-10; do
        curl -L --create-dirs -o Handskeleton/$id.json $BASE/Handskeleton/$id.json
        curl -L --create-dirs -o Fingering_GT/$id.tsv  $BASE/Fingering_GT/$id.tsv
    done

Usage: python scripts/benchmark_pianovam.py [--all-features] DATA_DIR [record_id ...]

``--all-features`` also scores the landmark features (joint bends, palm distance) and the
velocity + common-mode combination, on all non-thumb onsets only.

Events = MIDI onsets of non-thumb fingers (manual fingering labels); window
[onset − 0.15 s, onset + 0.10 s] against [onset − 0.5 s, onset − 0.2 s].
Background = times with no onset of that hand within ±1 s. The JSON hand
labels are checked against key positions for every recording (they come out
swapped) instead of being assumed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from handpiano.interaction.benchmark import (
    format_scores,
    hand_of_finger,
    score_signals,
    variants_for,
)
from handpiano.interaction.landmark_features import IMAGE_FEATURES
from handpiano.interaction.pianovam import (
    detect_label_swap,
    events_from_recording,
    label_errors,
    load_recording,
    series_from_recording,
)

PRE = (-0.5, -0.2)
POST = (-0.15, 0.10)


def background_times(rec, step: float = 0.5, clearance: float = 1.0) -> list[float]:
    onsets = np.array(sorted(float(n["onset"]) for n in rec.notes))
    last_onset = onsets.max() if len(onsets) else 0.0
    times = np.arange(1.0, last_onset, step)
    return [float(t) for t in times if len(onsets) == 0 or np.min(np.abs(onsets - t)) > clearance]


def main() -> int:
    args = sys.argv[1:]
    extended = "--all-features" in args
    args = [a for a in args if a != "--all-features"]
    if not args:
        print(__doc__)
        return 2
    data = Path(args[0])
    ids = args[1:] or sorted(p.stem for p in (data / "Fingering_GT").glob("*.tsv")
                                 if (data / "Handskeleton" / f"{p.stem}.json").is_file())
    for rid in ids:
        rec = load_recording(data / "Handskeleton" / f"{rid}.json", data / "Fingering_GT" / f"{rid}.tsv",
                             data / "metadata.json")
        errors = label_errors(rec)
        swapped = detect_label_swap(rec)
        events, isolated = events_from_recording(rec)
        print(f"{rid}: {len(rec.frames)} frames @ {rec.fps:.2f} fps; label check: kept {errors[False]:.1f}px, "
              f"swapped {errors[True]:.1f}px → using {'swapped' if swapped else 'kept'} labels; "
              f"{len(events)} non-thumb onsets, {len(isolated)} isolated")
        series = series_from_recording(rec, swapped)
        bg = background_times(rec)
        if extended:
            scores = score_signals(series, events, bg, hand_of_finger, pre=PRE, post=POST,
                                   variants=variants_for(IMAGE_FEATURES))
            print("\n".join(format_scores(scores, f"  -- {rid}, all features, all non-thumb onsets ({len(events)}), "
                                                   f"{len(bg)} background times")))
            continue
        for name, evs in (("isolated onsets", isolated), ("all non-thumb onsets", events)):
            scores = score_signals(series, evs, bg, hand_of_finger, pre=PRE, post=POST)
            print("\n".join(format_scores(scores, f"  -- {rid}, {name} ({len(evs)}), {len(bg)} background times")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
