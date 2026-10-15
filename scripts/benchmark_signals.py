"""Threshold-free comparison of press signals on HandPiano traces (v2 / focus protocols with cues).

Usage: python scripts/benchmark_signals.py traza_v2.csv traza_v3.csv [...]

Events are the cues (finger asked to tap, cue time); the event window is
[cue − 0.3 s, cue + 1.0 s] to include reaction time. Background = every 0.5 s of
rest / posture / slide. See handpiano/interaction/benchmark.py for the measures.
A cue the person did not answer still counts as an event, which lowers every
score equally; the trace cannot tell those apart.
"""

from __future__ import annotations

import sys
from pathlib import Path

from handpiano.interaction.benchmark import (
    format_scores,
    hand_of_finger,
    score_signals,
    series_from_trace,
    trace_events,
)
from handpiano.interaction.trace import load_trace


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    for path in sys.argv[1:]:
        rows = load_trace(Path(path))
        if "cue" not in rows[0]:
            print(f"{path}: no cue column (protocol v1): no events to score.\n")
            continue
        events, background = trace_events(rows)
        scores = score_signals(series_from_trace(rows), events, background, hand_of_finger)
        print("\n".join(format_scores(scores, f"== {path}: {len(events)} cue events, {len(background)} background times")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
