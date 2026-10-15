"""Replays a recorded press trace (python -m handpiano --record-trace) through the press detector.

Usage:
    python scripts/analyze_trace.py trace.csv                  # summary + replay with the production config
    python scripts/analyze_trace.py trace.csv --feature tip_z  # another press signal
    python scripts/analyze_trace.py trace.csv --sweep          # every signal × a few thresholds

What is counted
---------------
* Presses are EVENTS (IDLE → PRESSED transitions of the per-finger machine),
  never frames. Rejected gestures are counted separately, by reason.
* "Intended" = presses by the finger(s) the step asked for; "unintended" = presses
  by any other finger, or any press in a step where nobody should press.
* "Intended" alone cannot reveal double triggering (10 presses for 5 taps still
  count as intended). v2 traces carry timed cues: then presses are matched to
  cues per finger (hit / missed / extra), which does reveal it.
* Presses within the first 1.5 s of a step are also shown as "transition": they
  may belong to the previous instruction.

Per step × finger the summary shows valid vs missing samples (N/A when there is
no data, never 0), signal maximum and range (p95 − p5), and the p95 of the fast
local excursion before and after removing the hand's common mode (median of its
four fingers: closing/tilting the hand moves all fingers, a tap moves one).
All numbers come from the trace; signals are unitless (hand sizes).
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

from handpiano.interaction.press_detector import PressConfig
from handpiano.interaction.press_features import FEATURES
from handpiano.interaction.trace import (
    TRANSITION_S,
    cue_diagnostics,
    load_trace,
    match_cues,
    pressed_spans,
    replay,
    summarize,
)
from handpiano.instrument.keyboard import KeyboardLayout


def fmt(v: float | None, digits: int = 2) -> str:
    return "N/A" if v is None else f"{v:.{digits}f}"


def print_summary(rows: list[dict[str, str]], config: PressConfig, layout: KeyboardLayout) -> None:
    print(f"Per step × finger (signal used for presses: {config.feature}; excursion = p95 |x − moving median 1.5 s|,"
          f" raw → common mode removed)")
    header = (f"  {'step':<18} {'finger':<13} {'valid':>5} {'miss':>4} "
              + " ".join(f"{f + ' max/range':>15} {f + ' exc':>13}" for f in FEATURES)
              + f" {'press':>5} {'trans':>5} {'maxdown':>7}  rejected")
    print(header)
    for s in summarize(rows, config, layout):
        cells = " ".join(
            f"{fmt(s.signal_max[f]) + '/' + fmt(s.signal_range[f]):>15} "
            f"{fmt(s.excursion_p95[f]) + '→' + fmt(s.excursion_p95_cm[f]):>13}"
            for f in FEATURES
        )
        rej = ", ".join(f"{k}={n}" for k, n in sorted(s.rejections.items())) or "-"
        mark = " *" if s.expected else ""
        print(f"  {s.step:<18} {s.finger:<13} {s.valid:>5} {s.missing:>4} {cells} {s.presses:>5} "
              f"{s.transition_presses:>5} {s.longest_pressed_s:>6.1f}s  {rej}{mark}")
    print(f"  (* = finger asked to press in that step; 'trans' = presses in the first {TRANSITION_S} s of the step;"
          " 'maxdown' = longest continuous PRESSED span started in that step)")


def report(rows: list[dict[str, str]], config: PressConfig, layout: KeyboardLayout) -> tuple[int, int]:
    results = replay(rows, config, layout)
    intended = unintended = 0
    for r in results:
        if r.step == "prepare":
            continue
        intended += r.intended
        unintended += r.unintended
        detail = ", ".join(f"{f}={n}" for f, n in sorted(r.presses.items())) or "none"
        exp = "+".join(r.expected) or "nobody"
        print(f"  {r.step:<18} expected {exp:<24} intended {r.intended:>3}  unintended {r.unintended:>3}   ({detail})")
    return intended, unintended


def print_cues(rows: list[dict[str, str]], config: PressConfig, layout: KeyboardLayout) -> None:
    results = match_cues(rows, config, layout)
    if not results:
        print("\nNo timed cues in this trace (protocol v1): hits/misses per finger cannot be measured.")
        return
    print("\nCue matching (each press → nearest cue of its finger within ± half the cue interval; one-to-one):")
    for r in results:
        if r.cues == 0 and r.extra == 0:
            continue
        offs = ""
        if r.offsets:
            o = sorted(r.offsets)
            offs = f"  offset median {o[len(o) // 2]:+.2f} s (min {o[0]:+.2f}, max {o[-1]:+.2f})"
        print(f"  {r.finger:<13} cues {r.cues:>2}  hits {r.hits:>2}  missed {r.missed:>2}  extra presses {r.extra:>2}"
              f"  hit rate {fmt(r.hit_rate)}{offs}")
    print("  ('extra' and 'unintended' overlap: a wrong-finger press counts in both. Do not add them.)")


def print_cue_reasons(rows: list[dict[str, str]], config: PressConfig, layout: KeyboardLayout) -> None:
    diag = cue_diagnostics(rows, config, layout)
    if not diag:
        return
    from collections import Counter

    print("\nWhy each cue was missed (first matching reason; see trace.cue_diagnostics):")
    by: dict[str, Counter[str]] = {}
    for d in diag:
        by.setdefault(d.finger, Counter())["hit" if d.hit else (d.reason or "?")] += 1
    for finger in sorted(by):
        print(f"  {finger:<13} " + ", ".join(f"{k}={n}" for k, n in by[finger].most_common()))


def print_stuck(rows: list[dict[str, str]], config: PressConfig, layout: KeyboardLayout, min_s: float = 3.0) -> None:
    spans = [s for s in pressed_spans(rows, config, layout) if s.duration > min_s]
    print(f"\nPRESSED spans longer than {min_s:g} s: {len(spans)}")
    if not spans:
        return
    print(f"  {'finger':<13} {'step':<18} {'dur s':>5} {'min d':>6} {'med d':>6} {'idle d before':>13}"
          f" {'idle>on':>7} {'gap<3s':>6}  ended by          cause")
    for s in sorted(spans, key=lambda s: -s.duration):
        print(f"  {s.finger:<13} {s.step:<18} {s.duration:5.1f} {fmt(s.min_depth):>6} {fmt(s.median_depth):>6}"
              f" {fmt(s.idle_depth_before):>13} {fmt(s.idle_above_press_on):>7} {str(s.gap_before):>6}"
              f"  {s.ended_by:<16} {s.cause}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("--feature", choices=FEATURES)
    parser.add_argument("--press-on", type=float)
    parser.add_argument("--release-off", type=float)
    parser.add_argument("--min-velocity", type=float)
    parser.add_argument("--sweep", action="store_true", help="try every signal with a few press thresholds")
    parser.add_argument("--no-summary", action="store_true")
    args = parser.parse_args()

    rows = load_trace(args.trace)
    layout = KeyboardLayout.from_octaves("C4", 2)
    steps = list(dict.fromkeys(r["step"] for r in rows))
    print(f"{args.trace}: {len(rows)} rows ({len(rows) // 8} frames), steps: {', '.join(steps)}\n")

    config = PressConfig()
    overrides = {k: v for k, v in (("feature", args.feature), ("press_on", args.press_on),
                                    ("release_off", args.release_off), ("min_velocity", args.min_velocity)) if v is not None}
    config = replace(config, **overrides)
    if config.release_off >= config.press_on:
        print("release_off must be < press_on", file=sys.stderr)
        return 2

    if not args.no_summary:
        print_summary(rows, config, layout)

    if not args.sweep:
        print(f"\nReplay with {config}")
        i, u = report(rows, config, layout)
        print(f"\nTOTAL (steps after 'prepare') intended {i}, unintended {u}")
        print_cues(rows, config, layout)
        print_cue_reasons(rows, config, layout)
        print_stuck(rows, config, layout)
        return 0

    print("\nSweep (presses over all steps after 'prepare'; release_off = press_on / 2 in every row):")
    for feature in FEATURES:
        for press_on in (0.08, 0.12, 0.18, 0.25):
            cfg = replace(config, feature=feature, press_on=press_on, release_off=press_on / 2)
            results = replay(rows, cfg, layout)
            i = sum(r.intended for r in results if r.step != "prepare")
            u = sum(r.unintended for r in results if r.step != "prepare")
            cue_part = ""
            cues = match_cues(rows, cfg, layout)
            if cues:
                hits = sum(c.hits for c in cues)
                total = sum(c.cues for c in cues)
                extra = sum(c.extra for c in cues)
                cue_part = f"   cues hit {hits}/{total}, extra {extra}"
            print(f"  {feature:<6} press_on={press_on:<5} release_off={press_on / 2:<6} intended {i:>3}"
                  f"  unintended {u:>3}{cue_part}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
