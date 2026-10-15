"""A/B replay of a press trace with the previous detector, each fix alone, and all fixes.

Usage: python scripts/compare_detectors.py traza_v2.csv

For each configuration it reports, from the same recorded signals:
  presses       press events after the 'prepare' step
  intended      presses by the finger(s) a step asked for
  unintended    presses by other fingers, or in steps where nobody should press
  no-press      presses during rest / posture / slide (subset of unintended)
  cues hit      v2 cues answered by a press of that finger (one-to-one, ± half interval)
  missed        cues without such a press
  extra         presses of a finger not assigned to any of its cues (overlaps 'unintended')
  max PRESSED   longest continuous PRESSED span, seconds
  >3 s          number of PRESSED spans longer than 3 s

Fewer presses alone is not an improvement: compare hits and missed too, to tell
"fewer stuck fingers" apart from "presses suppressed".
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

from handpiano.interaction.press_detector import PressConfig
from handpiano.interaction.trace import load_trace, match_cues, pressed_spans, replay
from handpiano.instrument.keyboard import KeyboardLayout

NO_PRESS_STEPS = {"rest", "posture", "slide"}


def configurations() -> list[tuple[str, PressConfig]]:
    legacy = PressConfig.legacy()
    return [
        ("previous (legacy)", legacy),
        ("+ baseline any depth", replace(legacy, baseline_any_depth=True)),
        ("+ relearn after loss", replace(legacy, relearn_after_loss=True)),
        ("+ common mode", replace(legacy, common_mode="hand_low")),
        ("all fixes (new default)", PressConfig()),
        ("all fixes, common mode idle_min", PressConfig(common_mode="idle_min")),
        ("all fixes, common mode off", PressConfig(common_mode="off")),
    ]


def evaluate(rows: list[dict[str, str]], config: PressConfig, layout: KeyboardLayout) -> dict[str, float]:
    results = [r for r in replay(rows, config, layout) if r.step != "prepare"]
    cues = match_cues(rows, config, layout)
    spans = pressed_spans(rows, config, layout)
    return {
        "presses": sum(sum(r.presses.values()) for r in results),
        "intended": sum(r.intended for r in results),
        "unintended": sum(r.unintended for r in results),
        "no_press": sum(r.unintended for r in results if r.step in NO_PRESS_STEPS),
        "hits": sum(c.hits for c in cues),
        "cues": sum(c.cues for c in cues),
        "missed": sum(c.missed for c in cues),
        "extra": sum(c.extra for c in cues),
        "max_pressed": max((s.duration for s in spans), default=0.0),
        "long_spans": sum(1 for s in spans if s.duration > 3.0),
    }


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    rows = load_trace(Path(sys.argv[1]))
    layout = KeyboardLayout.from_octaves("C4", 2)
    print(f"{sys.argv[1]}: {len(rows)} rows\n")
    print(f"  {'configuration':<34}{'presses':>8}{'intended':>9}{'unint.':>7}{'no-press':>9}"
          f"{'cues hit':>10}{'missed':>7}{'extra':>6}{'max PRESSED':>13}{'>3 s':>5}")
    for name, cfg in configurations():
        m = evaluate(rows, cfg, layout)
        print(f"  {name:<34}{m['presses']:>8}{m['intended']:>9}{m['unintended']:>7}{m['no_press']:>9}"
              f"{m['hits']:>5}/{m['cues']:<4}{m['missed']:>7}{m['extra']:>6}{m['max_pressed']:>12.1f}s{m['long_spans']:>5}")
    print("\n  ('extra' and 'unintended' overlap; do not add them.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
