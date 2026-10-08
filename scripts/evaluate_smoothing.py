"""Prints a reproducible evaluation of the One Euro Filter vs. an equally stable EMA.

Synthetic signals (white noise at rest, a step, constant-speed ramps). It
characterizes the filter alone; it is not a measurement of the whole system.

Usage: python scripts/evaluate_smoothing.py [--fps 24] [--noise 0.0015]
"""

from __future__ import annotations

import argparse
import sys

from handpiano.tracking.smoothing import SmoothingConfig
from handpiano.tracking.smoothing_eval import evaluate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fps", type=float, default=24.0, help="frame rate (default: measured camera rate)")
    parser.add_argument("--noise", type=float, default=0.0015, help="noise std in image widths (0.0015 ≈ 2 px at 1280)")
    args = parser.parse_args()

    config = SmoothingConfig()
    r = evaluate(config, fps=args.fps, noise_std=args.noise)
    print(f"One Euro: min_cutoff={config.min_cutoff} Hz, beta={config.beta}, d_cutoff={config.d_cutoff} Hz")
    print(f"Frame rate {r.fps:g} fps, noise std {r.noise_std:g} widths\n")
    print(f"Noise at rest (std): raw {r.raw_std:.5f} → filtered {r.filtered_std:.5f}  ({r.noise_reduction:.0%} less)")
    print(f"Equally stable EMA: alpha = {r.ema_alpha:.3f}\n")
    print(f"Jump of 0.1 widths, frames to get within 10 %: One Euro {r.step_settle_one_euro}, EMA {r.step_settle_ema}\n")
    print("Steady lag at constant speed:")
    print("  speed (widths/s)   One Euro (ms)   EMA (ms)")
    for speed in sorted(r.ramp_lag_one_euro_ms):
        print(f"  {speed:>16g}   {r.ramp_lag_one_euro_ms[speed]:>13.1f}   {r.ramp_lag_ema_ms[speed]:>8.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
