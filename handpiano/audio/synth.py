"""HandPiano's own piano sound, computed from scratch (no recording, nothing to download).

Each note is additive synthesis shaped like a struck string:

* partials 1…N at slightly stretched frequencies (string stiffness: f_n = n·f0·√(1 + B·n²)),
  so the upper partials sit a little sharp, as on a real piano;
* each partial decays exponentially, the higher ones faster, and every note decays faster the
  higher it is (short treble, long bass);
* two "strings" a cent apart per note give a slow, warm beating;
* a 2 ms attack and a very short, quiet noise burst stand in for the hammer.

The result is normalised to a fixed peak so that every note has the same loudness before velocity.
"""

from __future__ import annotations

import numpy as np

from handpiano.instrument.note import midi_to_frequency

SAMPLE_RATE = 44100
SECONDS = 3.0
PEAK = 0.6  # of full scale, before velocity and master volume (laptop speakers distort near 1.0)
PARTIALS = 10
STIFFNESS = 0.0004  # inharmonicity coefficient B
DETUNE_CENTS = 1.2  # second string
TAIL_FADE_S = 0.3


def piano_note(midi: int, rate: int = SAMPLE_RATE, seconds: float = SECONDS, seed: int = 0) -> np.ndarray:
    """One note, mono float32 in [-PEAK, PEAK]. Deterministic for a given midi and seed."""
    f0 = midi_to_frequency(midi)
    t = np.arange(int(seconds * rate), dtype=np.float64) / rate
    # Bass rings for ~3 s, treble for well under 1 s (time constant of the fundamental).
    tau0 = float(np.clip(1.6 * (261.63 / f0) ** 0.6, 0.25, 3.0))
    out = np.zeros_like(t)
    nyquist = rate / 2
    for n in range(1, PARTIALS + 1):
        fn = n * f0 * np.sqrt(1.0 + STIFFNESS * n * n)
        if fn >= nyquist * 0.9:
            break
        amplitude = n ** -1.3
        tau = tau0 / (1.0 + 0.55 * (n - 1))
        envelope = amplitude * np.exp(-t / tau)
        detuned = fn * 2.0 ** (DETUNE_CENTS / 1200.0)
        out += envelope * (np.sin(2 * np.pi * fn * t) + 0.6 * np.sin(2 * np.pi * detuned * t + 0.7 * n))
    attack = np.minimum(1.0, t / 0.002)
    out *= attack
    # Hammer: 12 ms of decaying noise, low in level, high-passed by differencing.
    rng = np.random.default_rng(seed + midi)
    hammer_n = int(0.012 * rate)
    noise = np.diff(rng.standard_normal(hammer_n + 1)) * np.exp(-np.arange(hammer_n) / (hammer_n / 4))
    out[:hammer_n] += 0.04 * noise * np.abs(out).max()
    fade_n = int(TAIL_FADE_S * rate)
    out[-fade_n:] *= np.linspace(1.0, 0.0, fade_n)
    out *= PEAK / max(float(np.abs(out).max()), 1e-9)
    return out.astype(np.float32)
