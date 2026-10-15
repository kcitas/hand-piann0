"""Synthetic sound banks for tests: each sample is a decaying sine at its root's frequency."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from handpiano.audio.instruments import INSTRUMENTS
from handpiano.audio.soundbank import BANK_FORMAT
from handpiano.audio.wavio import write_wav16
from handpiano.instrument.note import midi_to_frequency

RATE = 44100


def sine(midi: int, seconds: float, amplitude: float = 0.5, rate: int = RATE, decay: float = 0.0) -> np.ndarray:
    t = np.arange(int(seconds * rate)) / rate
    return (amplitude * np.exp(-decay * t) * np.sin(2 * np.pi * midi_to_frequency(midi) * t)).astype(np.float32)


def make_bank(sounds_dir: Path, instrument_id: str, seconds: float = 0.4) -> Path:
    instrument = INSTRUMENTS[instrument_id]
    folder = sounds_dir / instrument_id
    folder.mkdir(parents=True, exist_ok=True)
    entries = []
    for src in instrument.samples:
        y = sine(src.root, seconds, amplitude=0.3 + 0.2 * src.layer)
        if instrument.stereo:
            y = np.stack([y, y], axis=1)
        write_wav16(folder / src.cache_name, y, RATE)
        entries.append({"file": src.cache_name, "root": src.root, "layer": src.layer})
    (folder / "bank.json").write_text(json.dumps({"format": BANK_FORMAT, "instrument": instrument_id,
                                                  "samples": entries}))
    return folder


def make_all(sounds_dir: Path, seconds: float = 0.4) -> Path:
    for instrument_id in INSTRUMENTS:
        make_bank(sounds_dir, instrument_id, seconds)
    return sounds_dir


def peak_frequency(x: np.ndarray, rate: int = RATE, lo: float = 20.0, hi: float = 5000.0) -> float:
    mono = x.mean(axis=1) if x.ndim == 2 else x
    spec = np.abs(np.fft.rfft(mono * np.hanning(len(mono)), 1 << 19))
    freqs = np.fft.rfftfreq(1 << 19, 1 / rate)
    band = (freqs >= lo) & (freqs <= hi)
    return float(freqs[band][np.argmax(spec[band])])


def render(engine_or_mixer, seconds: float, block: int = 256) -> np.ndarray:
    mixer = getattr(engine_or_mixer, "mixer", engine_or_mixer)
    return np.concatenate([mixer.render(block) for _ in range(max(1, int(seconds * RATE / block)))])
