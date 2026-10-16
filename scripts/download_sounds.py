"""Downloads the three piano sample sets once and builds the local cache. After this, audio works offline.

Usage: python scripts/download_sounds.py [--only grand|upright|electric] [--sounds-dir DIR] [--force]

Only what HandPiano uses is fetched: the first seconds of each sample (HTTP Range
requests on git-commit-pinned URLs), for the notes and velocity layers listed in
``handpiano/audio/instruments.py``. Measured on 2026-10-11: ~75 MB for the grand,
~75 MB for the upright, ~16 MB for the electric piano; the cache takes ~115 MB.

Every converted sample is checked against the note it is mapped to: its measured pitch class
must be within ±50 cents (any semitone error fails), and at least 90 % of an instrument's
samples must be in the exact octave (a wrong octave naming convention would shift all of
them; isolated treble octave confusions are a limit of the estimator). Otherwise the
instrument is NOT installed. Sources,
licences and the SHA-256 of every downloaded byte range go to ``bank.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from handpiano.audio.engine import DEFAULT_SOUNDS_DIR  # noqa: E402
from handpiano.audio.instruments import INSTRUMENTS, Instrument, SampleSource  # noqa: E402
from handpiano.audio.soundbank import BANK_FORMAT  # noqa: E402
from handpiano.audio.wavio import parse_wav, write_wav16  # noqa: E402

HEADER_SLACK = 65536
MAX_CENTS = 50.0
# A wrong octave convention shifts EVERY sample; the estimator alone can mistake the octave of a
# few treble samples (inharmonic partials, sympathetic resonance): see build().
MIN_OCTAVE_AGREEMENT = 0.9


def fetch_prefix(url: str, n_bytes: int, retries: int = 3) -> bytes:
    request = urllib.request.Request(url, headers={"Range": f"bytes=0-{n_bytes - 1}", "User-Agent": "HandPiano"})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return response.read()
        except OSError:
            if attempt == retries - 1:
                raise
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError("unreachable")


def _spectrum(x: np.ndarray, rate: int) -> tuple[np.ndarray, np.ndarray]:
    mono = x.mean(axis=1) if x.ndim == 2 else x
    peak = float(np.max(np.abs(mono)))
    onset = int(np.argmax(np.abs(mono) > 0.05 * peak))
    seg = mono[onset + int(0.1 * rate): onset + int(0.1 * rate) + 32768]
    seg = seg * np.hanning(len(seg))
    n = 4 * len(seg)
    return np.abs(np.fft.rfft(seg, n)), np.fft.rfftfreq(n, 1 / rate)


def _band_peak(spec: np.ndarray, freqs: np.ndarray, f: float, rel: float = 0.025) -> tuple[float, float]:
    lo, hi = np.searchsorted(freqs, f * (1 - rel)), np.searchsorted(freqs, f * (1 + rel)) + 1
    if lo >= len(spec) or hi <= lo:
        return 0.0, f
    i = lo + int(np.argmax(spec[lo:hi]))
    return float(spec[i]), float(freqs[i])


def detect_pitch(x: np.ndarray, rate: int, expected_midi: int) -> tuple[int, float]:
    """(detected MIDI note, fine deviation in cents from it) by harmonic templates.

    Every note within ±13 semitones of the expected one is scored by the log-energy of its
    first six partials (±2.5 %, room for piano inharmonicity). Sub-octave candidates are
    excluded by requiring real energy at the candidate's own fundamental (> −45 dB of the
    spectrum peak) and by the octave check below. Octave and semitone errors are NOT folded:
    a sample mapped to the wrong note fails.
    """
    spec, freqs = _spectrum(x, rate)
    # Room rumble / handling noise below the lowest candidate is not part of the note.
    spec = np.where(freqs < 0.7 * 440.0 * 2 ** ((expected_midi - 13 - 69) / 12), 0.0, spec)
    floor = float(spec.max()) * 10 ** (-45 / 20)
    scores: dict[int, float] = {}
    for m in range(expected_midi - 13, expected_midi + 14):
        f = 440.0 * 2 ** ((m - 69) / 12)
        fundamental = _band_peak(spec, freqs, f)[0]
        if fundamental < floor:
            continue
        # Treble partials above ~8 kHz are mostly gone: score the ones that exist (at least 2), and
        # weight the fundamental (dominant in the treble, weak in the bass) only above 1 kHz.
        partials = max(2, min(6, int(8000.0 // f)))
        scores[m] = (2.0 if f > 1000.0 else 0.0) * math.log10(fundamental + 1e-9) + sum(
            math.log10(_band_peak(spec, freqs, h * f)[0] + 1e-9) for h in range(1, partials + 1))
    if not scores:
        return -1, float("nan")
    midi = max(scores, key=lambda m: scores[m])
    # Octave check: in the treble, string inharmonicity pulls the upper partials off the template
    # and undamped strings resonate, so the octave below can score best. A real fundamental is
    # not 26 dB weaker than the candidate an octave above it.
    while midi + 12 in scores:
        here = _band_peak(spec, freqs, 440.0 * 2 ** ((midi - 69) / 12))[0]
        above = _band_peak(spec, freqs, 440.0 * 2 ** ((midi + 12 - 69) / 12))[0]
        if here >= 0.05 * above:
            break
        midi += 12
    _, f_peak = _band_peak(spec, freqs, 440.0 * 2 ** ((midi - 69) / 12), rel=0.03)
    return midi, 1200 * math.log2(f_peak / (440.0 * 2 ** ((midi - 69) / 12)))


def process(instrument: Instrument, src: SampleSource, raw: bytes) -> tuple[np.ndarray, int, float]:
    wav = parse_wav(raw, allow_truncated=True)
    x = wav.samples
    channels = 2 if instrument.stereo else 1
    if x.shape[1] != channels:
        x = x.mean(axis=1, keepdims=True) if channels == 1 else np.repeat(x[:, :1], 2, axis=1)
    rate = wav.sample_rate
    mono = np.abs(x).max(axis=1)
    onset = int(np.argmax(mono > 0.01 * float(mono.max())))
    start = max(0, onset - int(0.003 * rate))
    keep = int(instrument.keep_s * rate)
    y = x[start:start + keep].astype(np.float32)
    if y.shape[0] < int(0.5 * rate):
        raise ValueError(f"{src.cache_name}: only {y.shape[0] / rate:.2f} s of audio")
    pre = onset - start
    if pre > 0:  # fade in the few ms before the onset (no click from a cut in the noise floor)
        y[:pre] *= np.linspace(0.0, 1.0, pre, dtype=np.float32)[:, None]
    fade = min(int(instrument.fade_s * rate), y.shape[0] // 2)
    y[-fade:] *= (0.5 * (1 + np.cos(np.linspace(0, np.pi, fade)))).astype(np.float32)[:, None]
    peak = float(np.abs(y).max())
    if peak > 0.999:
        y *= 0.999 / peak
    detected, cents = detect_pitch(y, rate, src.root)
    return y, rate, (detected - src.root) * 100.0 + cents


def folded_cents(error_cents: float) -> float:
    """Pitch-class error: the octave removed (±600 cents)."""
    return (error_cents + 600.0) % 1200.0 - 600.0


def build(instrument: Instrument, sounds_dir: Path, force: bool) -> bool:
    target = sounds_dir / instrument.id
    if (target / "bank.json").is_file() and not force:
        print(f"{instrument.name}: ya instalado en {target} (usa --force para rehacerlo)")
        return True
    bytes_per_frame = 3 * 2 if instrument.stereo else 2  # VCSL: 24-bit stereo; jRhodes: 16-bit mono
    n_bytes = int(instrument.keep_s * 44100 * bytes_per_frame) + HEADER_SLACK
    tmp = sounds_dir / f".{instrument.id}.part"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    print(f"{instrument.name}: {len(instrument.samples)} muestras (~{len(instrument.samples) * n_bytes / 1e6:.0f} MB)…")

    def one(src: SampleSource) -> dict[str, object]:
        raw = fetch_prefix(src.url, n_bytes)
        y, rate, cents = process(instrument, src, raw)
        write_wav16(tmp / src.cache_name, y, rate)
        return {"file": src.cache_name, "root": src.root, "layer": src.layer, "url": src.url,
                "downloaded_bytes": len(raw), "sha256_downloaded": hashlib.sha256(raw).hexdigest(),
                "sample_rate": rate, "seconds": round(y.shape[0] / rate, 3), "pitch_error_cents": round(cents, 1)}

    with ThreadPoolExecutor(max_workers=6) as pool:
        entries = list(pool.map(one, instrument.samples))
    for e in entries:
        measured = float(e.pop("pitch_error_cents"))  # type: ignore[arg-type]
        e["pitch_class_error_cents"] = round(folded_cents(measured), 1)
        e["octave_agrees"] = abs(measured) <= 600.0
    worst = max(entries, key=lambda e: abs(float(e["pitch_class_error_cents"])))  # type: ignore[arg-type]
    octave_ok = sum(bool(e["octave_agrees"]) for e in entries) / len(entries)
    print(f"  afinación: peor clase de nota {worst['file']} {worst['pitch_class_error_cents']:+} cents; "
          f"octava coincidente en {octave_ok:.0%} de las muestras")
    bad = [e for e in entries if abs(float(e["pitch_class_error_cents"])) > MAX_CENTS]  # type: ignore[arg-type]
    if bad or octave_ok < MIN_OCTAVE_AGREEMENT:
        print(f"  ERROR: muestras que no suenan a su nota: {[(e['file'], e['pitch_class_error_cents']) for e in bad]}"
              f"; octava coincidente {octave_ok:.0%} (mínimo {MIN_OCTAVE_AGREEMENT:.0%})")
        shutil.rmtree(tmp, ignore_errors=True)
        return False
    manifest = {
        "format": BANK_FORMAT, "instrument": instrument.id, "name": instrument.name,
        "recorded": instrument.recorded, "source": instrument.source, "license": instrument.license,
        "attribution": instrument.attribution,
        "processing": f"first {instrument.keep_s:.1f} s from 3 ms before the onset, cosine fade-out over the last "
                      f"{instrument.fade_s:.1f} s, converted to 16-bit PCM" + ("" if instrument.stereo else " (mono)"),
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "samples": entries,
    }
    (tmp / "bank.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if target.exists():
        shutil.rmtree(target)
    tmp.rename(target)
    size = sum(p.stat().st_size for p in target.iterdir())
    print(f"  instalado: {target} ({size / 1e6:.1f} MB)")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", choices=sorted(i for i, inst in INSTRUMENTS.items() if not inst.synthesized),
                        action="append")
    parser.add_argument("--sounds-dir", type=Path, default=DEFAULT_SOUNDS_DIR)
    parser.add_argument("--force", action="store_true", help="vuelve a descargar aunque ya esté instalado")
    args = parser.parse_args()
    ok = True
    for instrument_id in args.only or list(INSTRUMENTS):
        if INSTRUMENTS[instrument_id].synthesized:
            print(f"{INSTRUMENTS[instrument_id].name}: lo genera la app, no hay nada que descargar")
            continue
        try:
            ok &= build(INSTRUMENTS[instrument_id], args.sounds_dir, args.force)
        except OSError as exc:
            print(f"{INSTRUMENTS[instrument_id].name}: error de red o disco: {exc}", file=sys.stderr)
            ok = False
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
