"""A loaded instrument: its samples in memory and the note/velocity → sample mapping.

Samples come from the local cache written by ``scripts/download_sounds.py``:

    <sounds dir>/<instrument id>/bank.json      what was converted, from where, at which rate
    <sounds dir>/<instrument id>/<root>_v<layer>.wav   16-bit PCM, trimmed

Samples are kept as int16 (half the memory of float32; about 50 MB for each
acoustic piano and 16 MB for the electric one) and converted per audio block.
A missing, incomplete or corrupt bank raises ``SoundBankError`` with a message
the UI can show; nothing is played from a bank that failed validation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from handpiano.audio.instruments import INSTRUMENTS, Instrument
from handpiano.audio.synth import SAMPLE_RATE as SYNTH_RATE
from handpiano.audio.synth import piano_note
from handpiano.audio.wavio import WavError, read_wav

BANK_FORMAT = 1
DOWNLOAD_HINT = "ejecuta 'python scripts/download_sounds.py' una vez (requiere Internet solo esa vez)"


class SoundBankError(Exception):
    pass


@dataclass(frozen=True, slots=True)
class Sample:
    root: int
    layer: int
    data: np.ndarray  # int16, (frames, channels), read-only
    sample_rate: int


@dataclass(frozen=True, slots=True)
class Region:
    """What to play for one note at one velocity."""

    sample: Sample
    ratio: float  # playback speed relative to the recording (pitch shift), before rate conversion
    gain: float


class SoundBank:
    def __init__(self, instrument: Instrument, samples: list[Sample]) -> None:
        if not samples:
            raise SoundBankError(f"{instrument.name}: sin muestras")
        self.instrument = instrument
        self.sample_rate = samples[0].sample_rate
        if any(s.sample_rate != self.sample_rate for s in samples):
            raise SoundBankError(f"{instrument.name}: muestras con distintas frecuencias de muestreo")
        self._by_layer: dict[int, list[Sample]] = {}
        for s in samples:
            self._by_layer.setdefault(s.layer, []).append(s)
        for layer in range(instrument.layers):
            if layer not in self._by_layer:
                raise SoundBankError(f"{instrument.name}: falta la capa de velocidad {layer}")
            self._by_layer[layer].sort(key=lambda s: s.root)
        self.roots = sorted({s.root for s in samples})

    @property
    def memory_bytes(self) -> int:
        return sum(s.data.nbytes for ss in self._by_layer.values() for s in ss)

    def layer_for(self, velocity: int) -> int:
        for layer, top in enumerate(self.instrument.layer_max_velocity):
            if velocity <= top:
                return layer
        return self.instrument.layers - 1

    def region(self, midi: int, velocity: int) -> Region:
        """Nearest recorded root in the velocity's layer (ties: the lower root, shifted up)."""
        velocity = max(1, min(127, int(velocity)))
        candidates = self._by_layer[self.layer_for(velocity)]
        sample = min(candidates, key=lambda s: (abs(s.root - midi), s.root))
        ratio = 2.0 ** ((midi - sample.root) / 12.0)
        gain = self.instrument.gain * (velocity / 127.0) ** self.instrument.velocity_curve
        return Region(sample, ratio, gain)


def synthesize_bank(instrument: Instrument) -> SoundBank:
    """A generated instrument: one note per sample root, computed in memory (no files)."""
    samples = []
    for src in instrument.samples:
        data = np.ascontiguousarray((piano_note(src.root) * 32767.0).astype(np.int16)[:, None])
        data.setflags(write=False)
        samples.append(Sample(src.root, src.layer, data, SYNTH_RATE))
    return SoundBank(instrument, samples)


def bank_dir(sounds_dir: Path, instrument_id: str) -> Path:
    return sounds_dir / instrument_id


def load_bank(sounds_dir: Path, instrument_id: str) -> SoundBank:
    """Loads and validates one instrument from the cache. Raises SoundBankError."""
    instrument = INSTRUMENTS.get(instrument_id)
    if instrument is None:
        raise SoundBankError(f"instrumento desconocido: {instrument_id!r}")
    if instrument.synthesized:
        return synthesize_bank(instrument)
    folder = bank_dir(sounds_dir, instrument_id)
    manifest_path = folder / "bank.json"
    if not manifest_path.is_file():
        raise SoundBankError(f"{instrument.name}: muestras no descargadas ({folder}); {DOWNLOAD_HINT}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        entries = manifest["samples"]
        if manifest.get("format") != BANK_FORMAT or manifest.get("instrument") != instrument_id:
            raise SoundBankError(f"{instrument.name}: bank.json de otra versión o instrumento; {DOWNLOAD_HINT}")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise SoundBankError(f"{instrument.name}: bank.json ilegible ({exc}); {DOWNLOAD_HINT}") from exc
    expected = {s.cache_name for s in instrument.samples}
    listed = {e.get("file") for e in entries if isinstance(e, dict)}
    if missing := expected - listed:
        raise SoundBankError(f"{instrument.name}: faltan {len(missing)} muestras en bank.json; {DOWNLOAD_HINT}")
    samples: list[Sample] = []
    channels = 2 if instrument.stereo else 1
    for src in instrument.samples:
        try:
            wav = read_wav(folder / src.cache_name)
        except WavError as exc:
            raise SoundBankError(f"{instrument.name}: muestra dañada o ausente ({exc}); {DOWNLOAD_HINT}") from exc
        if wav.channels != channels or wav.bits != 16 or wav.duration_s < 0.2:
            raise SoundBankError(f"{instrument.name}: {src.cache_name} no tiene el formato esperado")
        data = np.ascontiguousarray((wav.samples * 32768.0).clip(-32768, 32767).astype(np.int16))
        data.setflags(write=False)
        samples.append(Sample(src.root, src.layer, data, wav.sample_rate))
    return SoundBank(instrument, samples)
