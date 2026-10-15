"""Finger taps on a table, heard by the microphone: WHEN a touch happened (the camera says which finger).

``OnsetDetector`` (pure numpy, testable without a microphone): a tap on a hard
surface is a short broadband click: a SUDDEN jump of energy. The signal is
high-passed (second order: removes hum, voice fundamentals and most of the
table's low thud), its envelope is taken over 2 ms hops, and an onset is
reported when the envelope jumps above ``ratio`` × the larger of the running
noise floor and the mean envelope of the previous ``recent_s`` (so a sustained
or slowly rising sound — a held note, a voice — is not a tap), above an absolute
minimum, at most once per ``refractory_s``.

Our own piano sound reaches the microphone too. After each note we play, the
threshold is raised for ``self_sound_s`` (``note_played``): a real tap is far
sharper than a piano attack heard through the room, but quiet taps during that
window can be missed. Headphones remove the problem.

``MicTapInput`` runs the detector on the microphone's audio thread (``sounddevice``,
already installed) and timestamps each onset on the trace/frame clock
(``time.perf_counter``) using PortAudio's ADC time of the block.

All parameters are PROVISIONAL: chosen from the physics of a tap, not yet measured
on recordings of this setup.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class TapOnset:
    t: float  # perf_counter seconds of the onset (start of the 2 ms hop where it was detected)
    strength: float  # envelope / noise floor at the onset


@dataclass(frozen=True, slots=True)
class OnsetConfig:
    highpass_hz: float = 1200.0
    hop_s: float = 0.002
    ratio: float = 8.0  # envelope must exceed ratio × noise floor
    min_level: float = 0.004  # and this absolute level (full scale = 1.0)
    refractory_s: float = 0.06
    recent_s: float = 0.03
    noise_tau_s: float = 0.4
    self_sound_s: float = 0.15  # after a note we play: threshold raised by self_sound_factor
    self_sound_factor: float = 4.0


class OnsetDetector:
    def __init__(self, sample_rate: int, config: OnsetConfig | None = None) -> None:
        self.rate = sample_rate
        self.config = config or OnsetConfig()
        rc = 1.0 / (2 * np.pi * self.config.highpass_hz)
        dt = 1.0 / sample_rate
        self._alpha = rc / (rc + dt)  # two cascaded first-order high-pass stages
        self._state = [0.0, 0.0, 0.0, 0.0]  # (prev x, prev y) per stage
        self._hop = max(1, round(self.config.hop_s * sample_rate))
        self._recent: deque[float] = deque(maxlen=max(1, round(self.config.recent_s / self.config.hop_s)))
        self._carry = np.zeros(0, dtype=np.float64)
        self._carry_t = 0.0
        self.noise = self.config.min_level / self.config.ratio
        self._last_onset = -1e9
        self._self_sound_until = -1e9
        self._lock = threading.Lock()

    def note_played(self, t: float) -> None:
        """Our own note started at ``t`` (any thread): be deaf-ish to it for a moment."""
        with self._lock:
            self._self_sound_until = max(self._self_sound_until, t + self.config.self_sound_s)

    def _highpass(self, x: np.ndarray) -> np.ndarray:
        a = self._alpha
        for stage in (0, 2):
            y = np.empty_like(x)
            px, py = self._state[stage], self._state[stage + 1]
            for i in range(len(x)):  # short blocks (≤ 1024 samples): a plain loop is fast enough
                py = a * (py + x[i] - px)
                px = x[i]
                y[i] = py
            self._state[stage], self._state[stage + 1] = px, py
            x = y
        return x

    def process(self, block: np.ndarray, t_start: float) -> list[TapOnset]:
        """``block``: mono float samples; ``t_start``: perf_counter time of its first sample."""
        cfg = self.config
        x = self._highpass(np.asarray(block, dtype=np.float64).reshape(-1))
        if len(self._carry):
            t_start = self._carry_t
            x = np.concatenate([self._carry, x])
        n = len(x) // self._hop
        self._carry = x[n * self._hop:]
        self._carry_t = t_start + n * self._hop / self.rate
        if n == 0:
            return []
        env = np.abs(x[: n * self._hop]).reshape(n, self._hop).max(axis=1)
        with self._lock:
            self_sound_until = self._self_sound_until
        onsets: list[TapOnset] = []
        decay = min(1.0, cfg.hop_s / cfg.noise_tau_s)
        for i, e in enumerate(env):
            t = t_start + i * self._hop / self.rate
            ratio = cfg.ratio * (cfg.self_sound_factor if t < self_sound_until else 1.0)
            reference = max(self.noise, sum(self._recent) / len(self._recent)) if self._recent else self.noise
            self._recent.append(float(e))
            if e > ratio * reference and e > cfg.min_level and t - self._last_onset >= cfg.refractory_s:
                onsets.append(TapOnset(t, float(e / max(reference, 1e-9))))
                self._last_onset = t
            elif t - self._last_onset > cfg.refractory_s:  # learn the floor outside taps and their ring
                self.noise += decay * (float(e) - self.noise)
                self.noise = max(self.noise, 1e-6)
        return onsets


class MicTapInput:
    """The microphone, running ``OnsetDetector`` on PortAudio's input thread. Onsets queue up for ``drain``."""

    def __init__(self, sample_rate: int = 44100, block: int = 128, device: Any = None,
                 config: OnsetConfig | None = None) -> None:
        self.detector = OnsetDetector(sample_rate, config)
        self.sample_rate = sample_rate
        self.block = block
        self.device = device
        self._onsets: deque[TapOnset] = deque(maxlen=256)
        self._stream: Any = None
        self.error: str | None = None
        self.overflows = 0

    def start(self) -> bool:
        import sounddevice as sd

        def callback(indata: np.ndarray, frames: int, time_info: Any, status: Any) -> None:
            now = time.perf_counter()
            if status.input_overflow:
                self.overflows += 1
            try:  # age of the first sample of the block, from PortAudio's own clock
                age = max(0.0, float(time_info.currentTime - time_info.inputBufferAdcTime))
            except (AttributeError, TypeError):
                age = frames / self.sample_rate
            for onset in self.detector.process(indata[:, 0], now - age):
                self._onsets.append(onset)

        try:
            self._stream = sd.InputStream(samplerate=self.sample_rate, blocksize=self.block, channels=1,
                                          dtype="float32", latency="low", callback=callback, device=self.device)
            self._stream.start()
        except Exception as exc:  # no microphone / permission denied: the app keeps working without it
            self.error = f"Micrófono no disponible: {exc}"
            log.error(self.error)
            self._stream = None
            return False
        return True

    def drain(self) -> list[TapOnset]:
        out = []
        while self._onsets:
            try:
                out.append(self._onsets.popleft())
            except IndexError:
                break
        return out

    def note_played(self, t: float) -> None:
        self.detector.note_played(t)

    def close(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
