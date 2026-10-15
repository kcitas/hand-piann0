"""AudioEngine: the only audio API the rest of the app uses.

    engine.note_on(midi, velocity)      engine.note_off(midi)
    engine.set_instrument("grand")      engine.set_volume(0.0 … 1.0)
    engine.all_notes_off()              engine.close()

Threads
-------
* callers (UI thread, tracking thread, tests): every method only enqueues a
  command or submits a load; nothing blocks on audio or disk;
* the loader thread reads sample banks from disk (``set_instrument`` of a bank
  not yet in memory returns at once; the switch happens when it is loaded);
* the output thread (PortAudio's callback, via ``sounddevice``) renders blocks.

The output is pluggable: ``SoundDeviceOutput`` for the real device, or
``NullOutput`` (tests, no audio device): the app keeps working silently and
reports why. ``sounddevice`` is already installed as a MediaPipe dependency; no
new package is needed.
"""

from __future__ import annotations

import functools
import logging
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from handpiano.audio.instruments import INSTRUMENTS
from handpiano.audio.mixer import Mixer
from handpiano.audio.soundbank import SoundBank, SoundBankError, load_bank

log = logging.getLogger(__name__)

DEFAULT_SOUNDS_DIR = Path(__file__).resolve().parents[2] / "sounds"
SAMPLE_RATE = 44100
BLOCK_FRAMES = 256  # 5.8 ms at 44.1 kHz


class AudioOutput(Protocol):
    def start(self, render: Callable[[int], np.ndarray]) -> None: ...
    def close(self) -> None: ...
    @property
    def description(self) -> str: ...


@dataclass(slots=True)
class OutputStats:
    callbacks: int = 0
    underflows: int = 0  # PortAudio reported the device ran out of data (audible glitch)
    max_callback_ms: float = 0.0
    # PortAudio's own estimate of output latency (buffer → DAC), seconds. NOT a measurement of
    # the sound reaching the ear; see docs/AUDIO.md for how to measure that.
    reported_latency_s: float | None = None
    dac_lead_s: float | None = None  # outputBufferDacTime − currentTime of the last callback


class NullOutput:
    """No device: nothing is rendered unless a test calls ``render`` itself."""

    def __init__(self, reason: str = "salida de audio desactivada") -> None:
        self.reason = reason
        self.stats = OutputStats()

    def start(self, render: Callable[[int], np.ndarray]) -> None:
        self.render = render

    def close(self) -> None:
        pass

    @property
    def description(self) -> str:
        return self.reason


class SoundDeviceOutput:
    def __init__(self, sample_rate: int = SAMPLE_RATE, block: int = BLOCK_FRAMES, device: Any = None) -> None:
        self.sample_rate = sample_rate
        self.block = block
        self.device = device
        self.stats = OutputStats()
        self._stream: Any = None
        self._name = "?"

    def start(self, render: Callable[[int], np.ndarray]) -> None:
        import sounddevice as sd  # lazy: importing PortAudio is slow and tests do not need it

        stats = self.stats

        def callback(outdata: np.ndarray, frames: int, time_info: Any, status: Any) -> None:
            t0 = time.perf_counter()
            if status.output_underflow:
                stats.underflows += 1
            try:
                outdata[:] = render(frames)
            except Exception:  # never let an exception kill the audio thread silently
                outdata.fill(0)
                log.exception("Audio render failed")
            stats.callbacks += 1
            stats.max_callback_ms = max(stats.max_callback_ms, (time.perf_counter() - t0) * 1000)
            try:
                stats.dac_lead_s = float(time_info.outputBufferDacTime - time_info.currentTime)
            except (AttributeError, TypeError):
                pass

        self._stream = sd.OutputStream(samplerate=self.sample_rate, blocksize=self.block, channels=2,
                                       dtype="float32", latency="low", callback=callback, device=self.device)
        self._stream.start()
        self.stats.reported_latency_s = float(self._stream.latency)
        info = sd.query_devices(self._stream.device, "output")
        self._name = str(info["name"]) if isinstance(info, dict) else str(self._stream.device)

    def close(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    @property
    def description(self) -> str:
        lat = self.stats.reported_latency_s
        return f"{self._name} · {self.sample_rate} Hz · bloque {self.block}" + (
            f" · latencia PortAudio {lat * 1000:.0f} ms" if lat is not None else "")


@dataclass(frozen=True, slots=True)
class EngineStatus:
    instrument: str  # requested instrument id
    ready: bool  # its bank is loaded and in use
    message: str  # human-readable state (loading, error, ok)
    volume: float
    output: str
    voices: int


class AudioEngine:
    def __init__(self, sounds_dir: Path = DEFAULT_SOUNDS_DIR, output: AudioOutput | None = None,
                 sample_rate: int = SAMPLE_RATE, volume: float = 0.8, max_voices: int = 48) -> None:
        self.sounds_dir = sounds_dir
        self.output: AudioOutput = output if output is not None else NullOutput()
        self.mixer = Mixer(sample_rate=sample_rate, max_voices=max_voices, volume=volume)
        self._volume = volume
        self._lock = threading.Lock()  # protects the fields below (never held by the audio thread)
        self._banks: dict[str, SoundBank] = {}
        self._errors: dict[str, str] = {}
        self._loading: dict[str, Future[SoundBank]] = {}
        self._requested = "grand"
        self._active: str | None = None
        self._loader = ThreadPoolExecutor(max_workers=1, thread_name_prefix="audio-loader")
        self._started = False
        self._output_error: str | None = None
        # Called (on the loader thread) when an instrument becomes active or fails. Must be fast.
        self.on_change: Callable[[EngineStatus], None] | None = None
        # Called with time.perf_counter() on every accepted note_on, from the caller's thread
        # (e.g. the microphone ignores our own notes for a moment). Must be fast and never raise.
        self.note_listeners: list[Callable[[float], None]] = []

    # -- lifecycle ------------------------------------------------------------------------

    def start(self) -> None:
        if self._started:
            return
        self._started = True
        try:
            self.output.start(self.mixer.render)
        except Exception as exc:  # no device, permission, PortAudio error: keep the app running silently
            self._output_error = f"Sin salida de audio: {exc}"
            log.error(self._output_error)
            self.output = NullOutput(self._output_error)

    def close(self) -> None:
        self.output.close()
        self._loader.shutdown(wait=False, cancel_futures=True)

    # -- commands (non-blocking, any thread) ----------------------------------------------

    def note_on(self, midi: int, velocity: int = 100, tag: int | None = None) -> bool:
        """Queues a note. True = accepted into the audio queue (NOT proof that it was heard: what the
        audio thread did with it is in ``mixer.outcomes`` when a ``tag`` is given)."""
        if not (0 <= midi <= 127 and velocity > 0):
            return False
        if tag is None:
            self.mixer.post("on", int(midi), int(min(127, velocity)))
        else:
            self.mixer.post("on", int(midi), int(min(127, velocity)), tag)
        if self.note_listeners:
            now = time.perf_counter()
            for listener in self.note_listeners:
                listener(now)
        return True

    def note_off(self, midi: int, tag: int | None = None) -> bool:
        if tag is None:
            self.mixer.post("off", int(midi))
        else:
            self.mixer.post("off", int(midi), tag)
        return True

    def all_notes_off(self) -> None:
        self.mixer.post("all_off")

    def set_volume(self, value: float) -> None:
        self._volume = float(min(1.0, max(0.0, value)))
        self.mixer.post("volume", self._volume)

    @property
    def volume(self) -> float:
        return self._volume

    def set_instrument(self, instrument_id: str) -> None:
        """Switches instrument; loads it in the background if needed. Unknown ids raise KeyError."""
        if instrument_id not in INSTRUMENTS:
            raise KeyError(instrument_id)
        with self._lock:
            self._requested = instrument_id
            bank = self._banks.get(instrument_id)
        if bank is not None:
            self._activate(instrument_id, bank)
        else:
            self._load(instrument_id)

    def preload(self, ids: tuple[str, ...] = tuple(INSTRUMENTS)) -> None:
        """Loads instruments in the background so that later switches are instant."""
        for i in ids:
            self._load(i)

    def _load(self, instrument_id: str) -> Future[SoundBank]:
        with self._lock:
            existing = self._loading.get(instrument_id)
            if existing is not None:
                return existing
            future = self._loader.submit(load_bank, self.sounds_dir, instrument_id)
            self._loading[instrument_id] = future
        future.add_done_callback(functools.partial(self._loaded, instrument_id))
        return future

    def _loaded(self, instrument_id: str, future: Future[SoundBank]) -> None:
        try:
            bank = future.result()
        except SoundBankError as exc:
            with self._lock:
                self._errors[instrument_id] = str(exc)
                self._loading.pop(instrument_id, None)
            log.error("Sound bank %s: %s", instrument_id, exc)
            self._notify()
            return
        except Exception as exc:  # cancelled or unexpected
            with self._lock:
                self._errors[instrument_id] = f"error al cargar: {exc}"
                self._loading.pop(instrument_id, None)
            self._notify()
            return
        with self._lock:
            self._banks[instrument_id] = bank
            self._errors.pop(instrument_id, None)
            self._loading.pop(instrument_id, None)
            requested = self._requested == instrument_id
        log.info("Sound bank %s loaded (%d MB)", instrument_id, bank.memory_bytes // 1_000_000)
        if requested:
            self._activate(instrument_id, bank)
        else:
            self._notify()

    def _activate(self, instrument_id: str, bank: SoundBank) -> None:
        self.mixer.post("bank", bank)
        with self._lock:
            self._active = instrument_id
        self._notify()

    def wait_ready(self, timeout: float = 10.0) -> bool:
        """Blocks until the requested instrument is active or failed (scripts and tests only)."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            status = self.status()
            if status.ready or self._errors.get(status.instrument):
                return status.ready
            time.sleep(0.01)
        return False

    def _notify(self) -> None:
        callback = self.on_change
        if callback is not None:
            try:
                callback(self.status())
            except Exception:
                log.exception("on_change callback failed")

    # -- state ----------------------------------------------------------------------------

    def is_loaded(self, instrument_id: str) -> bool:
        with self._lock:
            return instrument_id in self._banks

    def error(self, instrument_id: str) -> str | None:
        with self._lock:
            return self._errors.get(instrument_id)

    def status(self) -> EngineStatus:
        with self._lock:
            requested, active = self._requested, self._active
            error = self._errors.get(requested)
            loading = requested in self._loading
        name = INSTRUMENTS[requested].name
        ready = active == requested
        if error:
            message = error
        elif ready:
            message = f"{name} listo"
        elif loading:
            message = f"Cargando {name}…"
        else:
            message = f"{name} sin cargar"
        return EngineStatus(requested, ready, message, self._volume, self.output.description,
                            len(self.mixer.voices))
