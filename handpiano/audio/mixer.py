"""Polyphonic sample mixer (pure numpy, no audio device).

``render(frames)`` produces the next block of stereo float32 audio. It runs on
the audio callback thread; every other thread only ENQUEUES commands
(``collections.deque.append`` is atomic in CPython), so ``note_on`` never waits
for the audio thread and the audio thread never waits for a lock.

Per voice: linear-interpolated playback of an int16 sample at
``ratio × sample_rate / output_rate``, a gain from the velocity, and on note-off
an exponential release reaching −60 dB after the instrument's ``release_s``.
Re-striking a sounding note quickly releases the previous voice (30 ms), like a
re-struck string. When ``max_voices`` is exceeded the oldest releasing voice is
dropped first, then the oldest one. The master volume is ramped over each block
(no zipper clicks) and the sum goes through a ``tanh`` soft limiter so chords
cannot clip harshly.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from handpiano.audio.soundbank import SoundBank

RETRIGGER_RELEASE_S = 0.03
SILENT = 1e-4  # −80 dB: a releasing voice below this is removed


@dataclass(slots=True)
class Voice:
    midi: int
    data: np.ndarray  # int16 (frames, channels)
    step: float  # source frames per output frame
    gain: float
    release_s: float
    started: int  # output frame counter at note-on (for stealing)
    pos: float = 0.0
    env: float = 1.0
    releasing: bool = False
    release_mult: float = 1.0  # per-frame envelope multiplier while releasing
    done: bool = False

    def release(self, seconds: float, rate: int) -> None:
        """Starts (or shortens) the release: −60 dB after ``seconds``."""
        if not self.releasing or seconds < self.release_s:
            self.releasing = True
            self.release_s = seconds
            self.release_mult = 10.0 ** (-3.0 / max(1.0, seconds * rate))

    def render(self, out: np.ndarray) -> None:
        """Adds this voice into ``out`` (frames, 2)."""
        n = out.shape[0]
        last = self.data.shape[0] - 1
        idx = self.pos + self.step * np.arange(n)
        valid = int(np.searchsorted(idx, last, side="left"))
        if valid <= 0:
            self.done = True
            return
        idx = idx[:valid]
        i0 = idx.astype(np.int64)
        frac = (idx - i0).astype(np.float32)[:, None]
        a = self.data[i0].astype(np.float32)
        b = self.data[i0 + 1].astype(np.float32)
        s = (a + (b - a) * frac) * (self.gain / 32768.0)
        if self.releasing:
            env = self.env * self.release_mult ** np.arange(1, valid + 1, dtype=np.float64)
            s *= env.astype(np.float32)[:, None]
            self.env = float(env[-1])
            if self.env < SILENT:
                self.done = True
        if s.shape[1] == 1:
            out[:valid] += s
        else:
            out[:valid] += s[:, :2]
        self.pos += self.step * n
        if valid < n:
            self.done = True


@dataclass(frozen=True, slots=True)
class NoteOutcome:
    """What the audio thread did with one tagged command (for diagnostics; written by the audio thread)."""

    tag: int
    kind: str  # "on" / "off"
    midi: int
    instrument: str | None  # bank in use when applied; None = no bank loaded
    result: str  # "voz iniciada", "sin banco: nota descartada", "voces liberadas: N"
    t: float  # time.perf_counter() when applied


@dataclass(slots=True)
class MixerStats:
    blocks: int = 0
    frames: int = 0
    note_ons: int = 0
    note_offs: int = 0
    stolen: int = 0
    max_voices_seen: int = 0
    clipped_blocks: int = 0
    # note_on/off call → applied by the audio thread (at the start of the next block). Software
    # queueing only: the device buffer and the DAC come after this.
    last_command_delay_ms: float = 0.0
    max_command_delay_ms: float = 0.0


@dataclass
class Mixer:
    sample_rate: int = 44100
    max_voices: int = 48
    voices: list[Voice] = field(default_factory=list)
    bank: SoundBank | None = None
    volume: float = 0.8
    stats: MixerStats = field(default_factory=MixerStats)

    def __post_init__(self) -> None:
        self._commands: deque[tuple[tuple[Any, ...], float]] = deque()
        self._target_volume = self.volume
        self._clock = 0
        # Outcomes of tagged commands, newest last (bounded; read by the UI thread).
        self.outcomes: deque[NoteOutcome] = deque(maxlen=1024)

    # -- called from any thread (non-blocking) ----------------------------------------

    def post(self, *command: Any) -> None:
        self._commands.append((command, time.perf_counter()))

    @property
    def pending(self) -> list[tuple[Any, ...]]:
        """Commands not yet applied (for tests and diagnostics)."""
        return [c for c, _ in list(self._commands)]

    # -- audio thread ---------------------------------------------------------------

    def _apply(self, command: tuple[Any, ...]) -> None:
        kind = command[0]
        tag = command[3] if kind == "on" and len(command) > 3 else command[2] if kind == "off" and len(command) > 2 \
            else None
        if kind == "on":
            started = self._note_on(command[1], command[2])
            if tag is not None:
                self.outcomes.append(NoteOutcome(
                    tag, "on", command[1], None if self.bank is None else self.bank.instrument.id,
                    "voz iniciada" if started else "sin banco: nota descartada", time.perf_counter()))
        elif kind == "off":
            self.stats.note_offs += 1
            released = 0
            for v in self.voices:
                if v.midi == command[1] and not v.releasing:
                    v.release(v.release_s, self.sample_rate)
                    released += 1
            if tag is not None:
                self.outcomes.append(NoteOutcome(
                    tag, "off", command[1], None if self.bank is None else self.bank.instrument.id,
                    f"voces liberadas: {released}", time.perf_counter()))
        elif kind == "all_off":
            for v in self.voices:
                v.release(min(v.release_s, 0.1), self.sample_rate)
        elif kind == "bank":
            self.bank = command[1]
        elif kind == "volume":
            self._target_volume = float(command[1])

    def _note_on(self, midi: int, velocity: int) -> bool:
        bank = self.bank
        if bank is None:
            return False
        self.stats.note_ons += 1
        for v in self.voices:
            if v.midi == midi:
                v.release(RETRIGGER_RELEASE_S, self.sample_rate)
        region = bank.region(midi, velocity)
        step = region.ratio * region.sample.sample_rate / self.sample_rate
        self.voices.append(Voice(midi, region.sample.data, step, region.gain, bank.instrument.release_s, self._clock))
        while len(self.voices) > self.max_voices:
            releasing = [v for v in self.voices if v.releasing]
            victim = min(releasing or self.voices, key=lambda v: v.started)
            self.voices.remove(victim)
            self.stats.stolen += 1
        self.stats.max_voices_seen = max(self.stats.max_voices_seen, len(self.voices))
        return True

    def render(self, frames: int) -> np.ndarray:
        """Next block, float32 (frames, 2) in [-1, 1]."""
        now = time.perf_counter()
        while self._commands:
            try:
                command, posted = self._commands.popleft()
            except IndexError:
                break
            if command[0] in ("on", "off"):
                delay = (now - posted) * 1000
                self.stats.last_command_delay_ms = delay
                self.stats.max_command_delay_ms = max(self.stats.max_command_delay_ms, delay)
            self._apply(command)
        out = np.zeros((frames, 2), dtype=np.float32)
        for v in self.voices:
            v.render(out)
        self.voices = [v for v in self.voices if not v.done]
        start, end = self.volume, self._target_volume
        if start != end:
            out *= np.linspace(start, end, frames, dtype=np.float32)[:, None]
            self.volume = end
        else:
            out *= start
        if float(np.max(np.abs(out), initial=0.0)) > 0.9:
            self.stats.clipped_blocks += 1
        np.tanh(out, out=out)
        self._clock += frames
        self.stats.blocks += 1
        self.stats.frames += frames
        return out

    @property
    def active_notes(self) -> list[int]:
        return sorted({v.midi for v in self.voices if not v.releasing})
