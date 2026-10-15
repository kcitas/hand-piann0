"""Detected presses → notes (Free Piano mode). Runs on the tracking thread, right after the detector.

Rules
-----
* Every ``press`` EVENT strikes its key: one ``note_on`` per event, never one per frame
  (holding a finger down produces no further events).
* A key struck while another finger already holds it is struck AGAIN (like re-striking a
  piano key). Measured on teclado_1–4: 35 of 555 presses were silent because a neighbouring
  finger was stuck in PRESSED on the same key and the previous router only counted a second
  "holder" without playing.
* ``note_off`` is sent when the LAST finger holding the key releases (the key stays lit, and
  sounding, while any finger holds it).
* Every frame the fingers holding notes are reconciled with the fingers the detector reports
  as pressed, so a release event that never comes (calibration removed mid-press, tracking
  lost) cannot leave a note sounding.
* An error from the audio side is recorded and never propagates: the pipeline would otherwise
  disable this listener for the whole session while the keys keep lighting up.

Diagnostics
-----------
Each press gets an id (``NoteRecord``) followed through: detector event → note_on sent →
accepted by the audio queue → what the audio thread did (``Mixer.outcomes``, matched by id)
→ note_off. "Voice started" means the mixer started playing the sample; it does NOT prove
that sound reached the speakers (device, volume and the room are outside the app).

Velocity: the detector's downward speed at the press (hand sizes per second) mapped linearly
to MIDI 40–120. PROVISIONAL: not calibrated against measured key velocities.
"""

from __future__ import annotations

import itertools
import math
import threading
import time
from collections import deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from handpiano.interaction.press_detector import PressEvent


class NoteSink(Protocol):
    def note_on(self, midi: int, velocity: int = 100, tag: int | None = None) -> bool: ...
    def note_off(self, midi: int, tag: int | None = None) -> bool: ...


def velocity_from_press(speed: float | None) -> int:
    if speed is None or math.isnan(speed):
        return 90
    return int(max(40, min(120, round(50 + 18 * max(-10.0, min(10.0, speed))))))


@dataclass
class NoteRecord:
    id: int
    t: float  # detector time of the press event (frame capture, perf_counter seconds)
    finger: str
    midi: int
    velocity: int
    restrike: bool  # another finger already held this key
    note_on: str = "pendiente"  # "encolada", "rechazada", "error: …"
    note_off: str = "pendiente"  # "encolada", "otro dedo sostiene la tecla", "error: …"
    released_t: float | None = None
    release_reason: str = ""  # "release:ok", "release:tracking_lost", "reconciliación"


class PressSound:
    def __init__(self, sink: NoteSink) -> None:
        self.sink = sink
        self.enabled = True
        self._held: dict[Any, tuple[int, int]] = {}  # finger → (midi, record id)
        self._lock = threading.Lock()  # on_frame (tracking thread) vs set_enabled / stop_all (UI thread)
        self._ids = itertools.count(1)
        self.records: deque[NoteRecord] = deque(maxlen=512)
        self._by_id: dict[int, NoteRecord] = {}
        self.errors = 0

    # -- audio calls that never raise --------------------------------------------------

    def _call(self, fn: Any, *args: Any, **kwargs: Any) -> str:
        try:
            accepted = fn(*args, **kwargs)
        except Exception as exc:  # the audio side must never break key state or the pipeline
            self.errors += 1
            return f"error: {type(exc).__name__}: {exc}"
        return "rechazada" if accepted is False else "encolada"

    def _record(self, record: NoteRecord) -> None:
        if len(self.records) == self.records.maxlen:
            self._by_id.pop(self.records[0].id, None)
        self.records.append(record)
        self._by_id[record.id] = record

    def _release(self, finger: Any, reason: str, t: float) -> None:
        midi, record_id = self._held.pop(finger)
        record = self._by_id.get(record_id)
        if record is not None:
            record.released_t, record.release_reason = t, reason
        if any(m == midi for m, _ in self._held.values()):
            if record is not None:
                record.note_off = "no enviada: otro dedo sostiene la tecla"
            return
        result = self._call(self.sink.note_off, midi, tag=record_id)
        if record is not None:
            record.note_off = result

    # -- per frame ------------------------------------------------------------------------

    def on_frame(self, events: Iterable[PressEvent], pressed_keys: Mapping[int, Iterable[Any]] | None) -> None:
        with self._lock:
            if not self.enabled:
                return
            now = time.perf_counter()
            for e in events:
                if e.kind == "press" and e.key is not None:
                    if e.finger in self._held:  # not expected from the detector; keep state coherent
                        self._release(e.finger, "nueva pulsación del mismo dedo", e.t)
                    midi = e.key.midi
                    restrike = any(m == midi for m, _ in self._held.values())
                    record = NoteRecord(next(self._ids), e.t, getattr(e.finger, "name", str(e.finger)), midi,
                                        velocity_from_press(e.velocity), restrike)
                    self._record(record)
                    self._held[e.finger] = (midi, record.id)
                    record.note_on = self._call(self.sink.note_on, midi, record.velocity, tag=record.id)
                elif e.kind == "release" and e.finger in self._held:
                    self._release(e.finger, f"release:{e.reason}", e.t)
            if pressed_keys is not None:
                holding = {f for fingers in pressed_keys.values() for f in fingers}
                for finger in [f for f in self._held if f not in holding]:
                    self._release(finger, "reconciliación (el detector ya no la tiene pulsada)", now)

    def stop_all(self) -> None:
        with self._lock:
            now = time.perf_counter()
            for finger in list(self._held):
                self._release(finger, "parada", now)

    def set_enabled(self, enabled: bool) -> None:
        if not enabled:
            self.stop_all()
        with self._lock:
            self.enabled = enabled

    @property
    def sounding(self) -> list[int]:
        with self._lock:
            return sorted({m for m, _ in self._held.values()})


def outcomes_by_tag(outcomes: Iterable[Any]) -> dict[tuple[int, str], Any]:
    """Mixer outcomes indexed by (tag, "on"/"off")."""
    return {(o.tag, o.kind): o for o in list(outcomes)}


def describe(record: NoteRecord, outcomes: Mapping[tuple[int, str], Any]) -> str:
    """One line per press: detector → note_on → audio thread → note_off."""
    on = outcomes.get((record.id, "on"))
    off = outcomes.get((record.id, "off"))
    engine = "motor: sin respuesta todavía" if on is None else (
        f"motor: {on.result} ({on.instrument or 'ningún instrumento'}, +{(on.t - record.t) * 1000:.0f} ms "
        f"desde la captura)")
    release = "" if record.released_t is None else (
        f" → liberada a los {record.released_t - record.t:.2f} s ({record.release_reason}) → note_off "
        f"{record.note_off}" + ("" if off is None else f" → {off.result}"))
    return (f"[nota #{record.id}] {record.finger} → MIDI {record.midi} vel {record.velocity}"
            f"{' (re-golpe: otro dedo la sostenía)' if record.restrike else ''} → note_on {record.note_on} → "
            f"{engine}{release}")


def summary(records: Iterable[NoteRecord], outcomes: Mapping[tuple[int, str], Any]) -> dict[str, int]:
    counts = {"pulsaciones": 0, "note_on encoladas": 0, "voz iniciada": 0, "sin banco": 0, "errores": 0,
              "liberadas": 0, "note_off encoladas": 0}
    for r in records:
        counts["pulsaciones"] += 1
        counts["note_on encoladas"] += r.note_on == "encolada"
        counts["errores"] += r.note_on.startswith("error") or r.note_off.startswith("error")
        on = outcomes.get((r.id, "on"))
        if on is not None:
            counts["voz iniciada"] += on.result == "voz iniciada"
            counts["sin banco"] += on.result.startswith("sin banco")
        counts["liberadas"] += r.released_t is not None
        counts["note_off encoladas"] += r.note_off == "encolada"
    return counts
