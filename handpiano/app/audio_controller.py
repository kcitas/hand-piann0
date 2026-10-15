"""Binds the AudioEngine to the instrument panel and to keyboard/mouse note input (Qt main thread).

* panel card click / keys 1–3 → ``set_instrument``; the active card follows the
  ENGINE's state (loading, ready, error), not the click;
* "▶ Nota de prueba" selects that instrument and plays C4 for one second;
* volume slider / keys − and + → ``set_volume``;
* ``NoteKeys``: computer-keyboard note layout (two rows, like a DAW), so the
  instruments can be played and tested without a camera.
"""

from __future__ import annotations

from PySide6.QtCore import QObject, QTimer
from shiboken6 import isValid

from handpiano.audio.engine import AudioEngine
from handpiano.audio.instruments import INSTRUMENTS
from handpiano.ui.instrument_panel import InstrumentPanel

TEST_NOTE = 60
TEST_VELOCITY = 96
TEST_DURATION_MS = 1000
INSTRUMENT_KEYS = {str(i): instrument_id for i, instrument_id in enumerate(INSTRUMENTS, start=1)}

# Semitones above the base note for each key (A = base). "Ñ" (ES layout) and ";" (US) are the same key.
NOTE_KEY_OFFSETS = {"A": 0, "W": 1, "S": 2, "E": 3, "D": 4, "F": 5, "T": 6, "G": 7, "Y": 8, "H": 9, "U": 10,
                    "J": 11, "K": 12, "O": 13, "L": 14, "P": 15, "Ñ": 16, ";": 16}


class NoteKeys:
    """Computer keys → MIDI notes, with octave shift (Z / X). Tracks held keys so a key
    held down (auto-repeat) or released after an octave change ends the RIGHT note."""

    def __init__(self, base: int = 60) -> None:
        self.base = base
        self._held: dict[str, int] = {}

    def press(self, key: str) -> int | None:
        key = key.upper()
        if key == "Z":
            self.base = max(24, self.base - 12)
            return None
        if key == "X":
            self.base = min(96, self.base + 12)
            return None
        offset = NOTE_KEY_OFFSETS.get(key)
        if offset is None or key in self._held:
            return None
        midi = self.base + offset
        self._held[key] = midi
        return midi

    def release(self, key: str) -> int | None:
        return self._held.pop(key.upper(), None)

    def release_all(self) -> list[int]:
        notes = list(self._held.values())
        self._held.clear()
        return notes


class AudioController(QObject):
    def __init__(self, engine: AudioEngine, panel: InstrumentPanel, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.engine = engine
        self.panel = panel
        panel.instrument_selected.connect(self.select)
        panel.test_requested.connect(self.test_note)
        panel.volume_changed.connect(engine.set_volume)
        panel.set_volume(engine.volume)
        # The engine reports from its loader thread; the panel is refreshed from here instead.
        self._stopped = False
        self._timer = QTimer(self)
        self._timer.setInterval(150)
        self._timer.timeout.connect(self.refresh)
        self._timer.start()
        self._test_pending: list[int] = []
        self.refresh()

    def stop(self) -> None:
        """No more panel refreshes (the window that owns the panel is closing)."""
        self._stopped = True
        self._timer.stop()

    def select(self, instrument_id: str) -> None:
        self.engine.set_instrument(instrument_id)
        self.refresh()

    def test_note(self, instrument_id: str) -> None:
        self.engine.all_notes_off()
        self.select(instrument_id)
        if self.engine.status().ready:
            self._play_test()
        else:  # still loading: play as soon as it is ready
            self._test_pending.append(TEST_NOTE)

    def _play_test(self) -> None:
        self.engine.note_on(TEST_NOTE, TEST_VELOCITY)
        QTimer.singleShot(TEST_DURATION_MS, lambda: self.engine.note_off(TEST_NOTE))

    def change_volume(self, delta: float) -> None:
        self.engine.set_volume(self.engine.volume + delta)
        self.panel.set_volume(self.engine.volume)

    def handle_shortcut(self, key: str) -> bool:
        """Keys 1–3 (instrument), − / + (volume). Returns True if handled."""
        if key in INSTRUMENT_KEYS:
            self.select(INSTRUMENT_KEYS[key])
            return True
        if key in ("-", "_"):
            self.change_volume(-0.1)
            return True
        if key in ("+", "="):
            self.change_volume(0.1)
            return True
        return False

    def refresh(self) -> None:
        # The panel can be deleted before this controller (a mode closed from the start menu).
        if self._stopped or not isValid(self.panel):
            self.stop()
            return
        status = self.engine.status()
        for instrument_id in INSTRUMENTS:
            error = self.engine.error(instrument_id)
            if error:
                text = "⚠ " + error
            elif instrument_id == status.instrument and status.ready:
                text = "● sonando por este instrumento"
            elif self.engine.is_loaded(instrument_id):
                text = "listo"
            elif instrument_id == status.instrument:
                text = "cargando muestras…"
            else:
                text = "sin cargar"
            self.panel.set_card_state(instrument_id, text)
        self.panel.set_active(status.instrument if status.ready else "")
        self.panel.set_output(status.output)
        if self._test_pending and status.ready:
            self._test_pending.clear()
            self._play_test()
