"""``python -m handpiano --audio-test``: the three instruments without camera (keyboard and mouse).

Keys: A W S E D F T G Y H U J K O L P Ñ play notes (two rows, like a DAW);
Z / X octave down / up; 1 2 3 instrument; − / + volume. Several keys at once
play chords. Mouse: click and hold a pad.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import QLabel, QMainWindow, QVBoxLayout, QWidget

from handpiano.app.audio_controller import AudioController, NoteKeys
from handpiano.audio.engine import AudioEngine
from handpiano.instrument.note import midi_to_note_name
from handpiano.ui.holo import STYLE
from handpiano.ui.instrument_panel import InstrumentPanel
from handpiano.ui.note_pads import NotePads

HELP = ("Teclas A W S E D F T G Y H U J K O L P Ñ: notas (acordes: varias a la vez) · Z/X: octava · "
        "1/2/3: instrumento · −/+: volumen · ratón: pulsa y mantén un círculo")


class AudioTestWindow(QMainWindow):
    def __init__(self, engine: AudioEngine) -> None:
        super().__init__()
        self.setWindowTitle("HandPiano · prueba de sonido")
        self.resize(1100, 520)
        self.setStyleSheet("QMainWindow { background: #070b14; } " + STYLE)
        self.engine = engine
        self.keys = NoteKeys()
        self.panel = InstrumentPanel()
        self.audio = AudioController(engine, self.panel, self)
        self.pads = NotePads()
        self.pads.note_pressed.connect(self._on)
        self.pads.note_released.connect(self._off)
        self.info = QLabel(HELP)
        self.info.setStyleSheet("color: #9fb2d4; padding: 6px 12px;")
        self.info.setWordWrap(True)
        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.panel)
        layout.addWidget(self.pads, 1)
        layout.addWidget(self.info)
        self.setCentralWidget(central)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def _on(self, midi: int) -> None:
        self.engine.note_on(midi, 100)
        self.pads.set_lit(self.pads.lit | {midi})
        self.info.setText(f"♪ {midi_to_note_name(midi)} (MIDI {midi})   ·   " + HELP)

    def _off(self, midi: int) -> None:
        self.engine.note_off(midi)
        self.pads.set_lit(self.pads.lit - {midi})

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802
        if event.isAutoRepeat():
            return
        text = event.text()
        if self.audio.handle_shortcut(text):
            return
        before = self.keys.base
        midi = self.keys.press(text) if text else None
        if self.keys.base != before:
            for m in self.keys.release_all():
                self._off(m)
            self.pads.set_base(self.keys.base)
        if midi is not None:
            self._on(midi)

    def keyReleaseEvent(self, event: QKeyEvent) -> None:  # noqa: N802
        if event.isAutoRepeat() or not event.text():
            return
        midi = self.keys.release(event.text())
        if midi is not None:
            self._off(midi)

    def closeEvent(self, event) -> None:  # noqa: N802
        self.engine.all_notes_off()
        super().closeEvent(event)
