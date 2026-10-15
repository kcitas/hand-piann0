"""Holo note pads: glowing circular pads for one and a half octaves, playable with the mouse.

Used by the audio test window (no camera). Each pad shows the note name and the
computer key that plays it; sounding notes light up. Emits note_pressed /
note_released; it does not know about audio.
"""

from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget

from handpiano.instrument.note import is_black_key, midi_to_note_name
from handpiano.ui.holo import CYAN, TEXT, TEXT_DIM, VIOLET, glow, paint_grid

KEY_LABELS = ("A", "W", "S", "E", "D", "F", "T", "G", "Y", "H", "U", "J", "K", "O", "L", "P", "Ñ")


class NotePads(QWidget):
    note_pressed = Signal(int)
    note_released = Signal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.base = 60
        self.lit: set[int] = set()
        self._mouse_note: int | None = None
        self.setMinimumHeight(220)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def set_base(self, base: int) -> None:
        self.base = base
        self.update()

    def set_lit(self, notes: set[int]) -> None:
        if notes != self.lit:
            self.lit = set(notes)
            self.update()

    def pads(self) -> list[tuple[int, QPointF, float, str]]:
        """(midi, centre, radius, key label) for each pad: naturals on the lower row, sharps above."""
        w, h = self.width(), self.height()
        naturals = [i for i in range(len(KEY_LABELS)) if not is_black_key(self.base + i)]
        step = w / (len(naturals) + 1)
        radius = min(step * 0.42, h * 0.2)
        out = []
        n = 0
        for i, label in enumerate(KEY_LABELS):
            midi = self.base + i
            if is_black_key(midi):
                out.append((midi, QPointF(step * (n + 0.5), h * 0.36), radius * 0.8, label))
            else:
                n += 1
                out.append((midi, QPointF(step * n, h * 0.68), radius, label))
        return out

    def pad_at(self, pos: QPointF) -> int | None:
        for midi, c, r, _ in self.pads():
            if (pos.x() - c.x()) ** 2 + (pos.y() - c.y()) ** 2 <= r * r:
                return midi
        return None

    def mousePressEvent(self, event) -> None:  # noqa: N802
        midi = self.pad_at(event.position())
        if midi is not None:
            self._mouse_note = midi
            self.note_pressed.emit(midi)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if self._mouse_note is not None:
            self.note_released.emit(self._mouse_note)
            self._mouse_note = None

    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        paint_grid(p, QRectF(self.rect()))
        for midi, c, r, label in self.pads():
            sharp = is_black_key(midi)
            color = VIOLET if sharp else CYAN
            on = midi in self.lit
            if on:
                glow(p, c, r * 1.9, color, 1.0)
            fill = QColor(color)
            fill.setAlpha(200 if on else 28)
            p.setBrush(fill)
            p.setPen(QPen(color, 2.5 if on else 1.5))
            p.drawEllipse(c, r, r)
            p.setPen(TEXT)
            p.setFont(QFont("Helvetica", max(8, int(r * 0.32)), QFont.Weight.Bold))
            p.drawText(QRectF(c.x() - r, c.y() - r * 0.55, 2 * r, r * 0.7), int(Qt.AlignmentFlag.AlignCenter),
                       midi_to_note_name(midi))
            p.setPen(TEXT_DIM)
            p.setFont(QFont("Helvetica", max(7, int(r * 0.24))))
            p.drawText(QRectF(c.x() - r, c.y() + r * 0.05, 2 * r, r * 0.6), int(Qt.AlignmentFlag.AlignCenter), label)
        p.end()
