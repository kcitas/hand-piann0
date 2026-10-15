"""Finger-note mode keyboard: eight ivory keys, one per finger, lit by the amber spotlight while they sound.

Left to right as the hands are held: left pinky (Do) … left index (Fa) | right index (Sol) … right
pinky (Do). Each key carries its finger's name and a strip in its hand's colour. Clicking a key plays
it, like the drawn keyboard.
"""

from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QFont, QLinearGradient, QMouseEvent, QPainter, QPaintEvent, QPen
from PySide6.QtWidgets import QWidget

from handpiano.interaction.finger_presses import FINGER_NOTES, PressFrame
from handpiano.tracking.landmarks import FingerId, Handedness
from handpiano.ui.theme import FONT, INK, IVORY, LEFT_HAND, LINE, MUTED, RIGHT_HAND, SPOTLIGHT, qcolor

ORDER = (FingerId.LEFT_PINKY, FingerId.LEFT_RING, FingerId.LEFT_MIDDLE, FingerId.LEFT_INDEX,
         FingerId.RIGHT_INDEX, FingerId.RIGHT_MIDDLE, FingerId.RIGHT_RING, FingerId.RIGHT_PINKY)
NOTE_NAMES = {60: "Do", 62: "Re", 64: "Mi", 65: "Fa", 67: "Sol", 69: "La", 71: "Si", 72: "Do"}
FINGER_NAMES = {"PINKY": "meñique", "RING": "anular", "MIDDLE": "medio", "INDEX": "índice"}


class FingerKeysView(QWidget):
    key_pressed = Signal(int)
    key_released = Signal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(110)
        self.setMaximumHeight(170)  # the video of the hands gets the rest of the window
        self._sounding: set[int] = set()
        self._mouse_midi: int | None = None

    # -- state -----------------------------------------------------------------------

    def set_presses(self, presses: PressFrame | None) -> None:
        sounding = set() if presses is None else set(presses.pressed_keys)
        if sounding != self._sounding:
            self._sounding = sounding
            self.update()

    def sounding(self) -> set[int]:
        return set(self._sounding) | ({self._mouse_midi} if self._mouse_midi is not None else set())

    # -- geometry ----------------------------------------------------------------------

    def _keys_rect(self) -> QRectF:
        return QRectF(self.rect()).adjusted(16, 12, -16, -12)

    def key_rect(self, index: int) -> QRectF:
        area = self._keys_rect()
        gap_between_hands = 22.0
        gap = 8.0
        width = (area.width() - gap_between_hands - 6 * gap) / 8
        x = area.left() + index * (width + gap) + (gap_between_hands - gap if index >= 4 else 0)
        return QRectF(x, area.top(), width, area.height())

    def midi_at(self, point: QPointF) -> int | None:
        for i, finger in enumerate(ORDER):
            if self.key_rect(i).contains(point):
                return FINGER_NOTES[finger]
        return None

    # -- mouse ---------------------------------------------------------------------------

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        midi = self.midi_at(event.position())
        if midi is not None:
            self._mouse_midi = midi
            self.key_pressed.emit(midi)
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if self._mouse_midi is not None:
            self.key_released.emit(self._mouse_midi)
            self._mouse_midi = None
            self.update()

    # -- painting ------------------------------------------------------------------------

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), qcolor(INK))
        sounding = self.sounding()
        for i, finger in enumerate(ORDER):
            self._draw_key(painter, self.key_rect(i), finger, FINGER_NOTES[finger] in sounding)
        painter.end()

    def _draw_key(self, painter: QPainter, rect: QRectF, finger: FingerId, lit: bool) -> None:
        hand_color = qcolor(LEFT_HAND if finger.hand is Handedness.LEFT else RIGHT_HAND)
        if lit:  # the spotlight: a warm glow around the key, the key itself turns amber
            for spread, alpha in ((14, 40), (8, 70), (3, 120)):
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(qcolor(SPOTLIGHT, alpha))
                painter.drawRoundedRect(rect.adjusted(-spread, -spread, spread, spread), 14 + spread, 14 + spread)
            fill = QLinearGradient(rect.topLeft(), rect.bottomLeft())
            fill.setColorAt(0.0, qcolor("#FFE2A3"))
            fill.setColorAt(1.0, qcolor(SPOTLIGHT))
            painter.setBrush(fill)
        else:
            fill = QLinearGradient(rect.topLeft(), rect.bottomLeft())
            fill.setColorAt(0.0, qcolor(IVORY))
            fill.setColorAt(1.0, qcolor("#D9D2C2"))
            painter.setBrush(fill)
        painter.setPen(QPen(qcolor(LINE), 1))
        painter.drawRoundedRect(rect, 12, 12)

        strip = QRectF(rect.left() + 10, rect.bottom() - 12, rect.width() - 20, 5)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(hand_color)
        painter.drawRoundedRect(strip, 2.5, 2.5)

        note = NOTE_NAMES[FINGER_NOTES[finger]]
        size = max(14, min(34, int(rect.height() * 0.32)))
        painter.setPen(qcolor(INK))
        painter.setFont(QFont(FONT, size, QFont.Weight.DemiBold))
        painter.drawText(rect.adjusted(0, 0, 0, -rect.height() * 0.28), int(Qt.AlignmentFlag.AlignCenter), note)
        painter.setPen(qcolor(INK if lit else MUTED))
        painter.setFont(QFont(FONT, max(9, size // 3)))
        label = FINGER_NAMES[finger.finger.name]
        painter.drawText(rect.adjusted(0, rect.height() * 0.45, 0, -18), int(Qt.AlignmentFlag.AlignCenter), label)
