"""On-screen keyboard: hover highlighting and finger markers. Drawn from KeyboardLayout (no hard-coded keys).

UI coordinate space: the keyboard (piano coordinates, white-key widths) is
stretched to fill the widget minus a margin, with independent scales so a
short panel still shows a full-width keyboard:

    ui_x = left + piano_x · sx        ui_y = top + piano_y · sy

This only affects drawing. Geometry and hit testing always use piano coordinates.

With a press detector, a PRESSED key is filled with the hand colour and a key
that only has a finger over it (hover) gets a thin outline, so fingers that
merely rest above the keyboard do not look like notes. Without press data
(older callers, tests) hover is filled, as before.
"""

from __future__ import annotations

from collections.abc import Callable

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget

from handpiano.instrument.keyboard import KeyboardLayout, PianoKey
from handpiano.interaction.finger_presses import PressFrame
from handpiano.mapping.key_mapper import MappingState
from handpiano.tracking.finger_tracker import TrackStatus
from handpiano.tracking.landmarks import PLAYING_FINGERS, FingerId, Handedness
from handpiano.ui.camera_view import HAND_COLORS, TEXT

WHITE = QColor("#ecebf5")
WHITE_DIM = QColor("#6b6b80")
BLACK = QColor("#14141f")
BLACK_DIM = QColor("#2a2a3a")
EDGE = QColor("#0b0b12")
TARGET = QColor("#fbbf24")
BACKGROUND = QColor("#0b0b12")
MARGIN = 12.0


class PianoView(QWidget):
    # Mouse alternative to hand presses (Free Piano): MIDI note under the cursor pressed / released.
    key_pressed = Signal(int)
    key_released = Signal(int)

    def __init__(self, layout: KeyboardLayout, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._mouse_midi: int | None = None
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.setMinimumHeight(130)
        self.layout_ = layout
        self._mapping: MappingState | None = None
        self._presses: PressFrame | None = None
        self._calibrated = False
        self._notice: str | None = "Calibración necesaria · pulsa «Calibrar piano»"
        self._target_midi: int | None = None
        self._lab_mode = False
        # When set (microphone mode), the keys lit as pressed are the ones SOUNDING (MIDI → fingers),
        # not the camera detector's pressed keys.
        self.pressed_source: Callable[[], dict[int, tuple[FingerId, ...]]] | None = None

    def set_lab_mode(self, enabled: bool) -> None:
        self._lab_mode = enabled
        self.update()

    def set_state(
        self, mapping: MappingState | None, calibrated: bool, notice: str | None = None, presses: PressFrame | None = None
    ) -> None:
        self._mapping = mapping
        self._presses = presses
        self._calibrated = calibrated
        self._notice = notice
        self.update()

    def set_target(self, midi: int | None) -> None:
        self._target_midi = midi
        self.update()

    # -- geometry ----------------------------------------------------------------

    def keyboard_rect(self) -> tuple[float, float, float, float]:
        """(left, top, sx, sy): piano → widget pixel transform."""
        w, h = self.layout_.size
        sx = max(0.0, (self.width() - 2 * MARGIN) / w)
        sy = max(0.0, (self.height() - 2 * MARGIN) / h)
        return MARGIN, MARGIN, sx, sy

    def key_rect(self, key: PianoKey) -> QRectF:
        left, top, sx, sy = self.keyboard_rect()
        return QRectF(left + key.x * sx, top + key.y * sy, key.width * sx, key.height * sy)

    def to_widget(self, piano_x: float, piano_y: float) -> QPointF:
        left, top, sx, sy = self.keyboard_rect()
        return QPointF(left + piano_x * sx, top + piano_y * sy)

    def key_at_widget(self, pos: QPointF) -> PianoKey | None:
        left, top, sx, sy = self.keyboard_rect()
        if sx <= 0 or sy <= 0:
            return None
        return self.layout_.key_at((pos.x() - left) / sx, (pos.y() - top) / sy)

    def mousePressEvent(self, event) -> None:  # noqa: N802
        key = self.key_at_widget(event.position())
        if key is not None and event.button() == Qt.MouseButton.LeftButton:
            self._mouse_midi = key.midi
            self.key_pressed.emit(key.midi)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if self._mouse_midi is not None:
            self.key_released.emit(self._mouse_midi)
            self._mouse_midi = None

    # -- painting ----------------------------------------------------------------

    def _hover_hands(self) -> dict[int, set[Handedness]]:
        hover: dict[int, set[Handedness]] = {}
        if self._mapping is None or not self._calibrated:
            return hover
        for midi, fingers in self._mapping.keys_under.items():
            hover[midi] = {f.hand for f in fingers}
        return hover

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), BACKGROUND)
        hover = self._hover_hands()
        pressed = self._pressed_hands()
        active = self._calibrated
        if pressed is None:  # no press data: hover is shown filled (pre-press behaviour)
            filled, outlined = hover, {}
        else:
            filled, outlined = pressed, {m: h for m, h in hover.items() if m not in pressed}

        for key in self.layout_.white_keys:
            self._draw_key(painter, key, WHITE if active else WHITE_DIM, filled.get(key.midi), outlined.get(key.midi))
        for key in self.layout_.black_keys:
            self._draw_key(painter, key, BLACK if active else BLACK_DIM, filled.get(key.midi), outlined.get(key.midi))
        self._draw_labels(painter)
        if self._target_midi is not None and (target := self.layout_.key_by_midi(self._target_midi)):
            painter.setPen(QPen(TARGET, 3))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(self.key_rect(target).adjusted(1.5, 1.5, -1.5, -1.5))
        if active:
            self._draw_fingers(painter)
        if self._notice:
            painter.setPen(TEXT)
            painter.setFont(QFont("Helvetica", 15, QFont.Weight.Bold))
            box = QRectF(self.rect()).adjusted(20, 0, -20, 0)
            painter.fillRect(QRectF(box.left(), box.center().y() - 18, box.width(), 36), QColor(11, 11, 18, 200))
            painter.drawText(box, int(Qt.AlignmentFlag.AlignCenter), self._notice)
        painter.end()

    def _pressed_hands(self) -> dict[int, set[Handedness]] | None:
        if self.pressed_source is not None:
            return {midi: {f.hand for f in fingers} for midi, fingers in self.pressed_source().items()} \
                if self._calibrated else {}
        if self._presses is None:
            return None
        if not self._calibrated:
            return {}
        return {midi: {f.hand for f in fingers} for midi, fingers in self._presses.pressed_keys.items()}

    def _draw_key(
        self, painter: QPainter, key: PianoKey, base: QColor, hands: set[Handedness] | None,
        hover_hands: set[Handedness] | None = None,
    ) -> None:
        rect = self.key_rect(key)
        painter.setPen(QPen(EDGE, 1.5))
        painter.setBrush(base)
        painter.drawRect(rect)
        if hover_hands:
            hand = Handedness.LEFT if Handedness.LEFT in hover_hands else Handedness.RIGHT
            painter.setPen(QPen(HAND_COLORS[hand], 2.5))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(rect.adjusted(2, 2, -2, -2))
        if not hands:
            return
        ordered = [h for h in (Handedness.LEFT, Handedness.RIGHT) if h in hands]
        part = rect.width() / len(ordered)
        painter.setPen(Qt.PenStyle.NoPen)
        for i, hand in enumerate(ordered):
            color = QColor(HAND_COLORS[hand])
            color.setAlpha(210 if key.is_black else 170)
            painter.setBrush(color)
            painter.drawRect(QRectF(rect.left() + i * part, rect.top(), part, rect.height()).adjusted(1, 1, -1, -1))

    def _draw_labels(self, painter: QPainter) -> None:
        _, _, sx, _ = self.keyboard_rect()
        size = max(7, min(11, int(sx * 0.22)))
        for key in self.layout_.keys:
            if not self._lab_mode and (key.is_black or key.note.pitch_class != "C"):
                continue
            rect = self.key_rect(key)
            if key.is_black:
                painter.setFont(QFont("Helvetica", max(6, size - 2)))
                painter.setPen(QColor("#a1a1b8"))
                painter.drawText(rect.adjusted(0, 3, 0, 0), int(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop),
                                 key.note.pitch_class.replace("#", "♯"))
            else:
                painter.setFont(QFont("Helvetica", size))
                painter.setPen(QColor("#4a4a63"))
                painter.drawText(rect.adjusted(0, 0, 0, -4),
                                 int(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignBottom), key.name)

    def _draw_fingers(self, painter: QPainter) -> None:
        if self._mapping is None:
            return
        painter.setFont(QFont("Helvetica", 9, QFont.Weight.Bold))
        stacked: dict[int, int] = {}  # labels of fingers sharing a key are stacked, not overprinted
        for finger in PLAYING_FINGERS:
            fk = self._mapping.fingers[finger]
            if fk.piano_xy is None or fk.key is None:
                continue
            p = self.to_widget(*fk.piano_xy)
            color = HAND_COLORS[finger.hand]
            # White outline so the dot stays visible over a key tinted in the same hand colour.
            if fk.status is TrackStatus.TRACKED:
                painter.setPen(QPen(QColor("#ffffff"), 2))
                painter.setBrush(color)
            else:  # HELD: hollow ring, stale position
                painter.setPen(QPen(color, 2.5))
                painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(p, 6, 6)
            row = stacked.get(fk.key.midi, 0)
            stacked[fk.key.midi] = row + 1
            painter.setPen(color)
            painter.drawText(p + QPointF(-16, -10 - 12 * row), finger.short)
