"""Video + hand overlay widget."""

from __future__ import annotations

import time

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget

from handpiano.app.pipeline import TrackingSnapshot
from handpiano.metrics.latency import RollingStats
from handpiano.tracking.finger_tracker import HandState, TrackStatus
from handpiano.tracking.landmarks import HAND_CONNECTIONS, PLAYING_FINGERS, Finger, Handedness, Landmark

HAND_COLORS = {Handedness.LEFT: QColor("#22d3ee"), Handedness.RIGHT: QColor("#a78bfa")}
BACKGROUND = QColor("#0b0b12")
TEXT = QColor("#ecebf5")
MUTED = QColor("#a1a1b8")


class CameraView(QWidget):
    """Paints the frame that was actually tracked, so landmarks line up with the image.

    Also measures (with real timestamps) how long painting takes and the delay
    from frame capture to the moment it is painted.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(320, 240)
        self._snapshot: TrackingSnapshot | None = None
        self._image: QImage | None = None
        self._message: str | None = "Iniciando cámara…"
        self._lab_mode = False
        self.paint_ms = RollingStats()
        self.capture_to_paint_ms = RollingStats()

    def set_lab_mode(self, enabled: bool) -> None:
        self._lab_mode = enabled
        self.update()

    def set_message(self, message: str | None) -> None:
        self._message = message
        self.update()

    def set_snapshot(self, snapshot: TrackingSnapshot) -> None:
        self._snapshot = snapshot
        img = snapshot.frame.image
        height, width = img.shape[:2]
        # QImage references the numpy buffer; the snapshot keeps it alive.
        self._image = QImage(img.data, width, height, img.strides[0], QImage.Format.Format_RGB888)
        self._message = None
        self.update()

    # -- painting --------------------------------------------------------------

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        started_ns = time.perf_counter_ns()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), BACKGROUND)

        if self._image is not None and self._snapshot is not None:
            target = self._fit_rect(self._image.width(), self._image.height())
            painter.drawImage(target, self._image)
            self._draw_hands(painter, target, self._snapshot)

        if self._message:
            self._draw_message(painter, self._message)
        painter.end()

        done_ns = time.perf_counter_ns()
        self.paint_ms.add((done_ns - started_ns) / 1e6)
        if self._snapshot is not None and self._message is None:
            self.capture_to_paint_ms.add((done_ns - self._snapshot.frame.captured_at_ns) / 1e6)

    def _fit_rect(self, width: int, height: int) -> QRectF:
        scale = min(self.width() / width, self.height() / height)
        w, h = width * scale, height * scale
        return QRectF((self.width() - w) / 2, (self.height() - h) / 2, w, h)

    @staticmethod
    def _to_widget(target: QRectF, x: float, y: float) -> QPointF:
        return QPointF(target.left() + x * target.width(), target.top() + y * target.height())

    def _draw_hands(self, painter: QPainter, target: QRectF, snapshot: TrackingSnapshot) -> None:
        for hand_state in snapshot.state.hands.values():
            if hand_state.landmarks is None:
                continue
            if self._lab_mode:
                self._draw_skeleton(painter, target, hand_state)
        for finger in PLAYING_FINGERS:
            state = snapshot.state.fingers[finger]
            if state.tip is None:
                continue
            color = HAND_COLORS[finger.hand]
            center = self._to_widget(target, float(state.tip[0]), float(state.tip[1]))
            if state.status is TrackStatus.TRACKED:
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(color)
            else:  # HELD: hollow marker, the position is stale
                painter.setPen(QPen(color, 2))
                painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(center, 7, 7)
            if self._lab_mode:
                painter.setPen(TEXT)
                painter.setFont(QFont("Helvetica", 10))
                painter.drawText(center + QPointF(9, -9), finger.short)

    def _draw_skeleton(self, painter: QPainter, target: QRectF, hand: HandState) -> None:
        assert hand.landmarks is not None
        color = HAND_COLORS[hand.hand]
        points = [self._to_widget(target, float(p[0]), float(p[1])) for p in hand.landmarks]

        line = QColor(color)
        line.setAlpha(170)
        painter.setPen(QPen(line, 2))
        for a, b in HAND_CONNECTIONS:
            painter.drawLine(points[a], points[b])

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor("#ffffff"))
        for i, p in enumerate(points):
            if i not in {f.tip for f in Finger}:
                painter.drawEllipse(p, 3, 3)
        # Thumb is drawn but is not a playing finger.
        painter.setBrush(MUTED)
        painter.drawEllipse(points[Landmark.THUMB_TIP], 5, 5)

        wrist = points[Landmark.WRIST]
        score = "N/A" if hand.score is None else f"{hand.score:.2f}"
        label = f"{'Izquierda' if hand.hand is Handedness.LEFT else 'Derecha'} · {hand.status.value} · {score}"
        painter.setPen(color)
        painter.setFont(QFont("Helvetica", 11, QFont.Weight.Bold))
        painter.drawText(wrist + QPointF(-40, 22), label)

    def _draw_message(self, painter: QPainter, message: str) -> None:
        box = QRectF(self.rect()).adjusted(40, 40, -40, -40)
        painter.setPen(TEXT)
        painter.setFont(QFont("Helvetica", 15))
        painter.drawText(box, int(Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap), message)
