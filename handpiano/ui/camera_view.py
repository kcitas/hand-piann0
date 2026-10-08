"""Video + hand overlay widget."""

from __future__ import annotations

import time

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QFontMetricsF, QImage, QPainter, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget

from handpiano.app.pipeline import TrackingSnapshot
from handpiano.metrics.latency import RollingStats
from handpiano.metrics.performance import CpuAccounting, CpuComponent
from handpiano.tracking.finger_tracker import HandState, TrackStatus
from handpiano.tracking.landmarks import HAND_CONNECTIONS, PLAYING_FINGERS, Finger, Handedness, Landmark

HAND_COLORS = {Handedness.LEFT: QColor("#22d3ee"), Handedness.RIGHT: QColor("#a78bfa")}
BACKGROUND = QColor("#0b0b12")
TEXT = QColor("#ecebf5")
MUTED = QColor("#a1a1b8")
WARNING = QColor("#fbbf24")

HAND_TAG = {Handedness.LEFT: "LEFT", Handedness.RIGHT: "RIGHT"}


class CameraView(QWidget):
    """Paints the frame that was actually tracked, so landmarks line up with the image.

    Measured with real timestamps, once per snapshot (repaints of the same
    snapshot, e.g. on resize, are not counted):

    * ``result_to_ui_ms``: tracker result ready → UI thread picked it up (polling delay).
    * ``ui_to_paint_ms``: UI picked it up → paintEvent finished.
    * ``capture_to_paint_ms``: frame delivered by the driver → paintEvent finished.
    * ``paint_ms``: duration of every paintEvent.

    "Paint finished" means Qt rendered the widget's backing store. The moment
    the pixels physically appear on the display (compositor, refresh) comes later
    and is not measurable from inside the app.
    """

    def __init__(self, parent: QWidget | None = None, cpu: CpuAccounting | None = None) -> None:
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(320, 240)
        self._cpu = cpu
        self._snapshot: TrackingSnapshot | None = None
        self._image: QImage | None = None
        self._message: str | None = "Iniciando cámara…"
        self._lab_mode = False
        # (title, detail lines, progress 0..1 or None, accent colour) shown on top of the video.
        self._banner: tuple[str, list[str], float | None, QColor] | None = None
        self._ui_received_ns: int | None = None
        self._last_painted_seq: int | None = None
        self.paint_ms = RollingStats()
        self.result_to_ui_ms = RollingStats()
        self.ui_to_paint_ms = RollingStats()
        self.capture_to_paint_ms = RollingStats()

    def reset_metrics(self) -> None:
        for stats in (self.paint_ms, self.result_to_ui_ms, self.ui_to_paint_ms, self.capture_to_paint_ms):
            stats.clear()
        self._snapshot = self._image = None
        self._last_painted_seq = None

    def set_lab_mode(self, enabled: bool) -> None:
        self._lab_mode = enabled
        self.update()

    def set_banner(
        self, title: str | None, lines: list[str] | None = None, progress: float | None = None, color: str = "#fbbf24"
    ) -> None:
        if progress is not None:
            progress = round(progress, 2)  # repaint for visible changes only, not on every poll
        banner = None if title is None else (title, lines or [], progress, QColor(color))
        if banner != self._banner:
            self._banner = banner
            self.update()

    def set_message(self, message: str | None) -> None:
        self._message = message
        self.update()

    def set_snapshot(self, snapshot: TrackingSnapshot) -> None:
        self._ui_received_ns = time.perf_counter_ns()
        self.result_to_ui_ms.add((self._ui_received_ns - snapshot.timestamps.finger_state_ready) / 1e6)
        self._snapshot = snapshot
        img = snapshot.frame.image
        height, width = img.shape[:2]
        # QImage references the numpy buffer; the snapshot keeps it alive.
        self._image = QImage(img.data, width, height, img.strides[0], QImage.Format.Format_RGB888)
        self._message = None
        self.update()

    # -- painting --------------------------------------------------------------

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        cpu_start = time.thread_time()
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
        if self._banner is not None:
            self._draw_banner(painter, *self._banner)
        painter.end()

        done_ns = time.perf_counter_ns()
        self.paint_ms.add((done_ns - started_ns) / 1e6)
        snapshot = self._snapshot
        if snapshot is not None and self._message is None and snapshot.seq != self._last_painted_seq:
            self._last_painted_seq = snapshot.seq
            self.capture_to_paint_ms.add((done_ns - snapshot.timestamps.frame_capture) / 1e6)
            if self._ui_received_ns is not None:
                self.ui_to_paint_ms.add((done_ns - self._ui_received_ns) / 1e6)
        if self._cpu is not None:
            self._cpu.add(CpuComponent.UI, time.thread_time() - cpu_start)

    def _fit_rect(self, width: int, height: int) -> QRectF:
        scale = min(self.width() / width, self.height() / height)
        w, h = width * scale, height * scale
        return QRectF((self.width() - w) / 2, (self.height() - h) / 2, w, h)

    @staticmethod
    def _to_widget(target: QRectF, x: float, y: float) -> QPointF:
        return QPointF(target.left() + x * target.width(), target.top() + y * target.height())

    def _draw_hands(self, painter: QPainter, target: QRectF, snapshot: TrackingSnapshot) -> None:
        if self._lab_mode:
            for hand_state in snapshot.state.hands.values():
                if hand_state.landmarks is not None:
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
        if self._lab_mode:
            for hand_state in snapshot.state.hands.values():
                if hand_state.landmarks is not None:
                    self._draw_hand_tag(painter, target, hand_state)

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

    def _draw_hand_tag(self, painter: QPainter, target: QRectF, hand: HandState) -> None:
        """Large LEFT/RIGHT badge below the wrist, for manual handedness validation."""
        assert hand.landmarks is not None
        wrist = self._to_widget(target, float(hand.landmarks[Landmark.WRIST, 0]), float(hand.landmarks[Landmark.WRIST, 1]))
        score = "N/A" if hand.score is None else f"{hand.score:.2f}"
        fingers = " ".join(f.short for f in PLAYING_FINGERS if f.hand is hand.hand)
        lines = [(HAND_TAG[hand.hand], QFont("Helvetica", 22, QFont.Weight.Bold), HAND_COLORS[hand.hand])]
        lines.append((f"{hand.status.value} · score {score}", QFont("Helvetica", 11), TEXT))
        lines.append((fingers, QFont("Helvetica", 10), MUTED))
        if hand.label_overridden and hand.detector_label is not None:
            lines.append(
                (f"detector: {HAND_TAG[hand.detector_label]} · se mantiene por continuidad", QFont("Helvetica", 10), WARNING)
            )

        widths = [QFontMetricsF(font).horizontalAdvance(text) for text, font, _ in lines]
        heights = [QFontMetricsF(font).height() for _, font, _ in lines]
        box = QRectF(0, 0, max(widths) + 16, sum(heights) + 10)
        box.moveCenter(wrist + QPointF(0, box.height() / 2 + 18))
        # Keep the badge inside the video area.
        box.moveLeft(min(max(box.left(), target.left()), target.right() - box.width()))
        box.moveTop(min(box.top(), target.bottom() - box.height()))

        painter.setPen(QPen(HAND_COLORS[hand.hand], 2))
        painter.setBrush(QColor(11, 11, 18, 210))
        painter.drawRoundedRect(box, 8, 8)
        y = box.top() + 5
        for (text, font, color), height in zip(lines, heights, strict=True):
            painter.setFont(font)
            painter.setPen(color)
            painter.drawText(QRectF(box.left(), y, box.width(), height), int(Qt.AlignmentFlag.AlignHCenter), text)
            y += height

    def _draw_banner(self, painter: QPainter, title: str, lines: list[str], progress: float | None, color: QColor) -> None:
        title_font = QFont("Helvetica", 26, QFont.Weight.Bold)
        line_font = QFont("Helvetica", 14)
        title_h = QFontMetricsF(title_font).height()
        line_h = QFontMetricsF(line_font).height()
        height = 24 + title_h + line_h * len(lines) + (14 if progress is not None else 0)
        box = QRectF(16, 16, self.width() - 32, height)
        painter.setPen(QPen(color, 3))
        painter.setBrush(QColor(11, 11, 18, 225))
        painter.drawRoundedRect(box, 12, 12)
        painter.setFont(title_font)
        painter.setPen(color)
        painter.drawText(QRectF(box.left(), box.top() + 10, box.width(), title_h), int(Qt.AlignmentFlag.AlignHCenter), title)
        painter.setFont(line_font)
        painter.setPen(TEXT)
        y = box.top() + 12 + title_h
        for line in lines:
            painter.drawText(QRectF(box.left(), y, box.width(), line_h), int(Qt.AlignmentFlag.AlignHCenter), line)
            y += line_h
        if progress is not None:
            bar = QRectF(box.left() + 24, box.bottom() - 14, box.width() - 48, 6)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor("#1f1f2e"))
            painter.drawRoundedRect(bar, 3, 3)
            painter.setBrush(color)
            painter.drawRoundedRect(QRectF(bar.left(), bar.top(), bar.width() * max(0.0, min(1.0, progress)), 6), 3, 3)

    def _draw_message(self, painter: QPainter, message: str) -> None:
        box = QRectF(self.rect()).adjusted(40, 40, -40, -40)
        painter.setPen(TEXT)
        painter.setFont(QFont("Helvetica", 15))
        painter.drawText(box, int(Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap), message)
