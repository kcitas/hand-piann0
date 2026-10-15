"""Video + hand overlay widget."""

from __future__ import annotations

import time

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QFontMetricsF, QImage, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import QSizePolicy, QWidget

from handpiano.app.pipeline import TrackingSnapshot
from handpiano.instrument.keyboard import KeyboardLayout, PianoKey
from handpiano.mapping.calibration import PianoCalibration
from handpiano.metrics.latency import RollingStats
from handpiano.metrics.performance import CpuAccounting, CpuComponent
from handpiano.tracking.finger_tracker import HandState, TrackStatus
from handpiano.tracking.landmarks import HAND_CONNECTIONS, PLAYING_FINGERS, Finger, Handedness, Landmark

HAND_COLORS = {Handedness.LEFT: QColor("#22d3ee"), Handedness.RIGHT: QColor("#a78bfa")}
BACKGROUND = QColor("#0F1426")  # theme.INK
TEXT = QColor("#ecebf5")
MUTED = QColor("#a1a1b8")
WARNING = QColor("#fbbf24")

HAND_TAG = {Handedness.LEFT: "LEFT", Handedness.RIGHT: "RIGHT"}


# The meter's full height is METER_TOP × the press threshold, so the threshold tick sits below the top.
METER_TOP = 1.4


def meter_fraction(depth: float | None, threshold: float) -> float:
    """How far a finger is towards a press: 0 at rest, 1 at the threshold (clamped to [0, METER_TOP])."""
    if depth is None or threshold <= 0:
        return 0.0
    return max(0.0, min(METER_TOP, depth / threshold))


class CameraView(QWidget):
    """Paints the frame that was actually tracked, so landmarks line up with the image.

    Measured with real timestamps, once per snapshot (repaints of the same
    snapshot, e.g. on resize, are not counted):

    * ``result_to_ui_ms``: full result (fingers + keys) ready → UI thread picked it up (polling delay).
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
        # Hand placement guide: (text, ok) shown as a bar at the bottom of the video; None = hidden.
        self._placement: tuple[str, bool] | None = None
        # Press meter next to each fingertip: the detector's depth against this threshold (None = hidden).
        self._press_threshold: float | None = None
        self._finger_labels: dict = {}  # FingerId → note name drawn above the fingertip (finger-note mode)
        self._ui_received_ns: int | None = None
        self._last_painted_seq: int | None = None
        # Calibrated keyboard projected into the camera image: (key, (4, 2) normalized corners).
        # Computed once per calibration, not per frame.
        self._key_polygons: list[tuple[PianoKey, np.ndarray]] = []
        self._show_piano = True  # False in finger-note mode: the notes are on the fingers, not on a drawn piano
        self._outline: np.ndarray | None = None
        self._target_midi: int | None = None
        # Accuracy-test target point, normalized camera coordinates.
        self._target_point: tuple[float, float] | None = None
        # Corner capture: (captured points, fingertip in use, dwell progress 0..1).
        self._calibration: tuple[list[tuple[float, float]], tuple[float, float] | None, float] | None = None
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

    def set_piano_overlay(self, layout: KeyboardLayout, calibration: PianoCalibration | None) -> None:
        self._key_polygons = []
        self._outline = None
        if calibration is not None:
            for key in (*layout.white_keys, *layout.black_keys):  # blacks last: drawn on top
                corners = [(key.x, key.y), (key.x + key.width, key.y),
                           (key.x + key.width, key.y + key.height), (key.x, key.y + key.height)]
                self._key_polygons.append((key, calibration.to_camera(corners)))
            self._outline = np.asarray(calibration.camera_points)
        self.update()

    def set_target(self, midi: int | None) -> None:
        self._target_midi = midi
        self.update()

    def set_target_point(self, point: tuple[float, float] | None) -> None:
        if point != self._target_point:
            self._target_point = point
            self.update()

    def set_calibration_overlay(
        self, points: list[tuple[float, float]] | None, cursor: tuple[float, float] | None = None, progress: float = 0.0
    ) -> None:
        overlay = None if points is None else (list(points), cursor, round(progress, 2))
        if overlay != self._calibration:
            self._calibration = overlay
            self.update()

    def set_snapshot(self, snapshot: TrackingSnapshot) -> None:
        self._ui_received_ns = time.perf_counter_ns()
        self.result_to_ui_ms.add((self._ui_received_ns - snapshot.timestamps.mapping_ready) / 1e6)
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
            if self._show_piano:
                self._draw_piano_overlay(painter, target, self._snapshot)
            self._draw_hands(painter, target, self._snapshot)
            self._draw_calibration(painter, target)
            self._draw_target_point(painter, target)
            if self._placement is not None:
                self._draw_placement(painter, target, *self._placement)

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

    def _polygon(self, target: QRectF, corners: np.ndarray) -> QPolygonF:
        return QPolygonF([self._to_widget(target, float(x), float(y)) for x, y in corners])

    def _draw_piano_overlay(self, painter: QPainter, target: QRectF, snapshot: TrackingSnapshot) -> None:
        if not self._key_polygons or self._outline is None:
            return
        hover: dict[int, set[Handedness]] = {}
        if snapshot.mapping is not None and snapshot.mapping.calibrated:
            # With press data only PRESSED keys are filled; hover alone is not a note.
            source = (snapshot.presses.pressed_keys if snapshot.presses is not None
                      else snapshot.mapping.keys_under)
            for midi, fingers in source.items():
                hover[midi] = {f.hand for f in fingers}
        for key, corners in self._key_polygons:
            polygon = self._polygon(target, corners)
            hands = hover.get(key.midi)
            if hands:
                color = QColor(HAND_COLORS[Handedness.LEFT if Handedness.LEFT in hands else Handedness.RIGHT])
                if len(hands) == 2:
                    color = QColor("#ecebf5")
                color.setAlpha(120)
                painter.setBrush(color)
            elif key.is_black:
                painter.setBrush(QColor(11, 11, 18, 110))
            else:
                painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(QColor(236, 235, 245, 110), 1))
            painter.drawPolygon(polygon)
            if key.midi == self._target_midi:
                painter.setPen(QPen(WARNING, 3))
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawPolygon(polygon)
        painter.setPen(QPen(QColor(236, 235, 245, 200), 2))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPolygon(self._polygon(target, self._outline))

    def _draw_target_point(self, painter: QPainter, target: QRectF) -> None:
        if self._target_point is None:
            return
        c = self._to_widget(target, *self._target_point)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(BACKGROUND, 6))
        painter.drawEllipse(c, 14, 14)
        painter.setPen(QPen(WARNING, 3))
        painter.drawEllipse(c, 14, 14)
        painter.drawLine(c + QPointF(-24, 0), c + QPointF(-6, 0))
        painter.drawLine(c + QPointF(6, 0), c + QPointF(24, 0))
        painter.drawLine(c + QPointF(0, -24), c + QPointF(0, -6))
        painter.drawLine(c + QPointF(0, 6), c + QPointF(0, 24))

    def _draw_calibration(self, painter: QPainter, target: QRectF) -> None:
        if self._calibration is None:
            return
        points, cursor, progress = self._calibration
        widget_points = [self._to_widget(target, x, y) for x, y in points]
        painter.setPen(QPen(WARNING, 2, Qt.PenStyle.DashLine))
        for a, b in zip(widget_points, widget_points[1:], strict=False):
            painter.drawLine(a, b)
        painter.setFont(QFont("Helvetica", 12, QFont.Weight.Bold))
        for i, p in enumerate(widget_points, start=1):
            painter.setPen(QPen(BACKGROUND, 2))
            painter.setBrush(WARNING)
            painter.drawEllipse(p, 11, 11)
            painter.setPen(BACKGROUND)
            painter.drawText(QRectF(p.x() - 11, p.y() - 11, 22, 22), int(Qt.AlignmentFlag.AlignCenter), f"P{i}")
        if cursor is not None:
            c = self._to_widget(target, *cursor)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(QColor(251, 191, 36, 90), 4))
            painter.drawEllipse(c, 22, 22)
            painter.setPen(QPen(WARNING, 4))
            painter.drawArc(QRectF(c.x() - 22, c.y() - 22, 44, 44), 90 * 16, int(-360 * 16 * progress))

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
            if snapshot.presses is not None and self._press_threshold and state.status is TrackStatus.TRACKED:
                fp = snapshot.presses.fingers[finger]
                pressed = fp.phase.value in ("PRESSED", "RELEASE_CANDIDATE")
                self._draw_meter(painter, center, meter_fraction(fp.depth, self._press_threshold), pressed, color)
            label = self._finger_labels.get(finger)
            if label:
                font = QFont("Helvetica", 13, QFont.Weight.Bold)
                painter.setFont(font)
                box = QRectF(center.x() - 30, center.y() - 34, 60, 20)
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QColor(7, 11, 20, 170))
                painter.drawRoundedRect(box.adjusted(8, 0, -8, 0), 6, 6)
                painter.setPen(QColor("#ffffff"))
                painter.drawText(box, int(Qt.AlignmentFlag.AlignCenter), label)
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

    def set_piano_visible(self, visible: bool) -> None:
        self._show_piano = visible
        self.update()

    def set_finger_labels(self, labels: dict) -> None:
        """Finger-note mode: the note each finger plays, drawn above its fingertip."""
        self._finger_labels = dict(labels)
        self.update()

    def set_press_threshold(self, threshold: float | None) -> None:
        """Show, next to each fingertip, how far the finger is from counting as a press."""
        self._press_threshold = threshold
        self.update()

    def _draw_meter(self, painter: QPainter, tip: QPointF, fraction: float, pressed: bool, color: QColor) -> None:
        height, width = 36.0, 7.0
        box = QRectF(tip.x() + 11, tip.y() - height / 2, width, height)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(7, 11, 20, 170))
        painter.drawRoundedRect(box.adjusted(-1.5, -1.5, 1.5, 1.5), 3, 3)
        fill_h = height * min(fraction, METER_TOP) / METER_TOP
        fill = QColor("#34d399") if pressed else QColor(color)
        painter.setBrush(fill)
        painter.drawRect(QRectF(box.left(), box.bottom() - fill_h, width, fill_h))
        tick_y = box.bottom() - height / METER_TOP  # the press threshold
        painter.setPen(QPen(QColor("#ffffff"), 2))
        painter.drawLine(QPointF(box.left() - 3, tick_y), QPointF(box.right() + 3, tick_y))

    def set_placement(self, text: str | None, ok: bool = False) -> None:
        placement = None if text is None else (text, ok)
        if placement != self._placement:
            self._placement = placement
            self.update()

    def _draw_placement(self, painter: QPainter, target: QRectF, text: str, ok: bool) -> None:
        label = ("✓ " if ok else "⚠ ") + text
        width = target.width() - 24
        # A pill sized to its text when it fits; it never covers more than the video's width.
        natural = QFontMetricsF(QFont("Helvetica", 16, QFont.Weight.Bold)).horizontalAdvance(label) + 56
        width = min(width, natural)
        size = 16
        font = QFont("Helvetica", size, QFont.Weight.Bold)
        while size > 10 and QFontMetricsF(font).horizontalAdvance(label) > width - 24:  # always fits the video
            size -= 1
            font = QFont("Helvetica", size, QFont.Weight.Bold)
        label = QFontMetricsF(font).elidedText(label, Qt.TextElideMode.ElideRight, width - 24)
        height = QFontMetricsF(font).height() + 16
        bar = QRectF(target.center().x() - width / 2, target.bottom() - height - 14, width, height)
        color = QColor("#34d399" if ok else "#f59e0b")
        fill = QColor(color)
        fill.setAlpha(225)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(fill)
        painter.drawRoundedRect(bar, height / 2, height / 2)
        painter.setPen(QColor("#070b14"))
        painter.setFont(font)
        painter.drawText(bar, int(Qt.AlignmentFlag.AlignCenter), label)

    def _draw_message(self, painter: QPainter, message: str) -> None:
        box = QRectF(self.rect()).adjusted(40, 40, -40, -40)
        painter.setPen(TEXT)
        painter.setFont(QFont("Helvetica", 15))
        painter.drawText(box, int(Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap), message)
