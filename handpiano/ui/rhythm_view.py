"""Rhythm game 4K view: song slots, four neon lanes, falling notes, hit line, score HUD.

Draws the ``RhythmGame`` it is given and emits what the player asks for; the
game rules and the audio live elsewhere. Lanes can be hit with D F J K, by
clicking a lane, or (in the app) with presses on the four quarters of the
calibrated keyboard.
"""

from __future__ import annotations

import time

from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from handpiano.game.chart import LANES, Chart, SongSlot, practice_pattern
from handpiano.game.rhythm import RhythmGame
from handpiano.instrument.note import midi_to_note_name
from handpiano.ui.holo import GREEN, LANE_COLORS, RED, TEXT, TEXT_DIM, glow, paint_grid

LANE_KEYS = ("D", "F", "J", "K")
LOOKAHEAD_S = 1.6
HIT_LINE = 0.84  # fraction of the canvas height
GRADE_COLORS = {"PERFECT": QColor("#22d3ee"), "GREAT": QColor("#a78bfa"), "GOOD": GREEN, "MISS": RED}

BUTTON_STYLE = """
QPushButton { background: rgba(18, 30, 52, 200); color: #e6f1ff; border: 1px solid #1d3354;
              border-radius: 10px; padding: 5px 12px; }
QPushButton:hover { border-color: #a78bfa; }
QPushButton:checked { border: 2px solid #22d3ee; color: #22d3ee; }
QPushButton:disabled { color: #4a5a78; }
"""


class RhythmCanvas(QWidget):
    lane_pressed = Signal(int)

    def __init__(self, game: RhythmGame, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.game = game
        self.setMinimumHeight(260)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._flash: dict[int, float] = {}
        self.notice = "Elige una canción o el ejercicio y pulsa ▶ Empezar"
        self.clock = time.perf_counter
        self._judged_at = -10.0

    def flash_lane(self, lane: int) -> None:
        self._flash[lane] = self.clock()
        self.update()

    def lane_rect(self, lane: int) -> QRectF:
        w = self.width() * 0.62
        left = (self.width() - w) / 2
        lw = w / LANES
        return QRectF(left + lane * lw, 0, lw, self.height())

    def lane_at(self, x: float) -> int | None:
        for lane in range(LANES):
            r = self.lane_rect(lane)
            if r.left() <= x < r.right():
                return lane
        return None

    def mousePressEvent(self, event) -> None:  # noqa: N802
        lane = self.lane_at(event.position().x())
        if lane is not None:
            self.lane_pressed.emit(lane)

    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        paint_grid(p, QRectF(self.rect()))
        now = self.clock()
        hit_y = self.height() * HIT_LINE
        for lane in range(LANES):
            r = self.lane_rect(lane)
            color = LANE_COLORS[lane]
            lane_bg = QColor(color)
            lane_bg.setAlpha(18)
            p.fillRect(r.adjusted(2, 0, -2, 0), lane_bg)
            age = now - self._flash.get(lane, -10.0)
            if age < 0.18:
                glow(p, QPointF(r.center().x(), hit_y), r.width() * 0.8, color, 1.0 - age / 0.18)
            p.setPen(QPen(color, 3))
            p.drawLine(QPointF(r.left() + 6, hit_y), QPointF(r.right() - 6, hit_y))
            p.setPen(TEXT_DIM)
            p.setFont(QFont("Helvetica", 12, QFont.Weight.Bold))
            p.drawText(QRectF(r.left(), hit_y + 8, r.width(), 24), int(Qt.AlignmentFlag.AlignCenter), LANE_KEYS[lane])
        game = self.game
        if game.running:
            t = game.song_time(now)
            for note in game.upcoming(now, LOOKAHEAD_S):
                r = self.lane_rect(note.lane)
                y = hit_y - (note.t - t) / LOOKAHEAD_S * hit_y
                color = LANE_COLORS[note.lane]
                rect = QRectF(r.left() + 10, y - 9, r.width() - 20, 18)
                glow(p, rect.center(), r.width() * 0.3, color, 0.35)
                p.setPen(QPen(QColor("#ffffff"), 1.5))
                p.setBrush(color)
                p.drawRoundedRect(rect, 7, 7)
                p.setPen(QColor("#070b14"))
                p.setFont(QFont("Helvetica", 9, QFont.Weight.Bold))
                p.drawText(rect, int(Qt.AlignmentFlag.AlignCenter), midi_to_note_name(note.midi))
            if t < 0:
                p.setPen(TEXT)
                p.setFont(QFont("Helvetica", 48, QFont.Weight.Black))
                p.drawText(QRectF(self.rect()), int(Qt.AlignmentFlag.AlignCenter), f"{-t:.0f}" if t < -0.5 else "¡YA!")
        if game.judgments and now - self._last_judgment_time() < 0.6:
            j = game.judgments[-1]
            p.setPen(GRADE_COLORS[j.grade])
            p.setFont(QFont("Helvetica", 30, QFont.Weight.Black))
            extra = "" if j.offset_s is None else f"  {j.offset_s * 1000:+.0f} ms"
            p.drawText(QRectF(0, hit_y * 0.45, self.width(), 50), int(Qt.AlignmentFlag.AlignCenter), j.grade + extra)
        self._draw_hud(p)
        if not game.running and self.notice:
            p.setPen(TEXT)
            p.setFont(QFont("Helvetica", 15, QFont.Weight.Bold))
            box = QRectF(self.rect()).adjusted(20, 0, -20, 0)
            p.fillRect(QRectF(box.left(), box.center().y() - 20, box.width(), 40), QColor(7, 11, 20, 210))
            p.drawText(box, int(Qt.AlignmentFlag.AlignCenter), self.notice)
        p.end()

    def _last_judgment_time(self) -> float:
        return self._judged_at

    def judged(self) -> None:
        self._judged_at = self.clock()

    def _draw_hud(self, p: QPainter) -> None:
        game = self.game
        acc = game.accuracy
        lines = [
            ("PUNTOS", f"{game.score:,}".replace(",", " ")),
            ("COMBO", f"{game.combo}  (máx {game.max_combo})"),
            ("PRECISIÓN", "—" if acc is None else f"{acc * 100:.1f} %"),
        ]
        p.setFont(QFont("Helvetica", 10, QFont.Weight.Bold))
        y = 18.0
        for label, value in lines:
            p.setPen(TEXT_DIM)
            p.drawText(QPointF(16, y), label)
            p.setPen(TEXT)
            p.setFont(QFont("Helvetica", 16, QFont.Weight.Black))
            p.drawText(QPointF(16, y + 22), value)
            p.setFont(QFont("Helvetica", 10, QFont.Weight.Bold))
            y += 52
        counts = "   ".join(f"{k[0]}{k[1:].lower()} {v}" for k, v in game.counts.items())
        p.setPen(TEXT_DIM)
        p.drawText(QPointF(16, y), counts)
        if game.chart is not None:  # in the left column, under the counts: never over the lanes
            p.setPen(TEXT)
            p.drawText(QRectF(16, y + 8, self.lane_rect(0).left() - 24, 40),
                       int(Qt.AlignmentFlag.AlignLeft | Qt.TextFlag.TextWordWrap), game.chart.title)


class RhythmView(QWidget):
    start_requested = Signal(object)  # Chart
    stop_requested = Signal()

    def __init__(self, game: RhythmGame, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setStyleSheet("background: #070b14;")
        self.game = game
        self.canvas = RhythmCanvas(game)
        bar = QHBoxLayout()
        bar.setContentsMargins(8, 6, 8, 6)
        title = QLabel("JUEGO RÍTMICO 4K")
        title.setStyleSheet("color: #22d3ee; font-weight: 900; letter-spacing: 2px;")
        bar.addWidget(title)
        self.slot_buttons: list[QPushButton] = []
        for i in range(4):
            b = QPushButton(f"{i + 1} · Hueco libre")
            b.setStyleSheet(BUTTON_STYLE)
            b.setCheckable(True)
            b.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            b.clicked.connect(lambda _=False, i=i: self._select(i))
            self.slot_buttons.append(b)
            bar.addWidget(b)
        self.practice_button = QPushButton("Ejercicio")
        self.practice_button.setStyleSheet(BUTTON_STYLE)
        self.practice_button.setCheckable(True)
        self.practice_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.practice_button.clicked.connect(lambda: self._select(None))
        bar.addWidget(self.practice_button)
        bar.addStretch(1)
        self.start_button = QPushButton("▶ Empezar")
        self.start_button.setStyleSheet(BUTTON_STYLE)
        self.start_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.start_button.clicked.connect(self._start_stop)
        bar.addWidget(self.start_button)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addLayout(bar)
        layout.addWidget(self.canvas, 1)
        self._slots: list[SongSlot] = []
        self.selected: Chart | None = practice_pattern()
        self.practice_button.setChecked(True)
        self._timer = QTimer(self)
        self._timer.setInterval(16)
        self._timer.timeout.connect(self.canvas.update)

    def set_slots(self, slots: list[SongSlot]) -> None:
        self._slots = slots
        for b, slot in zip(self.slot_buttons, slots, strict=False):
            b.setText(f"{slot.index + 1} · {slot.label}")
            b.setEnabled(slot.chart is not None)
            b.setToolTip(slot.error or (slot.chart.source if slot.chart else "Añade un .mid o .json en songs/"))
        if not any(s.chart for s in slots):
            self.canvas.notice = "Sin canciones: añade archivos .mid o .json en songs/ · prueba el Ejercicio"

    def _select(self, index: int | None) -> None:
        for i, b in enumerate(self.slot_buttons):
            b.setChecked(i == index)
        self.practice_button.setChecked(index is None)
        if index is None:
            self.selected = practice_pattern()
        elif index < len(self._slots) and self._slots[index].chart is not None:
            self.selected = self._slots[index].chart

    def _start_stop(self) -> None:
        if self.game.running:
            self.stop_requested.emit()
        elif self.selected is not None:
            self.start_requested.emit(self.selected)

    def set_running(self, running: bool) -> None:
        self.start_button.setText("■ Parar" if running else "▶ Empezar")
        if running:
            self._timer.start()
        else:
            self._timer.stop()
            self.canvas.update()
