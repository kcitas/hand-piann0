"""Holo-Grid visual language shared by the instrument panel, the note pads and the rhythm game.

Deep navy background with a faint perspective-free grid, neon cyan / violet
accents with soft glows, translucent cards. Painting helpers only; no state.
"""

from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QLinearGradient, QPainter, QPen, QRadialGradient

BG = QColor("#0F1426")  # theme.INK
BG_2 = QColor("#171E36")  # theme.STAGE
GRID = QColor(243, 238, 227, 10)  # faint ivory
GRID_MAJOR = QColor(243, 238, 227, 22)
CYAN = QColor("#22d3ee")
VIOLET = QColor("#a78bfa")
PINK = QColor("#f472b6")
AMBER = QColor("#fbbf24")
GREEN = QColor("#34d399")
RED = QColor("#f87171")
TEXT = QColor("#e6f1ff")
TEXT_DIM = QColor("#7a8bab")
LANE_COLORS = (CYAN, VIOLET, PINK, AMBER)

from handpiano.ui.theme import FONT, INK, IVORY, LINE, MUTED, SPOTLIGHT, STAGE  # noqa: E402

STYLE = f"""
QWidget#HoloPanel {{ background: {INK}; }}
QFrame#InstrumentCard {{ background: {STAGE}; border: 1px solid {LINE}; border-radius: 12px; }}
QFrame#InstrumentCard[active="true"] {{ border: 2px solid {SPOTLIGHT}; }}
QFrame#InstrumentCard:hover {{ border-color: {MUTED}; }}
QLabel#CardName {{ color: {IVORY}; font-family: "{FONT}"; font-size: 16px; font-weight: 700; background: transparent; }}
QLabel#KeyCap {{ color: {MUTED}; border: 1px solid {LINE}; border-radius: 5px; padding: 0 6px; font-size: 11px;
                 background: transparent; }}
QLabel#CardDesc {{ color: {MUTED}; font-family: "{FONT}"; font-size: 12px; background: transparent; }}
QLabel#CardState {{ color: {MUTED}; font-size: 11px; background: transparent; }}
QLabel#ActiveChip {{ color: {INK}; background: {SPOTLIGHT}; border-radius: 8px; padding: 1px 9px;
                    font-size: 11px; font-weight: 700; }}
QPushButton#TestButton {{ background: transparent; color: {IVORY}; border: 1px solid {LINE};
                         border-radius: 10px; padding: 3px 12px; font-weight: 600; }}
QPushButton#TestButton:hover {{ border-color: {SPOTLIGHT}; color: {SPOTLIGHT}; }}
QLabel#VolumeLabel {{ color: {MUTED}; font-size: 12px; background: transparent; }}
QSlider#Volume::groove:horizontal {{ height: 4px; background: {LINE}; border-radius: 2px; }}
QSlider#Volume::sub-page:horizontal {{ background: {SPOTLIGHT}; border-radius: 2px; }}
QSlider#Volume::handle:horizontal {{ background: {IVORY}; width: 14px; margin: -6px 0; border-radius: 7px; }}
"""


def paint_grid(painter: QPainter, rect: QRectF, cell: float = 28.0) -> None:
    """Background gradient plus a fine grid with a brighter line every 4 cells."""
    gradient = QLinearGradient(rect.topLeft(), rect.bottomLeft())
    gradient.setColorAt(0.0, BG_2)
    gradient.setColorAt(1.0, BG)
    painter.fillRect(rect, gradient)
    x, i = rect.left(), 0
    while x <= rect.right():
        painter.setPen(QPen(GRID_MAJOR if i % 4 == 0 else GRID, 1))
        painter.drawLine(QPointF(x, rect.top()), QPointF(x, rect.bottom()))
        x += cell
        i += 1
    y, i = rect.top(), 0
    while y <= rect.bottom():
        painter.setPen(QPen(GRID_MAJOR if i % 4 == 0 else GRID, 1))
        painter.drawLine(QPointF(rect.left(), y), QPointF(rect.right(), y))
        y += cell
        i += 1


def glow(painter: QPainter, center: QPointF, radius: float, color: QColor, strength: float = 1.0) -> None:
    """Soft radial glow (drawn under a shape)."""
    g = QRadialGradient(center, radius)
    c = QColor(color)
    c.setAlpha(int(150 * max(0.0, min(1.0, strength))))
    g.setColorAt(0.0, c)
    c2 = QColor(color)
    c2.setAlpha(0)
    g.setColorAt(1.0, c2)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(g)
    painter.drawEllipse(center, radius, radius)
