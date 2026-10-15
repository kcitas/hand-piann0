"""Holo-Grid visual language shared by the instrument panel, the note pads and the rhythm game.

Deep navy background with a faint perspective-free grid, neon cyan / violet
accents with soft glows, translucent cards. Painting helpers only; no state.
"""

from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QLinearGradient, QPainter, QPen, QRadialGradient

BG = QColor("#070b14")
BG_2 = QColor("#0c1424")
GRID = QColor(34, 211, 238, 22)
GRID_MAJOR = QColor(34, 211, 238, 45)
CYAN = QColor("#22d3ee")
VIOLET = QColor("#a78bfa")
PINK = QColor("#f472b6")
AMBER = QColor("#fbbf24")
GREEN = QColor("#34d399")
RED = QColor("#f87171")
TEXT = QColor("#e6f1ff")
TEXT_DIM = QColor("#7a8bab")
LANE_COLORS = (CYAN, VIOLET, PINK, AMBER)

STYLE = """
QWidget#HoloPanel { background: #070b14; }
QFrame#InstrumentCard {
    background: rgba(18, 30, 52, 190); border: 1px solid #1d3354; border-radius: 12px;
}
QFrame#InstrumentCard[active="true"] { border: 2px solid #22d3ee; background: rgba(16, 46, 66, 220); }
QFrame#InstrumentCard:hover { border-color: #a78bfa; }
QLabel#CardName { color: #e6f1ff; font-size: 15px; font-weight: 700; letter-spacing: 1px; background: transparent; }
QLabel#CardDesc { color: #9fb2d4; font-size: 11px; background: transparent; }
QLabel#CardState { color: #7a8bab; font-size: 10px; background: transparent; }
QLabel#ActiveChip { color: #070b14; background: #22d3ee; border-radius: 8px; padding: 1px 8px;
                    font-size: 10px; font-weight: 800; }
QPushButton#TestButton { background: transparent; color: #22d3ee; border: 1px solid #22d3ee;
                         border-radius: 10px; padding: 3px 10px; font-weight: 700; }
QPushButton#TestButton:hover { background: rgba(34, 211, 238, 40); }
QLabel#VolumeLabel { color: #9fb2d4; font-size: 11px; background: transparent; }
QSlider#Volume::groove:horizontal { height: 4px; background: #1d3354; border-radius: 2px; }
QSlider#Volume::sub-page:horizontal { background: #22d3ee; border-radius: 2px; }
QSlider#Volume::handle:horizontal { background: #e6f1ff; width: 14px; margin: -6px 0; border-radius: 7px; }
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
