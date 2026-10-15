"""HandPiano's look: a concert stage at night.

Ink-blue room, ivory keys, and a single amber spotlight reserved for what is sounding or active.
The hand colours (cyan = left, violet = right) are information, not decoration, and stay as they are.
"""

from __future__ import annotations

from PySide6.QtGui import QColor

INK = "#0F1426"  # the room: window background
STAGE = "#171E36"  # raised surfaces: toolbar, cards, panels
LINE = "#2A3456"  # borders and separators
IVORY = "#F3EEE3"  # keys and main text
MUTED = "#8E98B8"  # secondary text
SPOTLIGHT = "#F2B33D"  # what sounds / what is selected — used for nothing else
LEFT_HAND = "#22D3EE"
RIGHT_HAND = "#A78BFA"

FONT = "Avenir Next"  # ships with macOS; Qt falls back to the system sans elsewhere

# Main window (toolbar, mode switch, status bar). Child widgets with their own style keep it.
WINDOW_STYLE = f"""
QMainWindow, QWidget {{ background: {INK}; color: {IVORY}; font-family: "{FONT}"; font-size: 13px; }}
QToolBar {{ border: none; border-bottom: 1px solid {LINE}; spacing: 6px; padding: 8px 10px; background: {STAGE}; }}
QToolBar QLabel#AppTitle {{ font-size: 17px; font-weight: 700; padding: 0 10px 0 4px; background: transparent; }}
QComboBox, QPushButton {{ background: {INK}; border: 1px solid {LINE}; border-radius: 8px; padding: 5px 12px; }}
QPushButton:hover, QComboBox:hover {{ border-color: {MUTED}; }}
QPushButton:focus, QComboBox:focus {{ border-color: {SPOTLIGHT}; }}
QPushButton#ModeButton {{ border-radius: 14px; padding: 5px 16px; font-weight: 600; }}
QPushButton#ModeButton:checked {{ background: {SPOTLIGHT}; color: {INK}; border-color: {SPOTLIGHT}; }}
QPushButton#MenuButton {{ font-weight: 600; }}
QCheckBox {{ background: transparent; padding: 0 6px; }}
QGroupBox {{ border: 1px solid {LINE}; border-radius: 10px; margin-top: 14px; padding-top: 6px; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 10px; color: {MUTED}; }}
QTableWidget {{ background: {INK}; gridline-color: {LINE}; }}
QHeaderView::section {{ background: {STAGE}; color: {MUTED}; border: none; padding: 4px; }}
QStatusBar {{ background: {STAGE}; color: {MUTED}; border-top: 1px solid {LINE}; }}
QSplitter::handle {{ background: {LINE}; height: 2px; }}
QDockWidget {{ color: {MUTED}; }}
"""


def qcolor(hex_color: str, alpha: int = 255) -> QColor:
    color = QColor(hex_color)
    color.setAlpha(alpha)
    return color
