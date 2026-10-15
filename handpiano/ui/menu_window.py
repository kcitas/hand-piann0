"""Start menu: one big card per way of playing. Choosing one opens it; closing it comes back here."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QGridLayout, QLabel, QPushButton, QVBoxLayout, QWidget

# (id, title, how to play) in menu order. The ids are what ``chosen`` emits.
MODES: tuple[tuple[str, str, str], ...] = (
    ("libre", "🎹  Piano Libre",
     "Levanta las manos con las palmas hacia la cámara y los dedos hacia arriba.\n"
     "Cada dedo es una nota (Do Re Mi Fa · Sol La Si Do): dóblalo para tocarla."),
    ("juego", "🎮  Juego Rítmico",
     "Las notas caen por 4 carriles: dobla el dedo de su carril cuando lleguen a la línea.\n"
     "Medio izq. · índice izq. · índice der. · medio der. (o D F J K)."),
    ("dibujado", "🖐  Piano dibujado",
     "Un piano de 2 octavas dibujado sobre el vídeo.\n"
     "Manos sobre la mesa: baja un dedo encima de una tecla para tocarla."),
    ("sonido", "🔊  Prueba de sonido",
     "Sin cámara: los tres pianos con el teclado del ordenador y el ratón."),
)

STYLE = """
QWidget#Menu { background: #070b14; }
QLabel#Title { color: #e6f1ff; font-size: 40px; font-weight: 800; letter-spacing: 2px; }
QLabel#Subtitle { color: #7a8bab; font-size: 15px; }
QPushButton#ModeCard {
    background: rgba(18, 30, 52, 220); border: 1px solid #1d3354; border-radius: 16px;
    color: #e6f1ff; text-align: left; padding: 22px 26px; font-size: 14px;
}
QPushButton#ModeCard:hover { border: 2px solid #22d3ee; background: rgba(16, 46, 66, 235); }
QLabel#CardTitle { color: #e6f1ff; font-size: 22px; font-weight: 800; background: transparent; }
QLabel#CardHow { color: #9fb2d4; font-size: 14px; background: transparent; }
QPushButton#Quit { background: transparent; color: #7a8bab; border: 1px solid #1d3354; border-radius: 10px;
                   padding: 8px 22px; font-size: 13px; }
QPushButton#Quit:hover { color: #f87171; border-color: #f87171; }
"""


class MenuWindow(QWidget):
    chosen = Signal(str)  # a MODES id
    quit_requested = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("Menu")
        self.setWindowTitle("HandPiano")
        self.setStyleSheet(STYLE)
        self.resize(1000, 680)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(48, 40, 48, 32)
        layout.setSpacing(10)
        title = QLabel("HandPiano")
        title.setObjectName("Title")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle = QLabel("Un piano que se toca con las manos delante de la cámara · elige cómo quieres tocar")
        subtitle.setObjectName("Subtitle")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addSpacing(18)
        grid = QGridLayout()
        grid.setSpacing(18)
        self.buttons: dict[str, QPushButton] = {}
        for i, (mode, name, how) in enumerate(MODES):
            button = QPushButton()
            button.setObjectName("ModeCard")
            button.setAccessibleName(f"{name}. {how}")
            card = QVBoxLayout(button)
            card.setContentsMargins(26, 22, 26, 22)
            for text, style in ((name, "CardTitle"), (how, "CardHow")):
                label = QLabel(text)
                label.setObjectName(style)
                label.setWordWrap(True)
                label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
                card.addWidget(label)
            card.addStretch(1)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.setMinimumHeight(170)
            button.clicked.connect(lambda _=False, m=mode: self.chosen.emit(m))
            grid.addWidget(button, i // 2, i % 2)
            self.buttons[mode] = button
        layout.addLayout(grid, 1)
        layout.addSpacing(12)
        quit_button = QPushButton("Salir")
        quit_button.setObjectName("Quit")
        quit_button.clicked.connect(self.quit_requested.emit)
        layout.addWidget(quit_button, 0, Qt.AlignmentFlag.AlignCenter)

    def closeEvent(self, event) -> None:  # noqa: N802
        self.quit_requested.emit()
        super().closeEvent(event)
