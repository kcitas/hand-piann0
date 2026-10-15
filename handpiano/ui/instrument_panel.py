"""Instrument selector: three cards (name, description, state, active indicator, test note) and volume.

Pure view: it emits signals and shows the state it is given; the AudioEngine is
wired in by the application (so the panel works with any engine, or none).
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from handpiano.audio.instruments import INSTRUMENTS
from handpiano.ui.holo import STYLE


class InstrumentCard(QFrame):
    clicked = Signal(str)
    test_clicked = Signal(str)

    def __init__(self, instrument_id: str, shortcut: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.instrument_id = instrument_id
        instrument = INSTRUMENTS[instrument_id]
        self.setObjectName("InstrumentCard")
        self.setProperty("active", False)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(3)
        top = QHBoxLayout()
        self.name = QLabel(instrument.name)
        self.name.setObjectName("CardName")
        top.addWidget(self.name)
        key = QLabel(shortcut)
        key.setObjectName("KeyCap")
        key.setToolTip(f"Tecla {shortcut}")
        top.addWidget(key)
        top.addStretch(1)
        self.chip = QLabel("Sonando")
        self.chip.setObjectName("ActiveChip")
        self.chip.setVisible(False)
        top.addWidget(self.chip)
        layout.addLayout(top)
        self.desc = QLabel(instrument.description)
        self.desc.setObjectName("CardDesc")
        self.desc.setWordWrap(True)
        layout.addWidget(self.desc)
        bottom = QHBoxLayout()
        self.state = QLabel("—")
        self.state.setObjectName("CardState")
        bottom.addWidget(self.state, 1)
        self.test = QPushButton("▶ Nota de prueba")
        self.test.setObjectName("TestButton")
        self.test.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.test.clicked.connect(lambda: self.test_clicked.emit(self.instrument_id))
        bottom.addWidget(self.test)
        layout.addLayout(bottom)

    def set_active(self, active: bool) -> None:
        if self.property("active") != active:
            self.setProperty("active", active)
            self.chip.setVisible(active)
            self.style().unpolish(self)
            self.style().polish(self)

    @property
    def active(self) -> bool:
        return bool(self.property("active"))

    def set_state(self, text: str) -> None:
        self.state.setText(text)

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self.instrument_id)
        super().mousePressEvent(event)


class InstrumentPanel(QWidget):
    instrument_selected = Signal(str)
    test_requested = Signal(str)
    volume_changed = Signal(float)  # 0.0 … 1.0

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("HoloPanel")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setStyleSheet(STYLE)
        row = QHBoxLayout(self)
        row.setContentsMargins(10, 8, 10, 8)
        row.setSpacing(10)
        self.cards: dict[str, InstrumentCard] = {}
        for i, instrument_id in enumerate(INSTRUMENTS, start=1):
            card = InstrumentCard(instrument_id, str(i))
            card.clicked.connect(self.instrument_selected.emit)
            card.test_clicked.connect(self.test_requested.emit)
            self.cards[instrument_id] = card
            row.addWidget(card, 1)
        volume_box = QVBoxLayout()
        self.volume_label = QLabel("Volumen 50 %")
        self.volume_label.setObjectName("VolumeLabel")
        volume_box.addWidget(self.volume_label)
        self.volume = QSlider(Qt.Orientation.Horizontal)
        self.volume.setObjectName("Volume")
        self.volume.setRange(0, 100)
        self.volume.setValue(80)
        self.volume.setFixedWidth(140)
        self.volume.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.volume.valueChanged.connect(self._on_volume)
        volume_box.addWidget(self.volume)
        self.output_label = QLabel("")
        self.output_label.setObjectName("CardState")
        self.output_label.setWordWrap(True)
        self.output_label.setFixedWidth(160)
        volume_box.addWidget(self.output_label)
        row.addLayout(volume_box)

    def _on_volume(self, value: int) -> None:
        self.volume_label.setText(f"Volumen {value} %")
        self.volume_changed.emit(value / 100.0)

    def set_volume(self, value: float) -> None:
        """Shows ``value`` (0–1) without emitting volume_changed."""
        blocked = self.volume.blockSignals(True)
        self.volume.setValue(round(value * 100))
        self.volume.blockSignals(blocked)
        self.volume_label.setText(f"Volumen {round(value * 100)} %")

    def set_active(self, instrument_id: str) -> None:
        for i, card in self.cards.items():
            card.set_active(i == instrument_id)

    def set_card_state(self, instrument_id: str, text: str) -> None:
        self.cards[instrument_id].set_state(text)

    def set_output(self, text: str) -> None:
        self.output_label.setText(text)

    @property
    def active(self) -> str | None:
        return next((i for i, c in self.cards.items() if c.active), None)
