"""Main window: camera controls, live view, status line and the Tracking Lab dock."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDockWidget,
    QLabel,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QStackedWidget,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from handpiano.camera.camera_config import (
    FPS_PRESETS,
    RESOLUTION_PRESETS,
    CameraSettings,
)
from handpiano.game.rhythm import RhythmGame
from handpiano.instrument.keyboard import KeyboardLayout
from handpiano.metrics.performance import CpuAccounting
from handpiano.ui.camera_view import CameraView
from handpiano.ui.debug_view import DebugView
from handpiano.ui.instrument_panel import InstrumentPanel
from handpiano.ui.piano_view import PianoView
from handpiano.ui.rhythm_view import RhythmView

ICON_PATH = Path(__file__).parent / "assets" / "icon.svg"

STYLE = """
QMainWindow, QWidget { background: #0b0b12; color: #ecebf5; }
QToolBar { border: none; spacing: 8px; padding: 6px; background: #14141f; }
QComboBox, QPushButton { background: #1f1f2e; border: 1px solid #2c2c40; border-radius: 6px; padding: 4px 10px; }
QPushButton:hover { border-color: #8b5cf6; }
QGroupBox { border: 1px solid #1f1f2e; border-radius: 8px; margin-top: 14px; padding-top: 6px; }
QGroupBox::title { subcontrol-origin: margin; left: 8px; color: #a78bfa; }
QTableWidget { background: #0b0b12; gridline-color: #1f1f2e; }
QHeaderView::section { background: #14141f; color: #a1a1b8; border: none; padding: 4px; }
QStatusBar { background: #14141f; color: #a1a1b8; }
"""


class MainWindow(QMainWindow):
    apply_camera_requested = Signal(object)  # CameraSettings
    probe_requested = Signal()
    calibrate_requested = Signal()
    forget_calibration_requested = Signal()
    # "capture" (Space), "undo" (Backspace), "cancel" (Esc) while calibrating.
    calibration_key = Signal(str)
    # "piano" (Piano Libre) or "rhythm" (Juego Rítmico 4K).
    mode_changed = Signal(str)

    def __init__(
        self, settings: CameraSettings, cpu: CpuAccounting | None = None, layout: KeyboardLayout | None = None
    ) -> None:
        super().__init__()
        self.setWindowTitle("HandPiano")
        self.resize(1280, 800)
        self.setStyleSheet(STYLE)
        self._settings = settings

        self.camera_view = CameraView(cpu=cpu)
        self.piano_view = PianoView(layout or KeyboardLayout.from_octaves())
        self.rhythm_game = RhythmGame()
        self.rhythm_view = RhythmView(self.rhythm_game)
        self.instrument_panel = InstrumentPanel()
        # Mode view (free piano keyboard map or rhythm lanes) on top, camera image below.
        self.mode_stack = QStackedWidget()
        self.mode_stack.addWidget(self.piano_view)
        self.mode_stack.addWidget(self.rhythm_view)
        splitter = QSplitter(Qt.Orientation.Vertical)
        self._splitter = splitter
        splitter.addWidget(self.mode_stack)
        splitter.addWidget(self.camera_view)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 4)
        central = QWidget()
        column = QVBoxLayout(central)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(0)
        column.addWidget(self.instrument_panel)
        column.addWidget(splitter, 1)
        self.setCentralWidget(central)
        self.mode = "piano"

        self.debug_view = DebugView()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.debug_view)
        self._dock = QDockWidget("Tracking Lab", self)
        self._dock.setWidget(scroll)
        self._dock.setMinimumWidth(420)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self._dock)
        self._dock.hide()

        self.status_label = QLabel("Iniciando…")
        self.statusBar().addWidget(self.status_label, 1)
        # Shown only while a trace records physical keys (keys protocol).
        self.trace_keys_label = QLabel("")
        self.trace_keys_label.setStyleSheet("color: #fbbf24; font-weight: 700; padding: 0 8px;")
        self.trace_keys_label.setVisible(False)
        self.statusBar().addPermanentWidget(self.trace_keys_label)

        self._build_toolbar()
        self._install_shortcuts()

    def _build_toolbar(self) -> None:
        bar = QToolBar("Cámara")
        bar.setMovable(False)
        self.addToolBar(bar)
        title = QLabel("  HandPiano  ")
        title.setStyleSheet("font-weight: 700; font-size: 15px;")
        bar.addWidget(title)

        self.camera_combo = QComboBox()
        self.set_cameras([self._settings.device_index])
        bar.addWidget(self.camera_combo)

        probe = QPushButton("Buscar cámaras")
        probe.clicked.connect(self.probe_requested.emit)
        bar.addWidget(probe)

        self.resolution_combo = QComboBox()
        for w, h in RESOLUTION_PRESETS:
            self.resolution_combo.addItem(f"{w}×{h}", f"{w}x{h}")
        if self.resolution_combo.findData(f"{self._settings.width}x{self._settings.height}") < 0:
            # A non-preset resolution requested from the command line.
            self.resolution_combo.addItem(
                f"{self._settings.width}×{self._settings.height}", f"{self._settings.width}x{self._settings.height}"
            )
        self._select_data(self.resolution_combo, f"{self._settings.width}x{self._settings.height}")
        bar.addWidget(self.resolution_combo)

        self.fps_combo = QComboBox()
        for fps in FPS_PRESETS:
            self.fps_combo.addItem(f"{fps} fps solicitados", fps)
        self._select_data(self.fps_combo, self._settings.fps)
        bar.addWidget(self.fps_combo)

        apply = QPushButton("Aplicar")
        apply.clicked.connect(self._on_apply)
        bar.addWidget(apply)

        bar.addSeparator()
        self.mode_buttons: dict[str, QPushButton] = {}
        group = QButtonGroup(self)
        group.setExclusive(True)
        for mode, text in (("piano", "Piano Libre"), ("rhythm", "Juego Rítmico 4K")):
            button = QPushButton(text)
            button.setCheckable(True)
            button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            button.setStyleSheet("QPushButton:checked { border: 2px solid #22d3ee; color: #22d3ee; }")
            button.clicked.connect(lambda _=False, m=mode: self.set_mode(m))
            group.addButton(button)
            bar.addWidget(button)
            self.mode_buttons[mode] = button
        self.mode_buttons["piano"].setChecked(True)

        bar.addSeparator()
        self.calibrate_button = QPushButton("Calibrar piano")
        self.calibrate_button.clicked.connect(self.calibrate_requested.emit)
        bar.addWidget(self.calibrate_button)
        self.forget_button = QPushButton("Olvidar calibración")
        self.forget_button.clicked.connect(self.forget_calibration_requested.emit)
        bar.addWidget(self.forget_button)
        for button in (probe, apply, self.calibrate_button, self.forget_button):
            button.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        bar.addWidget(spacer)

        self.lab_checkbox = QCheckBox("Tracking Lab")
        self.lab_checkbox.toggled.connect(self.set_lab_mode)
        bar.addWidget(self.lab_checkbox)

    @staticmethod
    def _select_data(combo: QComboBox, data) -> None:
        index = combo.findData(data)
        if index >= 0:
            combo.setCurrentIndex(index)

    def set_cameras(self, indices: list[int]) -> None:
        current = self.camera_combo.currentData()
        self.camera_combo.clear()
        for index in indices:
            self.camera_combo.addItem(f"Cámara {index}", index)
        self._select_data(self.camera_combo, current if current is not None else self._settings.device_index)

    def set_mode(self, mode: str) -> None:
        if mode not in self.mode_buttons:
            raise ValueError(mode)
        self.mode_buttons[mode].setChecked(True)
        self.mode_stack.setCurrentWidget(self.piano_view if mode == "piano" else self.rhythm_view)
        # The lanes need height to read the falling notes; the keyboard map does not.
        total = max(1, sum(self._splitter.sizes()))
        top = 0.6 if mode == "rhythm" else 0.2
        self._splitter.setSizes([int(total * top), total - int(total * top)])
        if mode != self.mode:
            self.mode = mode
            self.mode_changed.emit(mode)

    def set_lab_mode(self, enabled: bool) -> None:
        if self.lab_checkbox.isChecked() != enabled:
            self.lab_checkbox.setChecked(enabled)
        self._dock.setVisible(enabled)
        self.camera_view.set_lab_mode(enabled)
        self.piano_view.set_lab_mode(enabled)

    def _install_shortcuts(self) -> None:
        # Window-wide shortcuts, so Space captures a corner even if a toolbar widget has focus.
        for key, command in ((Qt.Key.Key_Space, "capture"), (Qt.Key.Key_Backspace, "undo"),
                             (Qt.Key.Key_Escape, "cancel")):
            shortcut = QShortcut(QKeySequence(key), self)
            shortcut.setContext(Qt.ShortcutContext.WindowShortcut)
            shortcut.activated.connect(lambda c=command: self.calibration_key.emit(c))

    def _on_apply(self) -> None:
        device = self.camera_combo.currentData()
        width, height = (int(v) for v in self.resolution_combo.currentData().split("x"))
        self._settings = CameraSettings(
            device_index=self._settings.device_index if device is None else int(device),
            width=width,
            height=height,
            fps=int(self.fps_combo.currentData()),
            mirror=self._settings.mirror,
            exposure=self._settings.exposure,
            autofocus=self._settings.autofocus,
        )
        self.apply_camera_requested.emit(self._settings)
