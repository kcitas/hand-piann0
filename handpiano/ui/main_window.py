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
from handpiano.ui.finger_keys_view import FingerKeysView
from handpiano.ui.instrument_panel import InstrumentPanel
from handpiano.ui.piano_view import PianoView
from handpiano.ui.rhythm_view import RhythmView
from handpiano.ui.theme import WINDOW_STYLE

ICON_PATH = Path(__file__).parent / "assets" / "icon.svg"

STYLE = WINDOW_STYLE


class MainWindow(QMainWindow):
    apply_camera_requested = Signal(object)  # CameraSettings
    probe_requested = Signal()
    calibrate_requested = Signal()
    forget_calibration_requested = Signal()
    # "capture" (Space), "undo" (Backspace), "cancel" (Esc) while calibrating.
    calibration_key = Signal(str)
    # "piano" (Piano Libre) or "rhythm" (Juego Rítmico 4K).
    mode_changed = Signal(str)
    # The window was closed (or "☰ Menú" pressed): the start menu shows again.
    closed = Signal()

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
        # Finger-note mode replaces the 2-octave map with one key per finger (set_finger_notes).
        self.finger_keys_view = FingerKeysView()
        self._finger_notes = False
        self._calibration_used = True  # False: piano fixed on screen, nothing to calibrate
        self.mode_stack = QStackedWidget()
        self.mode_stack.addWidget(self.piano_view)
        self.mode_stack.addWidget(self.rhythm_view)
        self.mode_stack.addWidget(self.finger_keys_view)
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
        self.menu_button = QPushButton("☰ Menú")
        self.menu_button.setObjectName("MenuButton")
        self.menu_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.menu_button.setVisible(False)  # only when opened from the start menu
        self.menu_button.clicked.connect(self.close)
        self._menu_action = bar.addWidget(self.menu_button)
        self._menu_action.setVisible(False)
        title = QLabel("HandPiano")
        title.setObjectName("AppTitle")
        bar.addWidget(title)

        # Camera controls are for diagnosis: shown with the Tracking Lab only.
        self._camera_actions = []
        self.camera_combo = QComboBox()
        self.set_cameras([self._settings.device_index])
        self._camera_actions.append(bar.addWidget(self.camera_combo))

        probe = QPushButton("Buscar cámaras")
        probe.clicked.connect(self.probe_requested.emit)
        self._camera_actions.append(bar.addWidget(probe))

        self.resolution_combo = QComboBox()
        for w, h in RESOLUTION_PRESETS:
            self.resolution_combo.addItem(f"{w}×{h}", f"{w}x{h}")
        if self.resolution_combo.findData(f"{self._settings.width}x{self._settings.height}") < 0:
            # A non-preset resolution requested from the command line.
            self.resolution_combo.addItem(
                f"{self._settings.width}×{self._settings.height}", f"{self._settings.width}x{self._settings.height}"
            )
        self._select_data(self.resolution_combo, f"{self._settings.width}x{self._settings.height}")
        self._camera_actions.append(bar.addWidget(self.resolution_combo))

        self.fps_combo = QComboBox()
        for fps in FPS_PRESETS:
            self.fps_combo.addItem(f"{fps} fps solicitados", fps)
        self._select_data(self.fps_combo, self._settings.fps)
        self._camera_actions.append(bar.addWidget(self.fps_combo))

        apply = QPushButton("Aplicar")
        apply.clicked.connect(self._on_apply)
        self._camera_actions.append(bar.addWidget(apply))
        self._camera_actions.append(bar.addSeparator())
        for action in self._camera_actions:
            action.setVisible(False)

        self.mode_buttons: dict[str, QPushButton] = {}
        group = QButtonGroup(self)
        group.setExclusive(True)
        for mode, text in (("piano", "Piano libre"), ("rhythm", "Juego rítmico")):
            button = QPushButton(text)
            button.setCheckable(True)
            button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            button.setObjectName("ModeButton")
            button.clicked.connect(lambda _=False, m=mode: self.set_mode(m))
            group.addButton(button)
            bar.addWidget(button)
            self.mode_buttons[mode] = button
        self.mode_buttons["piano"].setChecked(True)

        # Calibration only means something for the drawn keyboard (hidden in finger-note mode).
        self._calibration_actions = [bar.addSeparator()]
        self.calibrate_button = QPushButton("Calibrar piano")
        self.calibrate_button.clicked.connect(self.calibrate_requested.emit)
        self._calibration_actions.append(bar.addWidget(self.calibrate_button))
        self.forget_button = QPushButton("Olvidar calibración")
        self.forget_button.clicked.connect(self.forget_calibration_requested.emit)
        self._calibration_actions.append(bar.addWidget(self.forget_button))
        for button in (probe, apply, self.calibrate_button, self.forget_button):
            button.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        spacer = QWidget()
        spacer.setStyleSheet("background: transparent;")
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

    def enable_menu_button(self) -> None:
        self._menu_action.setVisible(True)
        self.menu_button.setVisible(True)

    def closeEvent(self, event) -> None:  # noqa: N802
        super().closeEvent(event)
        self.closed.emit()

    def set_finger_notes(self, enabled: bool) -> None:
        """Finger-note mode: one key per finger instead of the 2-octave map, and a larger video
        (the player has to see the hands)."""
        self._finger_notes = enabled
        self._show_calibration()
        self.set_mode(self.mode)

    def set_calibration_used(self, used: bool) -> None:
        """Only the table piano (--piano-mesa) is calibrated: otherwise its buttons are hidden."""
        self._calibration_used = used
        self._show_calibration()

    def _show_calibration(self) -> None:
        for action in self._calibration_actions:
            action.setVisible(self._calibration_used and not self._finger_notes)

    def set_mode(self, mode: str) -> None:
        if mode not in self.mode_buttons:
            raise ValueError(mode)
        self.mode_buttons[mode].setChecked(True)
        keys = self.finger_keys_view if self._finger_notes else self.piano_view
        self.mode_stack.setCurrentWidget(keys if mode == "piano" else self.rhythm_view)
        # A stacked widget is as tall as its tallest page; the finger keys need far less than the lanes.
        compact = self._finger_notes and mode == "piano"
        self.mode_stack.setMaximumHeight(self.finger_keys_view.maximumHeight() if compact else 16777215)
        # The lanes need height to read the falling notes; the keyboard map does not.
        # Before the window is shown the splitter has no size yet: use the window's height.
        total = max(sum(self._splitter.sizes()), self.height() - 200, 1)
        top = 0.6 if mode == "rhythm" else 0.18 if self._finger_notes else 0.2
        self._splitter.setSizes([int(total * top), total - int(total * top)])
        if mode != self.mode:
            self.mode = mode
            self.mode_changed.emit(mode)

    def set_lab_mode(self, enabled: bool) -> None:
        if self.lab_checkbox.isChecked() != enabled:
            self.lab_checkbox.setChecked(enabled)
        self._dock.setVisible(enabled)
        for action in self._camera_actions:
            action.setVisible(enabled)
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
