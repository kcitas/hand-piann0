"""Main window: camera controls, live view, status line and the Tracking Lab dock."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDockWidget,
    QLabel,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QToolBar,
    QWidget,
)

from handpiano.camera.camera_config import FPS_PRESETS, RESOLUTION_PRESETS, CameraSettings
from handpiano.ui.camera_view import CameraView
from handpiano.ui.debug_view import DebugView

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

    def __init__(self, settings: CameraSettings) -> None:
        super().__init__()
        self.setWindowTitle("HandPiano")
        self.resize(1280, 800)
        self.setStyleSheet(STYLE)
        self._settings = settings

        self.camera_view = CameraView()
        self.setCentralWidget(self.camera_view)

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

        self._build_toolbar()

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

    def set_lab_mode(self, enabled: bool) -> None:
        if self.lab_checkbox.isChecked() != enabled:
            self.lab_checkbox.setChecked(enabled)
        self._dock.setVisible(enabled)
        self.camera_view.set_lab_mode(enabled)

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
