"""Tracking Lab: raw numbers behind the tracking, for debugging and tuning."""

from __future__ import annotations

from PySide6.QtWidgets import QFormLayout, QGroupBox, QLabel, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget

from handpiano.app.pipeline import PipelineStats, TrackingSnapshot
from handpiano.camera.camera_config import CameraMode
from handpiano.metrics.latency import StatsSummary
from handpiano.tracking.landmarks import PLAYING_FINGERS, Handedness
from handpiano.ui.format import NA, fmt, fmt_pct

_FINGER_COLUMNS = ("Dedo", "Estado", "x px", "y px", "z", "vel px/s", "jitter px")

NOT_YET_AVAILABLE = "N/A (fase posterior)"


def _p50_p95(s: StatsSummary) -> str:
    if s.p50 is None:
        return NA
    return f"p50 {s.p50:.1f} · p95 {s.p95:.1f} ms"


class DebugView(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._labels: dict[str, QLabel] = {}
        layout = QVBoxLayout(self)

        layout.addWidget(
            self._group(
                "Cámara",
                ["Dispositivo", "Backend", "Solicitado", "Entregado", "FPS reportado", "FPS medido",
                 "Exposición", "Autofoco"],
            )
        )
        layout.addWidget(
            self._group(
                "Rendimiento",
                ["FPS tracking", "Inferencia", "Procesado", "Espera en cola", "Captura → resultado",
                 "Captura → pantalla", "Pintado UI", "Frames descartados", "CPU proceso"],
            )
        )
        layout.addWidget(
            self._group(
                "Tracking",
                ["Manos", "Score izq.", "Score der.", "Sin mano izq.", "Sin mano der.", "Outliers rechazados",
                 "Latencia audio", "Falsos disparos"],
            )
        )

        self._table = QTableWidget(len(PLAYING_FINGERS), len(_FINGER_COLUMNS))
        self._table.setHorizontalHeaderLabels(_FINGER_COLUMNS)
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        for row, finger in enumerate(PLAYING_FINGERS):
            self._table.setItem(row, 0, QTableWidgetItem(finger.short))
        self._table.resizeColumnsToContents()
        layout.addWidget(self._table, stretch=1)

        # Not measurable until the corresponding layers exist; shown honestly.
        self._labels["Latencia audio"].setText(NOT_YET_AVAILABLE)
        self._labels["Falsos disparos"].setText(NOT_YET_AVAILABLE)

    def _group(self, title: str, keys: list[str]) -> QGroupBox:
        box = QGroupBox(title)
        form = QFormLayout(box)
        for key in keys:
            label = QLabel(NA)
            self._labels[key] = label
            form.addRow(key, label)
        return box

    def _set(self, key: str, text: str) -> None:
        self._labels[key].setText(text)

    def update_camera(self, mode: CameraMode | None) -> None:
        if mode is None:
            for key in ("Dispositivo", "Backend", "Solicitado", "Entregado", "FPS reportado", "Exposición", "Autofoco"):
                self._set(key, NA)
            return
        self._set("Dispositivo", f"Cámara {mode.device_index}")
        self._set("Backend", mode.backend)
        self._set("Solicitado", f"{mode.requested_width}×{mode.requested_height} @ {mode.requested_fps}")
        self._set("Entregado", f"{mode.actual_width}×{mode.actual_height}")
        self._set("FPS reportado", fmt(mode.reported_fps, "fps"))
        self._set("Exposición", "control aceptado" if mode.exposure_supported else "no configurable / no solicitado")
        self._set("Autofoco", "control aceptado" if mode.autofocus_supported else "no configurable / no solicitado")

    def update_stats(
        self,
        stats: PipelineStats,
        capture_to_paint: StatsSummary,
        paint: StatsSummary,
        cpu_percent: float | None,
    ) -> None:
        self._set("FPS medido", fmt(stats.camera_fps, "fps"))
        self._set("FPS tracking", fmt(stats.tracking_fps, "fps"))
        self._set("Inferencia", _p50_p95(stats.inference_ms))
        self._set("Procesado", _p50_p95(stats.processing_ms))
        self._set("Espera en cola", _p50_p95(stats.queue_wait_ms))
        self._set("Captura → resultado", _p50_p95(stats.capture_to_result_ms))
        self._set("Captura → pantalla", _p50_p95(capture_to_paint))
        self._set("Pintado UI", _p50_p95(paint))
        dropped = f"{stats.frames_dropped} de {stats.frames_captured}" if stats.frames_captured else NA
        self._set("Frames descartados", dropped)
        self._set("CPU proceso", fmt(cpu_percent, "% de 1 núcleo", 0))
        self._set("Sin mano izq.", fmt_pct(stats.hand_lost_rate[Handedness.LEFT]))
        self._set("Sin mano der.", fmt_pct(stats.hand_lost_rate[Handedness.RIGHT]))

    def update_snapshot(self, snapshot: TrackingSnapshot | None) -> None:
        if snapshot is None:
            return
        state = snapshot.state
        w, h = snapshot.frame.width, snapshot.frame.height
        self._set("Manos", str(state.hands_tracked))
        self._set("Score izq.", fmt(state.hands[Handedness.LEFT].score, digits=2))
        self._set("Score der.", fmt(state.hands[Handedness.RIGHT].score, digits=2))
        self._set("Outliers rechazados", str(sum(hs.rejected_outliers for hs in state.hands.values())))

        for row, finger in enumerate(PLAYING_FINGERS):
            fs = state.fingers[finger]
            tip, vel = fs.tip, fs.velocity
            speed_px = None if vel is None else float(((vel[0] * w) ** 2 + (vel[1] * h) ** 2) ** 0.5)
            cells = (
                fs.status.value,
                fmt(None if tip is None else float(tip[0]) * w, digits=0),
                fmt(None if tip is None else float(tip[1]) * h, digits=0),
                fmt(None if tip is None else float(tip[2]), digits=3),
                fmt(speed_px, digits=0),
                fmt(None if fs.jitter is None else fs.jitter * w, digits=1),
            )
            for col, text in enumerate(cells, start=1):
                self._table.setItem(row, col, QTableWidgetItem(text))
