"""Tracking Lab: raw numbers behind the tracking, for debugging and tuning.

Anything not measured shows "N/A"; metrics of layers that do not exist yet
show "N/A (fase posterior)". A 0 is only shown when 0 was actually measured.
"""

from __future__ import annotations

from PySide6.QtWidgets import QFormLayout, QGroupBox, QLabel, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget

from handpiano.app.pipeline import PipelineStats, TrackingSnapshot
from handpiano.camera.camera_config import CameraMode
from handpiano.instrument.keyboard import KeyboardLayout
from handpiano.interaction.finger_presses import PressFrame
from handpiano.interaction.press_detector import PressEvent
from handpiano.mapping.key_mapper import MappingState
from handpiano.metrics.latency import StatsSummary
from handpiano.metrics.performance import CpuBreakdown, CpuComponent
from handpiano.tracking.finger_tracker import jitter_px
from handpiano.tracking.landmarks import PLAYING_FINGERS, Handedness
from handpiano.ui.format import NA, fmt, fmt_pct

FINGER_COLUMNS = (
    "Dedo", "Estado", "x px", "y px", "z", "vel px/s", "jitter crudo px", "jitter suav. px", "Tecla", "Dist. centro",
    "Pulsación", "Prof.", "Vel. puls.",
)
KEY_COL, DIST_COL, PRESS_COL, DEPTH_COL, PVEL_COL = 8, 9, 10, 11, 12

NOT_YET_AVAILABLE = "N/A (fase posterior)"

HANDEDNESS_HINT = (
    "Validar LEFT/RIGHT: levanta solo tu mano izquierda → debe verse LEFT (cian). "
    "Solo la derecha → RIGHT (violeta). Para una prueba guiada con cuenta atrás: "
    "python -m handpiano --check-hands"
)

CAMERA_KEYS = (
    "Dispositivo", "Backend", "Resolución solicitada", "Resolución entregada", "FPS solicitado",
    "FPS reportado (driver)", "FPS medido", "Exposición", "Autofoco",
)
PIPELINE_KEYS = (
    "FPS tracking", "Conversión (espejo+RGB)", "Espera en cola", "Inferencia", "Procesado tracker", "Mapeo a tecla",
    "Captura → resultado", "Resultado → UI", "UI → pintado", "Captura → pintado", "Pintado UI",
    "Frames descartados",
)
CPU_KEYS = ("CPU proceso", "· captura", "· inferencia (hilo tracking)", "· tracker", "· UI", "· no atribuida")
PIANO_KEYS = (
    "Calibración", "Teclado", "Dedos con posición", "Dentro del piano", "Fuera del piano", "Teclas con varios dedos",
)
TRACKING_KEYS = (
    "Manos (TRACKED)", "Score LEFT", "Score RIGHT", "Sin LEFT", "Sin RIGHT", "Orden espacial coherente",
    "Etiqueta corregida", "Outliers rechazados", "Latencia audio", "Falsos disparos",
)
SOUND_KEYS = ("Pulsaciones → note_on", "Motor de audio", "Liberaciones", "Última pulsación")


def p50_p95(s: StatsSummary) -> str:
    if s.p50 is None or s.p95 is None:
        return NA
    return f"p50 {s.p50:.1f} · p95 {s.p95:.1f} ms"


class DebugView(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.labels: dict[str, QLabel] = {}
        layout = QVBoxLayout(self)
        hint = QLabel(HANDEDNESS_HINT)
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #a1a1b8;")
        layout.addWidget(hint)
        layout.addWidget(self._group("Cámara", CAMERA_KEYS))
        layout.addWidget(self._group("Pipeline (medido internamente)", PIPELINE_KEYS))
        layout.addWidget(self._group("CPU (% de un núcleo)", CPU_KEYS))
        layout.addWidget(self._group("Tracking", TRACKING_KEYS))
        layout.addWidget(self._group("Pulsación → sonido (cada pulsación con su id)", SOUND_KEYS))
        layout.addWidget(self._group("Piano (hover, sin sonido)", PIANO_KEYS))

        self.table = QTableWidget(len(PLAYING_FINGERS), len(FINGER_COLUMNS))
        self.table.setHorizontalHeaderLabels(FINGER_COLUMNS)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setMinimumHeight(300)
        layout.addWidget(self.table, stretch=1)
        self.events_label = QLabel("Eventos de pulsación: N/A")
        self.events_label.setWordWrap(True)
        self.events_label.setStyleSheet("font-family: Menlo, monospace; font-size: 11px;")
        layout.addWidget(self.events_label)
        self.reset()

    def _group(self, title: str, keys: tuple[str, ...]) -> QGroupBox:
        box = QGroupBox(title)
        form = QFormLayout(box)
        for key in keys:
            label = QLabel(NA)
            self.labels[key] = label
            form.addRow(key, label)
        return box

    def _set(self, key: str, text: str) -> None:
        self.labels[key].setText(text)

    def reset(self) -> None:
        """Back to "nothing measured" (used when the camera restarts)."""
        for label in self.labels.values():
            label.setText(NA)
        # Not measurable until the corresponding layers exist.
        self._set("Latencia audio", NOT_YET_AVAILABLE)
        self._set("Falsos disparos", NOT_YET_AVAILABLE)
        for row, finger in enumerate(PLAYING_FINGERS):
            self.table.setItem(row, 0, QTableWidgetItem(finger.short))
            for col in range(1, len(FINGER_COLUMNS)):
                self.table.setItem(row, col, QTableWidgetItem(NA))
        self.table.resizeColumnsToContents()

    def update_sound(self, counts: dict[str, int], last: str | None) -> None:
        """Press → sound chain. "Voz iniciada" = the mixer started the sample; not proof it was heard."""
        self._set("Pulsaciones → note_on", f"{counts['pulsaciones']} → {counts['note_on encoladas']} encoladas"
                  + (f" · {counts['errores']} errores" if counts["errores"] else ""))
        self._set("Motor de audio", f"{counts['voz iniciada']} voces iniciadas · {counts['sin banco']} sin banco")
        self._set("Liberaciones", f"{counts['liberadas']} dedos levantados → {counts['note_off encoladas']} note_off")
        self._set("Última pulsación", last or NA)

    def update_camera(self, mode: CameraMode | None) -> None:
        if mode is None:
            for key in CAMERA_KEYS:
                if key != "FPS medido":
                    self._set(key, NA)
            return
        self._set("Dispositivo", f"Cámara {mode.device_index}")
        self._set("Backend", mode.backend)
        self._set("Resolución solicitada", f"{mode.requested_width}×{mode.requested_height}")
        self._set("Resolución entregada", f"{mode.actual_width}×{mode.actual_height}")
        self._set("FPS solicitado", str(mode.requested_fps))
        self._set("FPS reportado (driver)", fmt(mode.reported_fps, "fps"))
        self._set("Exposición", "control aceptado" if mode.exposure_supported else "no configurable / no solicitado")
        self._set("Autofoco", "control aceptado" if mode.autofocus_supported else "no configurable / no solicitado")

    def update_stats(
        self,
        stats: PipelineStats,
        ui: dict[str, StatsSummary],
        cpu: CpuBreakdown | None,
    ) -> None:
        self._set("FPS medido", fmt(stats.camera_fps, "fps"))
        self._set("FPS tracking", fmt(stats.tracking_fps, "fps"))
        self._set("Conversión (espejo+RGB)", p50_p95(stats.conversion_ms))
        self._set("Espera en cola", p50_p95(stats.queue_wait_ms))
        self._set("Inferencia", p50_p95(stats.inference_ms))
        self._set("Procesado tracker", p50_p95(stats.processing_ms))
        self._set("Mapeo a tecla", p50_p95(stats.mapping_ms))
        self._set("Captura → resultado", p50_p95(stats.capture_to_result_ms))
        self._set("Resultado → UI", p50_p95(ui["result_to_ui"]))
        self._set("UI → pintado", p50_p95(ui["ui_to_paint"]))
        self._set("Captura → pintado", p50_p95(ui["capture_to_paint"]))
        self._set("Pintado UI", p50_p95(ui["paint"]))
        dropped = f"{stats.frames_dropped} de {stats.frames_captured}" if stats.frames_captured else NA
        self._set("Frames descartados", dropped)

        if cpu is not None:
            self._set("CPU proceso", fmt(cpu.process_pct, "%", 0))
            self._set("· captura", fmt(cpu.components_pct.get(CpuComponent.CAPTURE), "%", 0))
            self._set("· inferencia (hilo tracking)", fmt(cpu.components_pct.get(CpuComponent.INFERENCE), "%", 0))
            self._set("· tracker", fmt(cpu.components_pct.get(CpuComponent.TRACKER), "%", 0))
            self._set("· UI", fmt(cpu.components_pct.get(CpuComponent.UI), "%", 0))
            self._set("· no atribuida", fmt(cpu.unattributed_pct, "%", 0))

        self._set("Sin LEFT", fmt_pct(stats.hand_lost_rate[Handedness.LEFT]))
        self._set("Sin RIGHT", fmt_pct(stats.hand_lost_rate[Handedness.RIGHT]))
        order = fmt_pct(stats.spatial_order_ok_rate)
        frames = f" de {stats.two_hand_frames} frames con 2 manos" if stats.two_hand_frames else ""
        self._set("Orden espacial coherente", order + frames)
        self._set("Etiqueta corregida", f"{stats.label_overrides} frames")

    def update_snapshot(self, snapshot: TrackingSnapshot | None) -> None:
        if snapshot is None:
            return
        state = snapshot.state
        w, h = snapshot.frame.width, snapshot.frame.height
        self._set("Manos (TRACKED)", str(state.hands_tracked))
        self._set("Score LEFT", fmt(state.hands[Handedness.LEFT].score, digits=2))
        self._set("Score RIGHT", fmt(state.hands[Handedness.RIGHT].score, digits=2))
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
                fmt(jitter_px(fs.raw_jitter, w, h), digits=2),
                fmt(jitter_px(fs.smoothed_jitter, w, h), digits=2),
            )
            for col, text in enumerate(cells, start=1):
                self.table.setItem(row, col, QTableWidgetItem(text))

    def update_piano(self, layout: KeyboardLayout, status: str, mapping: MappingState | None) -> None:
        self._set("Calibración", status)
        self._set("Teclado", f"{layout.keys[0].name}–{layout.keys[-1].name} · {len(layout.keys)} teclas")
        if mapping is None:
            return
        self._set("Dedos con posición", f"{mapping.with_position}/8")
        inside, outside = mapping.inside, mapping.outside
        self._set("Dentro del piano", NA if inside is None else f"{inside}/8")
        self._set("Fuera del piano", NA if outside is None else f"{outside}/8")
        if not mapping.calibrated:
            self._set("Teclas con varios dedos", NA)
        else:
            shared = [
                f"{self._key_name(layout, midi)}: {', '.join(f.short for f in fingers)}"
                for midi, fingers in mapping.keys_under.items()
                if len(fingers) > 1
            ]
            self._set("Teclas con varios dedos", "; ".join(shared) if shared else "ninguna")
        for row, finger in enumerate(PLAYING_FINGERS):
            fk = mapping.fingers[finger]
            if not mapping.calibrated or fk.piano_xy is None:
                key_text = dist_text = NA
            elif fk.key is None:
                key_text, dist_text = "fuera", NA
            else:
                key_text = fk.key.name
                dist_text = f"{fk.distance_to_center:.2f} teclas"
            self.table.setItem(row, KEY_COL, QTableWidgetItem(key_text))
            self.table.setItem(row, DIST_COL, QTableWidgetItem(dist_text))

    @staticmethod
    def _key_name(layout: KeyboardLayout, midi: int) -> str:
        key = layout.key_by_midi(midi)
        return NA if key is None else key.name

    def update_presses(self, presses: PressFrame | None, recent: list[PressEvent]) -> None:
        """Press state per finger (in hand-size units) and the latest events with their reason."""
        for row, finger in enumerate(PLAYING_FINGERS):
            fp = None if presses is None else presses.fingers[finger]
            cells = (
                NA if fp is None else fp.phase.value,
                NA if fp is None else fmt(fp.depth, digits=2),
                NA if fp is None else fmt(fp.velocity, digits=1),
            )
            for col, text in zip((PRESS_COL, DEPTH_COL, PVEL_COL), cells, strict=True):
                self.table.setItem(row, col, QTableWidgetItem(text))
        if not recent:
            self.events_label.setText("Eventos de pulsación: ninguno todavía")
            return
        lines = [
            f"{e.t % 1000:8.2f}s {e.kind:<8} {e.finger.short:<5} {('-' if e.key is None else e.key.name):<4} {e.reason}"
            for e in reversed(recent)
        ]
        self.events_label.setText("Eventos de pulsación (último primero):\n" + "\n".join(lines))
