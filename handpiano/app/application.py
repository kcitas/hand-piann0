"""Wires the tracking pipeline to the window. Runs on the Qt main thread."""

from __future__ import annotations

import logging
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import replace

from PySide6.QtCore import QObject, QTimer

from handpiano.app.config import AppConfig
from handpiano.app.pipeline import PipelineState, TrackingPipeline
from handpiano.camera.camera_config import CameraSettings
from handpiano.camera.camera_manager import CameraManager, probe_cameras
from handpiano.metrics.performance import ProcessCpuMonitor
from handpiano.tracking.finger_tracker import FingerTracker
from handpiano.tracking.hand_tracker import HandTracker
from handpiano.ui.format import fmt
from handpiano.ui.main_window import MainWindow

log = logging.getLogger(__name__)

SLOW_CAMERA_FPS = 15.0


class Application(QObject):
    def __init__(self, config: AppConfig, window: MainWindow) -> None:
        super().__init__()
        self.config = config
        self.window = window
        self.pipeline: TrackingPipeline | None = None
        self._last_seq = -1
        self._reported_state: str | None = None
        self._cpu = ProcessCpuMonitor()
        self._probe_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="camera-probe")
        self._probe_future: Future[list[int]] | None = None

        window.apply_camera_requested.connect(self.restart_camera)
        window.probe_requested.connect(self.probe_cameras)

        self._frame_timer = QTimer(self)
        self._frame_timer.setInterval(config.ui.poll_interval_ms)
        self._frame_timer.timeout.connect(self._poll_frame)

        self._stats_timer = QTimer(self)
        self._stats_timer.setInterval(config.ui.stats_interval_ms)
        self._stats_timer.timeout.connect(self._poll_stats)

    # -- lifecycle -------------------------------------------------------------

    def start(self) -> None:
        self._start_pipeline(self.config.camera)
        self._frame_timer.start()
        self._stats_timer.start()

    def shutdown(self) -> None:
        self._frame_timer.stop()
        self._stats_timer.stop()
        if self.pipeline is not None:
            self.pipeline.stop()
            self.pipeline = None
        self._probe_executor.shutdown(wait=False, cancel_futures=True)

    def restart_camera(self, settings: CameraSettings) -> None:
        self.config = replace(self.config, camera=settings)
        if self.pipeline is not None:
            self.pipeline.stop()
        self._start_pipeline(settings)

    def _start_pipeline(self, settings: CameraSettings) -> None:
        detector_config = self.config.detector
        self.pipeline = TrackingPipeline(
            camera=CameraManager(settings),
            detector_factory=lambda: HandTracker(detector_config),
            finger_tracker=FingerTracker(self.config.fingers),
            mirror=settings.mirror,
        )
        self._last_seq = -1
        self._reported_state = None
        self.window.camera_view.set_message("Iniciando cámara y detector de manos…")
        self.window.debug_view.update_camera(None)
        self.pipeline.start()

    def probe_cameras(self) -> None:
        if self._probe_future is not None and not self._probe_future.done():
            return
        # The active camera is released first: some drivers refuse a second open.
        if self.pipeline is not None:
            self.pipeline.stop()
            self.pipeline = None
        self.window.camera_view.set_message("Buscando cámaras…")
        self._probe_future = self._probe_executor.submit(probe_cameras)

    # -- polling ---------------------------------------------------------------

    def _poll_frame(self) -> None:
        self._check_probe()
        pipeline = self.pipeline
        if pipeline is None:
            return
        if pipeline.state != self._reported_state:
            self._on_state_change(pipeline)
        snapshot = pipeline.latest_snapshot()
        if snapshot is not None and snapshot.seq != self._last_seq:
            self._last_seq = snapshot.seq
            self.window.camera_view.set_snapshot(snapshot)

    def _on_state_change(self, pipeline: TrackingPipeline) -> None:
        self._reported_state = pipeline.state
        if pipeline.state == PipelineState.FAILED:
            self.window.camera_view.set_message(pipeline.error_message)
            self.window.status_label.setText("Sin seguimiento")
        elif pipeline.state == PipelineState.RUNNING:
            self.window.debug_view.update_camera(pipeline.mode)

    def _check_probe(self) -> None:
        future = self._probe_future
        if future is None or not future.done():
            return
        self._probe_future = None
        try:
            found = future.result()
        except Exception:
            log.exception("Camera probe failed")
            found = []
        if found:
            self.window.set_cameras(found)
        current = self.config.camera
        if found and current.device_index not in found:
            current = replace(current, device_index=found[0])
        self._start_pipeline(current)

    def _poll_stats(self) -> None:
        cpu = self._cpu.sample()
        pipeline = self.pipeline
        if pipeline is None or pipeline.state != PipelineState.RUNNING:
            return
        stats = pipeline.stats()
        view = self.window.camera_view
        snapshot = pipeline.latest_snapshot()

        mode = pipeline.mode
        resolution = f"{mode.actual_width}×{mode.actual_height}" if mode else "N/A"
        warning = ""
        if stats.camera_fps is not None and stats.camera_fps < SLOW_CAMERA_FPS:
            warning = "  ·  ⚠ cámara lenta: mejora la iluminación o baja la resolución"
        hands = snapshot.state.hands_tracked if snapshot else 0
        latency = view.capture_to_paint_ms.summary()
        self.window.status_label.setText(
            f"Cámara {resolution} · {fmt(stats.camera_fps, 'fps')}   ·   "
            f"Tracking {fmt(stats.tracking_fps, 'fps')}   ·   "
            f"Captura→pantalla p50 {fmt(latency.p50, 'ms', 0)}   ·   Manos {hands}{warning}"
        )
        if self.window.lab_checkbox.isChecked():
            self.window.debug_view.update_stats(stats, latency, view.paint_ms.summary(), cpu)
            self.window.debug_view.update_snapshot(snapshot)
