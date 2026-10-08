"""Wires the tracking pipeline to the window. Runs on the Qt main thread."""

from __future__ import annotations

import logging
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import replace

from PySide6.QtCore import QObject, QTimer

from handpiano.app.config import AppConfig
from handpiano.app.handedness_check import HandednessCheck
from handpiano.app.pipeline import PipelineState, TrackingPipeline
from handpiano.camera.camera_config import CameraSettings
from handpiano.camera.camera_manager import CameraManager, probe_cameras
from handpiano.metrics.performance import CpuAccounting, CpuComponent, CpuSampler
from handpiano.tracking.finger_tracker import FingerTracker
from handpiano.tracking.hand_tracker import HandTracker
from handpiano.ui.format import NA, fmt
from handpiano.ui.main_window import MainWindow

log = logging.getLogger(__name__)

SLOW_CAMERA_FPS = 15.0


class Application(QObject):
    def __init__(self, config: AppConfig, window: MainWindow, cpu: CpuAccounting | None = None) -> None:
        super().__init__()
        self.config = config
        self.window = window
        self.pipeline: TrackingPipeline | None = None
        self._last_seq = -1
        self._reported_state: str | None = None
        self.cpu = cpu if cpu is not None else CpuAccounting()
        self._cpu_sampler = CpuSampler(self.cpu)
        self._probe_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="camera-probe")
        self._probe_future: Future[list[int]] | None = None
        self.handedness_check: HandednessCheck | None = None
        self._check_requested = False

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

    def request_handedness_check(self) -> None:
        """Starts the guided LEFT/RIGHT check as soon as the camera is running."""
        self._check_requested = True
        self.handedness_check = None
        self.window.camera_view.set_banner("Validación LEFT/RIGHT", ["Esperando a la cámara…"])

    def restart_camera(self, settings: CameraSettings) -> None:
        self.config = replace(self.config, camera=settings)
        if self.pipeline is not None:
            self.pipeline.stop()
        self._start_pipeline(settings)

    def _start_pipeline(self, settings: CameraSettings) -> None:
        detector_config = self.config.detector
        fingers = self.config.fingers
        fingers = replace(fingers, identity=replace(fingers.identity, mirrored_view=settings.mirror))
        self.pipeline = TrackingPipeline(
            camera=CameraManager(settings),
            detector_factory=lambda: HandTracker(detector_config, mirrored_input=settings.mirror),
            finger_tracker=FingerTracker(fingers),
            mirror=settings.mirror,
            cpu=self.cpu,
        )
        self._last_seq = -1
        self._reported_state = None
        self.window.camera_view.set_message("Iniciando cámara y detector de manos…")
        self.window.camera_view.reset_metrics()
        self.window.debug_view.reset()
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
        cpu_start = time.thread_time()
        try:
            self._poll_frame_inner()
        finally:
            self.cpu.add(CpuComponent.UI, time.thread_time() - cpu_start)

    def _poll_frame_inner(self) -> None:
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
            if self.handedness_check is not None:
                self.handedness_check.update(time.monotonic(), snapshot.state)
        self._update_handedness_check(pipeline)

    def _update_handedness_check(self, pipeline: TrackingPipeline) -> None:
        now = time.monotonic()
        if self._check_requested and pipeline.state == PipelineState.RUNNING:
            self._check_requested = False
            self.handedness_check = HandednessCheck(mirrored=self.config.camera.mirror)
            self.handedness_check.start(now)
        check = self.handedness_check
        if check is None:
            return
        view = self.window.camera_view
        spec, remaining, elapsed = check.current(now)
        if spec is None:
            if check.started_at is not None:
                passed = check.passed()
                title = "Validación LEFT/RIGHT: OK" if passed else "Validación LEFT/RIGHT: NO SUPERADA"
                lines = check.summary_lines()
                view.set_banner(title, lines, None, "#34d399" if passed else "#f87171")
                log.info("Handedness check finished (%s): %s", "OK" if passed else "FAILED", " | ".join(lines))
                check.started_at = None  # report once; the banner stays
            return
        detail = [f"{remaining:.0f} s"]
        if spec.step in check.results:
            r = check.results[spec.step]
            rate = "N/A" if r.rate is None else f"{r.rate:.0%}"
            detail.append(f"Coincidencia hasta ahora: {rate} ({r.evaluated} frames)")
        view.set_banner(spec.instruction, detail, elapsed / spec.duration_s)

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
        cpu_start = time.thread_time()
        try:
            self._poll_stats_inner()
        finally:
            self.cpu.add(CpuComponent.UI, time.thread_time() - cpu_start)

    def _poll_stats_inner(self) -> None:
        cpu = self._cpu_sampler.sample()
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
        hands = str(snapshot.state.hands_tracked) if snapshot else NA
        latency = view.capture_to_paint_ms.summary()
        self.window.status_label.setText(
            f"Cámara {resolution} · {fmt(stats.camera_fps, 'fps')}   ·   "
            f"Tracking {fmt(stats.tracking_fps, 'fps')}   ·   "
            f"Captura→pintado p50 {fmt(latency.p50, 'ms', 0)}   ·   Manos {hands}{warning}"
        )
        if self.window.lab_checkbox.isChecked():
            ui = {
                "result_to_ui": view.result_to_ui_ms.summary(),
                "ui_to_paint": view.ui_to_paint_ms.summary(),
                "capture_to_paint": latency,
                "paint": view.paint_ms.summary(),
            }
            self.window.debug_view.update_stats(stats, ui, cpu)
            self.window.debug_view.update_snapshot(snapshot)
