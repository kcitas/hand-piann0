"""Camera → detector → finger tracker, running off the UI thread.

Threads:
* ``camera-capture``: reads the camera at its own pace into a LatestFrameSlot.
* ``tracking``: opens everything, then repeatedly takes the newest frame, runs
  MediaPipe and the finger tracker, and publishes an immutable snapshot.

The UI only ever reads the latest snapshot; nothing queues up, so a slow
detector costs frame rate, never accumulated delay.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from handpiano.camera.camera_config import CameraMode
from handpiano.camera.camera_manager import CameraError, CameraManager
from handpiano.camera.capture_thread import CaptureThread
from handpiano.camera.frame import Frame, LatestFrameSlot
from handpiano.metrics.latency import RollingStats, StatsSummary
from handpiano.metrics.performance import RateCounter
from handpiano.metrics.tracking_metrics import PresenceRate
from handpiano.tracking.finger_tracker import FingerTracker, TrackingState, TrackStatus
from handpiano.tracking.hand_tracker import ModelLoadError
from handpiano.tracking.landmarks import Handedness, HandObservation

log = logging.getLogger(__name__)

_NS_PER_MS = 1_000_000


class Detector(Protocol):
    def detect(self, rgb, timestamp_ms: int) -> list[HandObservation]: ...
    def close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class FrameTimings:
    # Frame waiting in the slot before the tracking thread picked it up.
    queue_wait_ms: float
    inference_ms: float
    # Identity + smoothing + finger tracking.
    processing_ms: float
    # From "driver returned the frame" to "tracking result ready".
    capture_to_result_ms: float


@dataclass(frozen=True, slots=True)
class TrackingSnapshot:
    seq: int
    frame: Frame
    state: TrackingState
    timings: FrameTimings


@dataclass(frozen=True, slots=True)
class PipelineStats:
    camera_fps: float | None
    tracking_fps: float | None
    inference_ms: StatsSummary
    processing_ms: StatsSummary
    queue_wait_ms: StatsSummary
    capture_to_result_ms: StatsSummary
    frames_captured: int
    frames_dropped: int
    hand_lost_rate: dict[Handedness, float | None]


class PipelineState:
    STARTING = "starting"
    RUNNING = "running"
    FAILED = "failed"
    STOPPED = "stopped"


class TrackingPipeline:
    def __init__(
        self,
        camera: CameraManager,
        detector_factory: Callable[[], Detector],
        finger_tracker: FingerTracker,
        mirror: bool = True,
    ) -> None:
        self._camera = camera
        self._detector_factory = detector_factory
        self._finger_tracker = finger_tracker
        self._mirror = mirror
        self._slot = LatestFrameSlot()
        self._capture: CaptureThread | None = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._snapshot_lock = threading.Lock()
        self._snapshot: TrackingSnapshot | None = None

        self.state = PipelineState.STOPPED
        self.mode: CameraMode | None = None
        self.error_message: str | None = None

        self._tracking_fps = RateCounter()
        self._inference = RollingStats()
        self._processing = RollingStats()
        self._queue_wait = RollingStats()
        self._capture_to_result = RollingStats()
        self._presence = {h: PresenceRate() for h in Handedness}

    # -- lifecycle -------------------------------------------------------------

    def start(self) -> None:
        """Non-blocking. Watch ``state`` / ``error_message`` for the outcome."""
        if self._thread is not None:
            raise RuntimeError("pipeline already started")
        self.state = PipelineState.STARTING
        self._thread = threading.Thread(target=self._run, name="tracking", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        self._stop_event.set()
        if self._capture is not None:
            self._capture.stop()
        self._slot.close()
        if self._thread is not None:
            self._thread.join(timeout)
        if self._capture is not None:
            self._capture.join(timeout)
        self._camera.release()
        if self.state != PipelineState.FAILED:
            self.state = PipelineState.STOPPED

    # -- reading (any thread) --------------------------------------------------

    def latest_snapshot(self) -> TrackingSnapshot | None:
        with self._snapshot_lock:
            return self._snapshot

    def stats(self) -> PipelineStats:
        capture = self._capture
        return PipelineStats(
            camera_fps=capture.camera_fps.rate() if capture else None,
            tracking_fps=self._tracking_fps.rate(),
            inference_ms=self._inference.summary(),
            processing_ms=self._processing.summary(),
            queue_wait_ms=self._queue_wait.summary(),
            capture_to_result_ms=self._capture_to_result.summary(),
            frames_captured=self._slot.produced,
            frames_dropped=self._slot.dropped,
            hand_lost_rate={h: p.lost_rate() for h, p in self._presence.items()},
        )

    # -- worker ----------------------------------------------------------------

    def _fail(self, message: str) -> None:
        self.error_message = message
        self.state = PipelineState.FAILED

    def _run(self) -> None:
        try:
            detector = self._detector_factory()
        except ModelLoadError as exc:
            log.error("Detector failed to load: %s", exc)
            self._fail(exc.user_message)
            return
        try:
            try:
                self.mode = self._camera.open()
            except CameraError as exc:
                log.error("Camera failed to open: %s", exc)
                self._fail(exc.user_message)
                return
            if self._stop_event.is_set():
                return
            self._capture = CaptureThread(self._camera, self._slot, mirror=self._mirror)
            self._capture.start()
            self.state = PipelineState.RUNNING
            self._loop(detector)
        except Exception:
            log.exception("Tracking pipeline crashed")
            self._fail("El seguimiento de manos se detuvo por un error inesperado. Reinicia la cámara.")
        finally:
            detector.close()

    def _loop(self, detector: Detector) -> None:
        seq = 0
        while not self._stop_event.is_set():
            frame = self._slot.take(timeout=0.5)
            if frame is None:
                capture = self._capture
                if capture is not None and capture.error is not None:
                    self._fail(capture.error.user_message)
                    return
                continue

            t_start = time.perf_counter_ns()
            observations = detector.detect(frame.image, frame.captured_at_ns // _NS_PER_MS)
            t_inferred = time.perf_counter_ns()
            state = self._finger_tracker.update(observations, frame.captured_at_ns / 1e9)
            t_done = time.perf_counter_ns()

            timings = FrameTimings(
                queue_wait_ms=(t_start - frame.captured_at_ns) / _NS_PER_MS,
                inference_ms=(t_inferred - t_start) / _NS_PER_MS,
                processing_ms=(t_done - t_inferred) / _NS_PER_MS,
                capture_to_result_ms=(t_done - frame.captured_at_ns) / _NS_PER_MS,
            )
            self._record(timings, state)
            seq += 1
            with self._snapshot_lock:
                self._snapshot = TrackingSnapshot(seq=seq, frame=frame, state=state, timings=timings)

    def _record(self, timings: FrameTimings, state: TrackingState) -> None:
        self._tracking_fps.tick()
        self._inference.add(timings.inference_ms)
        self._processing.add(timings.processing_ms)
        self._queue_wait.add(timings.queue_wait_ms)
        self._capture_to_result.add(timings.capture_to_result_ms)
        for hand, presence in self._presence.items():
            presence.add(state.hands[hand].status is TrackStatus.TRACKED)
