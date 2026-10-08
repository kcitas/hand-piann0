"""Camera → detector → finger tracker, running off the UI thread.

Threads:
* ``camera-capture``: reads the camera at its own pace into a LatestFrameSlot.
* ``tracking``: loads the detector, opens the camera, starts the capture
  thread, then repeatedly takes the newest frame, runs MediaPipe and the
  finger tracker, and publishes an immutable snapshot. It owns the camera and
  is the only thread that releases it.

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
from handpiano.metrics.performance import CpuAccounting, CpuComponent, RateCounter
from handpiano.metrics.tracking_metrics import RollingFraction
from handpiano.tracking.finger_tracker import FingerTracker, TrackingState, TrackStatus
from handpiano.tracking.hand_tracker import ModelLoadError
from handpiano.tracking.landmarks import Handedness, HandObservation, Landmark

log = logging.getLogger(__name__)

_NS_PER_MS = 1_000_000


class Detector(Protocol):
    def detect(self, rgb, timestamp_ms: int) -> list[HandObservation]: ...
    def close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class FrameTimestamps:
    """``time.perf_counter_ns()`` at each internal stage of one frame.

    The physical stages before ``frame_capture`` (sensor exposure, driver
    buffering) and after the UI paint (compositor, display) are not visible here.
    """

    frame_capture: int  # driver returned the frame to the capture thread
    frame_published: int  # mirrored + converted to RGB, put in the slot
    tracking_start: int  # tracking thread took it, about to call MediaPipe
    tracking_end: int  # MediaPipe returned
    finger_state_ready: int  # identity + outliers + smoothing + finger state done

    @property
    def conversion_ms(self) -> float:
        return (self.frame_published - self.frame_capture) / _NS_PER_MS

    @property
    def queue_wait_ms(self) -> float:
        return (self.tracking_start - self.frame_published) / _NS_PER_MS

    @property
    def inference_ms(self) -> float:
        return (self.tracking_end - self.tracking_start) / _NS_PER_MS

    @property
    def processing_ms(self) -> float:
        return (self.finger_state_ready - self.tracking_end) / _NS_PER_MS

    @property
    def capture_to_result_ms(self) -> float:
        """Frame delivered by the driver → tracker result ready. NOT an end-to-end latency."""
        return (self.finger_state_ready - self.frame_capture) / _NS_PER_MS


@dataclass(frozen=True, slots=True)
class TrackingSnapshot:
    seq: int
    frame: Frame
    state: TrackingState
    timestamps: FrameTimestamps


@dataclass(frozen=True, slots=True)
class PipelineStats:
    camera_fps: float | None
    tracking_fps: float | None
    conversion_ms: StatsSummary
    queue_wait_ms: StatsSummary
    inference_ms: StatsSummary
    processing_ms: StatsSummary
    capture_to_result_ms: StatsSummary
    frames_captured: int
    frames_dropped: int
    # Fraction of recent processed frames in which this hand was NOT tracked.
    hand_lost_rate: dict[Handedness, float | None]
    # Of recent frames with both hands, fraction where LEFT and RIGHT are on the side
    # where the user's left/right should appear (image left in mirrored view, image
    # right otherwise). A diagnostic, not proof: crossed hands legitimately fail it.
    spatial_order_ok_rate: float | None
    two_hand_frames: int
    # Frames where temporal identity kept a label different from MediaPipe's.
    label_overrides: int


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
        cpu: CpuAccounting | None = None,
    ) -> None:
        self._camera = camera
        self._detector_factory = detector_factory
        self._finger_tracker = finger_tracker
        self._mirror = mirror
        self._cpu = cpu
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
        self._conversion = RollingStats()
        self._queue_wait = RollingStats()
        self._inference = RollingStats()
        self._processing = RollingStats()
        self._capture_to_result = RollingStats()
        self._presence = {h: RollingFraction() for h in Handedness}
        self._spatial_order_ok = RollingFraction()
        self._label_overrides = 0

    # -- lifecycle -------------------------------------------------------------

    def start(self) -> None:
        """Non-blocking. Watch ``state`` / ``error_message`` for the outcome."""
        if self._thread is not None:
            raise RuntimeError("pipeline already started")
        self.state = PipelineState.STARTING
        self._thread = threading.Thread(target=self._run, name="tracking", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 3.0) -> bool:
        """Stops both threads. Returns False if a thread did not finish within ``timeout``.

        The tracking thread releases the camera itself, after the capture thread
        has ended, so the camera is never released while still being read.
        """
        self._stop_event.set()
        capture = self._capture
        if capture is not None:
            capture.stop()
        self._slot.close()
        deadline = time.monotonic() + timeout
        for thread in (self._thread, capture):
            if thread is not None:
                thread.join(max(0.0, deadline - time.monotonic()))
        if self._thread is None:
            self._camera.release()
        if self.state != PipelineState.FAILED:
            self.state = PipelineState.STOPPED
        alive = self.threads_alive()
        if alive:
            log.warning("Pipeline threads still running after stop(): %s", alive)
        return not alive

    def threads_alive(self) -> list[str]:
        return [t.name for t in (self._thread, self._capture) if t is not None and t.is_alive()]

    # -- reading (any thread) --------------------------------------------------

    def latest_snapshot(self) -> TrackingSnapshot | None:
        with self._snapshot_lock:
            return self._snapshot

    def stats(self) -> PipelineStats:
        capture = self._capture
        return PipelineStats(
            camera_fps=capture.camera_fps.rate() if capture else None,
            tracking_fps=self._tracking_fps.rate(),
            conversion_ms=self._conversion.summary(),
            queue_wait_ms=self._queue_wait.summary(),
            inference_ms=self._inference.summary(),
            processing_ms=self._processing.summary(),
            capture_to_result_ms=self._capture_to_result.summary(),
            frames_captured=self._slot.produced,
            frames_dropped=self._slot.dropped,
            hand_lost_rate={
                h: None if (f := p.fraction()) is None else 1.0 - f for h, p in self._presence.items()
            },
            spatial_order_ok_rate=self._spatial_order_ok.fraction(),
            two_hand_frames=self._spatial_order_ok.count(),
            label_overrides=self._label_overrides,
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
            self._capture = CaptureThread(self._camera, self._slot, mirror=self._mirror, cpu=self._cpu)
            self._capture.start()
            self.state = PipelineState.RUNNING
            self._loop(detector)
        except Exception:
            log.exception("Tracking pipeline crashed")
            self._fail("El seguimiento de manos se detuvo por un error inesperado. Reinicia la cámara.")
        finally:
            capture = self._capture
            if capture is not None:
                capture.stop()
                capture.join()
            self._camera.release()
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

            cpu_0 = time.thread_time()
            tracking_start = time.perf_counter_ns()
            observations = detector.detect(frame.image, frame.captured_at_ns // _NS_PER_MS)
            tracking_end = time.perf_counter_ns()
            cpu_1 = time.thread_time()
            state = self._finger_tracker.update(observations, frame.captured_at_ns / 1e9)
            finger_state_ready = time.perf_counter_ns()
            if self._cpu is not None:
                self._cpu.add(CpuComponent.INFERENCE, cpu_1 - cpu_0)
                self._cpu.add(CpuComponent.TRACKER, time.thread_time() - cpu_1)

            timestamps = FrameTimestamps(
                frame_capture=frame.captured_at_ns,
                frame_published=frame.published_at_ns,
                tracking_start=tracking_start,
                tracking_end=tracking_end,
                finger_state_ready=finger_state_ready,
            )
            self._record(timestamps, state)
            seq += 1
            with self._snapshot_lock:
                self._snapshot = TrackingSnapshot(seq=seq, frame=frame, state=state, timestamps=timestamps)

    def _record(self, ts: FrameTimestamps, state: TrackingState) -> None:
        self._tracking_fps.tick()
        self._conversion.add(ts.conversion_ms)
        self._queue_wait.add(ts.queue_wait_ms)
        self._inference.add(ts.inference_ms)
        self._processing.add(ts.processing_ms)
        self._capture_to_result.add(ts.capture_to_result_ms)
        hands = state.hands
        for hand, presence in self._presence.items():
            presence.add(hands[hand].status is TrackStatus.TRACKED)
        if all(hands[h].status is TrackStatus.TRACKED for h in Handedness):
            left, right = hands[Handedness.LEFT].landmarks, hands[Handedness.RIGHT].landmarks
            assert left is not None and right is not None
            left_is_image_left = bool(left[Landmark.WRIST, 0] < right[Landmark.WRIST, 0])
            self._spatial_order_ok.add(left_is_image_left == self._mirror)
        self._label_overrides += sum(1 for h in hands.values() if h.label_overridden)
