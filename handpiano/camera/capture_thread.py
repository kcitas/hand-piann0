"""Background thread that reads the camera as fast as it delivers frames."""

from __future__ import annotations

import logging
import threading
import time

import cv2

from handpiano.camera.camera_manager import CameraError, CameraErrorKind, CameraManager
from handpiano.camera.frame import Frame, LatestFrameSlot
from handpiano.metrics.performance import RateCounter

log = logging.getLogger(__name__)


class CaptureThread(threading.Thread):
    """Reads frames, converts BGR→RGB (+mirror) and publishes them to a LatestFrameSlot.

    Conversion happens here so it overlaps with inference on the tracking thread.
    """

    MAX_CONSECUTIVE_FAILURES = 30

    def __init__(self, camera: CameraManager, slot: LatestFrameSlot, mirror: bool) -> None:
        super().__init__(name="camera-capture", daemon=True)
        self._camera = camera
        self._slot = slot
        self._mirror = mirror
        self._stop_event = threading.Event()
        self.camera_fps = RateCounter()
        self.error: CameraError | None = None

    def run(self) -> None:
        index = 0
        failures = 0
        while not self._stop_event.is_set():
            bgr = self._camera.read()
            captured_at_ns = time.perf_counter_ns()
            if bgr is None:
                failures += 1
                if failures >= self.MAX_CONSECUTIVE_FAILURES:
                    self.error = CameraError(CameraErrorKind.NO_FRAMES, "read failed repeatedly")
                    log.error("Camera stopped delivering frames")
                    break
                continue
            failures = 0
            self.camera_fps.tick()
            if self._mirror:
                bgr = cv2.flip(bgr, 1)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            self._slot.put(Frame(index=index, image=rgb, captured_at_ns=captured_at_ns))
            index += 1
        self._slot.close()

    def stop(self) -> None:
        self._stop_event.set()
