"""End-to-end pipeline with a fake camera and a scripted detector (no webcam, no MediaPipe)."""

import time

from handpiano.app.pipeline import PipelineState, TrackingPipeline
from handpiano.camera.camera_config import CameraSettings
from handpiano.camera.camera_manager import CameraManager
from handpiano.tracking.finger_tracker import FingerTracker, TrackStatus
from handpiano.tracking.hand_tracker import ModelLoadError
from handpiano.tracking.landmarks import PLAYING_FINGERS, Handedness
from tests.fakes import FakeCapture, make_hand


class ScriptedDetector:
    def __init__(self, hands):
        self.hands = hands
        self.closed = False
        self.timestamps: list[int] = []

    def detect(self, rgb, timestamp_ms):
        self.timestamps.append(timestamp_ms)
        time.sleep(0.002)
        return list(self.hands)

    def close(self):
        self.closed = True


def _wait_for(predicate, timeout=3.0):
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def _pipeline(capture, detector_factory):
    camera = CameraManager(CameraSettings(), factory=lambda i, b: capture)
    return TrackingPipeline(camera, detector_factory, FingerTracker())


def test_pipeline_tracks_eight_fingers_from_fake_camera():
    detector = ScriptedDetector([make_hand(Handedness.LEFT, (0.3, 0.5)), make_hand(Handedness.RIGHT, (0.7, 0.5))])
    pipeline = _pipeline(FakeCapture(width=320, height=240, delay_s=0.01), lambda: detector)
    pipeline.start()
    try:
        assert _wait_for(lambda: (s := pipeline.latest_snapshot()) is not None and s.seq >= 5)
        snap = pipeline.latest_snapshot()
        assert pipeline.state == PipelineState.RUNNING
        assert pipeline.mode is not None and pipeline.mode.actual_width == 320
        assert snap.frame.image.shape == (240, 320, 3)
        assert all(snap.state.fingers[f].status is TrackStatus.TRACKED for f in PLAYING_FINGERS)
        assert snap.timings.inference_ms >= 1.0  # the scripted 2 ms sleep is measured
        stats = pipeline.stats()
        assert stats.inference_ms.count >= 5
        assert stats.hand_lost_rate[Handedness.LEFT] == 0.0
        assert detector.timestamps == sorted(detector.timestamps)
    finally:
        pipeline.stop()
    assert detector.closed


def test_missing_model_fails_with_user_message():
    def factory():
        raise ModelLoadError("missing", missing=True)

    pipeline = _pipeline(FakeCapture(), factory)
    pipeline.start()
    assert _wait_for(lambda: pipeline.state == PipelineState.FAILED)
    assert "download_model" in pipeline.error_message
    pipeline.stop()


def test_camera_denied_fails_with_user_message():
    detector = ScriptedDetector([])
    pipeline = _pipeline(FakeCapture(opened=False), lambda: detector)
    pipeline.start()
    assert _wait_for(lambda: pipeline.state == PipelineState.FAILED)
    assert "Cámara" in pipeline.error_message
    pipeline.stop()
    assert detector.closed


def test_camera_disconnect_is_reported():
    detector = ScriptedDetector([])
    pipeline = _pipeline(FakeCapture(frames=20, delay_s=0.001), lambda: detector)
    pipeline.start()
    assert _wait_for(lambda: pipeline.state == PipelineState.FAILED, timeout=5)
    assert "no está entregando" in pipeline.error_message
    pipeline.stop()
