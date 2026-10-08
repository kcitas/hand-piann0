"""End-to-end pipeline with a fake camera and a scripted detector (no webcam, no MediaPipe)."""

import threading
import time

import pytest

from handpiano.app.pipeline import PipelineState, TrackingPipeline
from handpiano.camera.camera_config import CameraSettings
from handpiano.camera.camera_manager import CameraManager
from handpiano.tracking.finger_tracker import FingerTracker, TrackStatus
from handpiano.tracking.hand_tracker import ModelLoadError
from handpiano.tracking.landmarks import PLAYING_FINGERS, Handedness
from handpiano.metrics.performance import CpuAccounting, CpuComponent
from tests.fakes import FakeCapture, make_hand, shifted


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


def _pipeline(capture, detector_factory, open_delay_s=0.0, cpu=None, mirror=True):
    def factory(index, backend):
        time.sleep(open_delay_s)  # a slow driver open
        return capture

    camera = CameraManager(CameraSettings(), factory=factory)
    return TrackingPipeline(camera, detector_factory, FingerTracker(), cpu=cpu, mirror=mirror)


def _pipeline_threads():
    return [t for t in threading.enumerate() if t.name in ("tracking", "camera-capture") and t.is_alive()]


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
        ts = snap.timestamps
        # Internal stages are strictly ordered and kept separate.
        assert ts.frame_capture <= ts.frame_published <= ts.tracking_start <= ts.tracking_end <= ts.finger_state_ready
        assert ts.inference_ms >= 1.0  # the scripted 2 ms sleep is measured
        assert ts.capture_to_result_ms == pytest.approx(
            ts.conversion_ms + ts.queue_wait_ms + ts.inference_ms + ts.processing_ms
        )
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


def test_stats_are_unknown_before_any_frame():
    pipeline = _pipeline(FakeCapture(), lambda: ScriptedDetector([]))
    stats = pipeline.stats()
    assert stats.camera_fps is None and stats.tracking_fps is None
    assert stats.inference_ms.p50 is None
    assert stats.hand_lost_rate == {Handedness.LEFT: None, Handedness.RIGHT: None}
    assert stats.spatial_order_ok_rate is None


def test_stop_leaves_no_threads_and_releases_camera():
    capture = FakeCapture(delay_s=0.01)
    pipeline = _pipeline(capture, lambda: ScriptedDetector([]))
    pipeline.start()
    assert _wait_for(lambda: pipeline.state == PipelineState.RUNNING)
    assert pipeline.stop() is True
    assert pipeline.threads_alive() == []
    assert capture.released
    assert pipeline.state == PipelineState.STOPPED


def test_repeated_restarts_do_not_leak_threads():
    baseline = len(_pipeline_threads())
    for _ in range(5):
        pipeline = _pipeline(FakeCapture(delay_s=0.005), lambda: ScriptedDetector([make_hand(Handedness.RIGHT)]))
        pipeline.start()
        assert _wait_for(lambda: pipeline.latest_snapshot() is not None)
        assert pipeline.stop()
    assert len(_pipeline_threads()) == baseline


def test_stop_while_camera_is_still_opening_releases_it():
    """Regression: stop() during a slow open used to leave the camera open."""
    capture = FakeCapture(delay_s=0.005)
    detector = ScriptedDetector([])
    pipeline = _pipeline(capture, lambda: detector, open_delay_s=0.3)
    pipeline.start()
    time.sleep(0.05)  # the tracking thread is inside camera.open()
    pipeline.stop(timeout=2.0)
    assert _wait_for(lambda: capture.released, timeout=2.0)
    assert pipeline.threads_alive() == []
    assert detector.closed


def test_latest_snapshot_is_always_the_newest():
    pipeline = _pipeline(FakeCapture(delay_s=0.002), lambda: ScriptedDetector([make_hand(Handedness.RIGHT)]))
    pipeline.start()
    try:
        seen = []
        deadline = time.perf_counter() + 0.5
        while time.perf_counter() < deadline:
            snap = pipeline.latest_snapshot()
            if snap is not None:
                seen.append((snap.seq, snap.frame.index))
            time.sleep(0.003)
        assert len(seen) > 10
        seqs = [s for s, _ in seen]
        frames = [f for _, f in seen]
        assert seqs == sorted(seqs) and frames == sorted(frames)
    finally:
        pipeline.stop()


def test_cpu_is_accounted_per_component():
    cpu = CpuAccounting()
    pipeline = _pipeline(
        FakeCapture(width=640, height=480, delay_s=0.005), lambda: ScriptedDetector([make_hand(Handedness.RIGHT)]), cpu=cpu
    )
    pipeline.start()
    try:
        assert _wait_for(lambda: (s := pipeline.latest_snapshot()) is not None and s.seq >= 20)
    finally:
        pipeline.stop()
    totals = cpu.totals()
    assert totals.get(CpuComponent.CAPTURE, 0) > 0  # mirror + BGR→RGB of 640×480 frames
    assert totals.get(CpuComponent.TRACKER, 0) > 0


def test_left_right_spatial_order_and_label_overrides_are_counted():
    left, right = make_hand(Handedness.LEFT, (0.3, 0.5)), make_hand(Handedness.RIGHT, (0.7, 0.5))
    pipeline = _pipeline(FakeCapture(delay_s=0.005), lambda: ScriptedDetector([left, right]))
    pipeline.start()
    try:
        assert _wait_for(lambda: pipeline.stats().two_hand_frames >= 5)
        stats = pipeline.stats()
        assert stats.spatial_order_ok_rate == 1.0
        assert stats.label_overrides == 0
    finally:
        pipeline.stop()

    # One right hand whose label flickers to LEFT on every frame after the first.
    class Flicker(ScriptedDetector):
        def detect(self, rgb, timestamp_ms):
            self.n = getattr(self, "n", 0) + 1
            return [right if self.n == 1 else shifted(right, label=Handedness.LEFT)]

    pipeline = _pipeline(FakeCapture(delay_s=0.005), lambda: Flicker([]))
    pipeline.start()
    try:
        assert _wait_for(lambda: pipeline.stats().label_overrides >= 3)
    finally:
        pipeline.stop()


def test_spatial_order_depends_on_mirror():
    """Unmirrored view: the user's LEFT hand should be on the image's right."""
    left, right = make_hand(Handedness.LEFT, (0.3, 0.5)), make_hand(Handedness.RIGHT, (0.7, 0.5))
    pipeline = _pipeline(FakeCapture(delay_s=0.005), lambda: ScriptedDetector([left, right]), mirror=False)
    pipeline.start()
    try:
        assert _wait_for(lambda: pipeline.stats().two_hand_frames >= 5)
        assert pipeline.stats().spatial_order_ok_rate == 0.0
    finally:
        pipeline.stop()
