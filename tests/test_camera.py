import threading
import time

import cv2
import numpy as np
import pytest

from handpiano.camera.camera_config import CameraSettings
from handpiano.camera.camera_manager import CameraError, CameraErrorKind, CameraManager, probe_cameras
from handpiano.camera.capture_thread import CaptureThread
from handpiano.camera.frame import Frame, LatestFrameSlot
from tests.fakes import FakeCapture


def _frame(i: int) -> Frame:
    return Frame(index=i, image=np.zeros((2, 2, 3), np.uint8), captured_at_ns=i, published_at_ns=i)


def test_camera_reports_actual_mode_not_requested():
    fake = FakeCapture(width=640, height=480, fps=30.0)
    cam = CameraManager(CameraSettings(width=1920, height=1080, fps=60), factory=lambda i, b: fake)
    mode = cam.open()
    assert (mode.requested_width, mode.requested_height, mode.requested_fps) == (1920, 1080, 60)
    assert (mode.actual_width, mode.actual_height) == (640, 480)
    assert mode.reported_fps == 30.0


def test_camera_reported_fps_zero_is_unknown():
    fake = FakeCapture(fps=0.0)
    mode = CameraManager(CameraSettings(), factory=lambda i, b: fake).open()
    assert mode.reported_fps is None


def test_exposure_support_reflects_driver_answer():
    accepting = FakeCapture(accepts=frozenset({cv2.CAP_PROP_EXPOSURE}))
    mode = CameraManager(CameraSettings(exposure=-5), factory=lambda i, b: accepting).open()
    assert mode.exposure_supported
    assert not mode.autofocus_supported  # not requested

    refusing = FakeCapture()
    mode = CameraManager(CameraSettings(exposure=-5, autofocus=False), factory=lambda i, b: refusing).open()
    assert not mode.exposure_supported
    assert not mode.autofocus_supported


def test_camera_cannot_open_has_user_message():
    fake = FakeCapture(opened=False)
    with pytest.raises(CameraError) as err:
        CameraManager(CameraSettings(), factory=lambda i, b: fake).open()
    assert err.value.kind is CameraErrorKind.CANNOT_OPEN
    assert "Privacidad" in err.value.user_message
    assert fake.released


def test_camera_without_frames():
    fake = FakeCapture(frames=0)
    with pytest.raises(CameraError) as err:
        CameraManager(CameraSettings(), factory=lambda i, b: fake).open()
    assert err.value.kind is CameraErrorKind.NO_FRAMES


def test_probe_lists_only_working_devices():
    devices = {0: FakeCapture(), 1: FakeCapture(opened=False), 2: FakeCapture(frames=0)}
    found = probe_cameras(max_devices=4, factory=lambda i, b: devices.get(i, FakeCapture(opened=False)))
    assert found == [0]
    assert all(d.released for d in devices.values())


def test_slot_keeps_only_latest_and_counts_drops():
    slot = LatestFrameSlot()
    for i in range(5):
        slot.put(_frame(i))
    taken = slot.take(timeout=0)
    assert taken is not None and taken.index == 4
    assert slot.dropped == 4
    assert slot.take(timeout=0) is None  # no new frame since last take


def test_slot_never_holds_more_than_one_frame_under_fast_producer():
    slot = LatestFrameSlot()
    producer = threading.Thread(target=lambda: [slot.put(_frame(i)) for i in range(5000)])
    producer.start()
    max_pending = 0
    taken = []
    while producer.is_alive():
        max_pending = max(max_pending, slot.pending)
        frame = slot.take(timeout=0.001)
        if frame is not None:
            taken.append(frame.index)
    producer.join()
    last = slot.take(timeout=0)
    if last is not None:
        taken.append(last.index)
    assert max_pending <= 1
    assert taken == sorted(taken)  # never an older frame after a newer one
    assert taken[-1] == 4999  # the newest frame is always delivered
    assert slot.produced == 5000
    assert slot.dropped == 5000 - len(taken)  # every frame is either taken or counted as dropped


def test_slot_take_wakes_on_put():
    slot = LatestFrameSlot()
    result: list[Frame | None] = []
    t = threading.Thread(target=lambda: result.append(slot.take(timeout=2)))
    t.start()
    time.sleep(0.05)
    slot.put(_frame(7))
    t.join(1)
    assert result and result[0] is not None and result[0].index == 7


def test_slot_close_unblocks_consumer():
    slot = LatestFrameSlot()
    t0 = time.perf_counter()
    threading.Timer(0.05, slot.close).start()
    assert slot.take(timeout=2) is None
    assert time.perf_counter() - t0 < 1


def test_capture_thread_mirrors_and_converts_to_rgb():
    fake = FakeCapture(width=8, height=4, frames=3)
    cam = CameraManager(CameraSettings(), factory=lambda i, b: fake)
    cam.open()
    slot = LatestFrameSlot()
    thread = CaptureThread(cam, slot, mirror=True)
    thread.start()
    thread.join(2)
    frame = slot.take(timeout=0)
    assert frame is not None
    assert frame.captured_at_ns <= frame.published_at_ns
    assert not frame.image.flags.writeable  # shared with the UI thread: read-only
    # Fake paints the LEFT half blue in BGR; mirrored + RGB → RIGHT half has blue in channel 2.
    assert frame.image[0, -1, 2] == 255 and frame.image[0, 0, 2] == 0
    assert thread.error is not None  # fake ran out of frames → reported, not crashed
