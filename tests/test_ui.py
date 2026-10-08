"""Renders the real widgets offscreen with synthetic tracking data."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from handpiano.app.pipeline import FrameTimings, PipelineStats, TrackingSnapshot  # noqa: E402
from handpiano.camera.camera_config import CameraMode, CameraSettings  # noqa: E402
from handpiano.camera.frame import Frame  # noqa: E402
from handpiano.metrics.latency import EMPTY_SUMMARY  # noqa: E402
from handpiano.tracking.finger_tracker import FingerTracker  # noqa: E402
from handpiano.tracking.landmarks import FingerId, Handedness  # noqa: E402
from handpiano.ui.format import fmt, fmt_pct  # noqa: E402
from handpiano.ui.main_window import MainWindow  # noqa: E402
from tests.fakes import make_hand  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _snapshot() -> TrackingSnapshot:
    state = FingerTracker().update([make_hand(Handedness.LEFT, (0.3, 0.5)), make_hand(Handedness.RIGHT, (0.7, 0.5))], 0.0)
    image = np.full((240, 320, 3), 40, np.uint8)
    return TrackingSnapshot(
        seq=1,
        frame=Frame(index=0, image=image, captured_at_ns=0),
        state=state,
        timings=FrameTimings(0.1, 10.0, 0.2, 10.3),
    )


def test_format_never_invents_values():
    assert fmt(None, "ms") == "N/A"
    assert fmt(12.345, "ms") == "12.3 ms"
    assert fmt_pct(None) == "N/A"
    assert fmt_pct(0.25) == "25 %"


def test_camera_view_draws_fingertips_over_frame(qapp):
    window = MainWindow(CameraSettings())
    window.resize(640, 520)
    window.set_lab_mode(True)
    snapshot = _snapshot()
    window.camera_view.set_snapshot(snapshot)
    window.show()
    image = window.camera_view.grab().toImage()

    view = window.camera_view
    scale = min(view.width() / 320, view.height() / 240)
    ox, oy = (view.width() - 320 * scale) / 2, (view.height() - 240 * scale) / 2
    tip = snapshot.state.fingers[FingerId.RIGHT_INDEX].tip
    px = image.pixelColor(int(ox + tip[0] * 320 * scale), int(oy + tip[1] * 240 * scale))
    # Right-hand fingertip marker is violet (#a78bfa), not the grey background.
    assert (px.red(), px.green(), px.blue()) == (0xA7, 0x8B, 0xFA)
    assert view.paint_ms.summary().count >= 1
    window.close()


def test_debug_view_shows_na_for_unmeasured(qapp):
    window = MainWindow(CameraSettings())
    stats = PipelineStats(
        camera_fps=None,
        tracking_fps=29.7,
        inference_ms=EMPTY_SUMMARY,
        processing_ms=EMPTY_SUMMARY,
        queue_wait_ms=EMPTY_SUMMARY,
        capture_to_result_ms=EMPTY_SUMMARY,
        frames_captured=0,
        frames_dropped=0,
        hand_lost_rate={Handedness.LEFT: None, Handedness.RIGHT: 0.1},
    )
    dv = window.debug_view
    dv.update_stats(stats, EMPTY_SUMMARY, EMPTY_SUMMARY, None)
    dv.update_camera(
        CameraMode(0, "AVFoundation", 1920, 1080, 60, 1280, 720, 30.0, False, False)
    )
    dv.update_snapshot(_snapshot())
    labels = dv._labels
    assert labels["FPS medido"].text() == "N/A"
    assert labels["FPS tracking"].text() == "29.7 fps"
    assert labels["Inferencia"].text() == "N/A"
    assert labels["Solicitado"].text() == "1920×1080 @ 60"
    assert labels["Entregado"].text() == "1280×720"
    assert labels["Latencia audio"].text().startswith("N/A")
    assert labels["Manos"].text() == "2"
    assert dv._table.item(0, 1).text() == "TRACKED"
    window.close()


def test_toolbar_reflects_requested_settings_and_applies_them(qapp):
    window = MainWindow(CameraSettings(width=1280, height=720, fps=30))
    assert window.resolution_combo.currentText() == "1280×720"
    applied = []
    window.apply_camera_requested.connect(applied.append)
    window.resolution_combo.setCurrentIndex(window.resolution_combo.findData("640x480"))
    window.fps_combo.setCurrentIndex(window.fps_combo.findData(60))
    window._on_apply()
    assert (applied[0].width, applied[0].height, applied[0].fps) == (640, 480, 60)
    window.close()
