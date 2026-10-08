"""Renders the real widgets offscreen with synthetic tracking data."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from handpiano.app.application import Application  # noqa: E402
from handpiano.app.config import AppConfig  # noqa: E402
from handpiano.app.pipeline import FrameTimestamps, PipelineState, PipelineStats, TrackingSnapshot  # noqa: E402
from handpiano.camera.camera_config import CameraMode, CameraSettings  # noqa: E402
from handpiano.camera.frame import Frame  # noqa: E402
from handpiano.metrics.latency import EMPTY_SUMMARY  # noqa: E402
from handpiano.metrics.performance import CpuBreakdown, CpuComponent  # noqa: E402
from handpiano.tracking.finger_tracker import FingerTracker  # noqa: E402
from handpiano.tracking.landmarks import FingerId, Handedness, Landmark  # noqa: E402
from handpiano.ui.debug_view import NOT_YET_AVAILABLE  # noqa: E402
from handpiano.ui.format import NA, fmt, fmt_pct  # noqa: E402
from handpiano.ui.main_window import MainWindow  # noqa: E402
from tests.fakes import make_hand  # noqa: E402

FRAME_W, FRAME_H = 320, 240


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _snapshot(seq: int = 1) -> TrackingSnapshot:
    state = FingerTracker().update(
        [make_hand(Handedness.LEFT, (0.3, 0.4)), make_hand(Handedness.RIGHT, (0.7, 0.4))], 0.0
    )
    image = np.full((FRAME_H, FRAME_W, 3), 40, np.uint8)
    image.setflags(write=False)
    return TrackingSnapshot(
        seq=seq,
        frame=Frame(index=seq, image=image, captured_at_ns=0, published_at_ns=1),
        state=state,
        timestamps=FrameTimestamps(0, 1, 2, 3, 4),
    )


def _empty_stats(**overrides) -> PipelineStats:
    values = dict(
        camera_fps=None,
        tracking_fps=None,
        conversion_ms=EMPTY_SUMMARY,
        queue_wait_ms=EMPTY_SUMMARY,
        inference_ms=EMPTY_SUMMARY,
        processing_ms=EMPTY_SUMMARY,
        capture_to_result_ms=EMPTY_SUMMARY,
        frames_captured=0,
        frames_dropped=0,
        hand_lost_rate={Handedness.LEFT: None, Handedness.RIGHT: None},
        spatial_order_ok_rate=None,
        two_hand_frames=0,
        label_overrides=0,
    )
    values.update(overrides)
    return PipelineStats(**values)


def _widget_point(view, x, y):
    scale = min(view.width() / FRAME_W, view.height() / FRAME_H)
    ox, oy = (view.width() - FRAME_W * scale) / 2, (view.height() - FRAME_H * scale) / 2
    return int(ox + x * FRAME_W * scale), int(oy + y * FRAME_H * scale)


def test_format_never_invents_values():
    assert fmt(None, "ms") == "N/A"
    assert fmt(12.345, "ms") == "12.3 ms"
    assert fmt(0.0, "ms") == "0.0 ms"  # a measured zero is shown as zero
    assert fmt_pct(None) == "N/A"
    assert fmt_pct(0.25) == "25 %"


def test_camera_view_draws_fingertips_over_frame(qapp):
    window = MainWindow(CameraSettings())
    window.resize(640, 520)
    snapshot = _snapshot()
    window.camera_view.set_snapshot(snapshot)
    window.show()
    image = window.camera_view.grab().toImage()
    tip = snapshot.state.fingers[FingerId.RIGHT_INDEX].tip
    px = image.pixelColor(*_widget_point(window.camera_view, tip[0], tip[1]))
    # Right-hand fingertip marker is violet (#a78bfa), not the grey background.
    assert (px.red(), px.green(), px.blue()) == (0xA7, 0x8B, 0xFA)
    window.close()


def test_lab_mode_draws_left_right_badges_under_each_wrist(qapp):
    window = MainWindow(CameraSettings())
    window.resize(900, 700)
    window.set_lab_mode(True)
    snapshot = _snapshot()
    window.camera_view.set_snapshot(snapshot)
    window.show()
    image = window.camera_view.grab().toImage()
    view = window.camera_view

    def colors_below_wrist(hand):
        wrist = snapshot.state.hands[hand].landmarks[Landmark.WRIST]
        x, y = _widget_point(view, wrist[0], wrist[1])
        found = set()
        for dy in range(10, 90):
            for dx in range(-70, 71, 2):
                c = image.pixelColor(x + dx, y + dy)
                found.add((c.red(), c.green(), c.blue()))
        return found

    # Badge border/title use the hand colour: cyan for LEFT, violet for RIGHT.
    assert (0x22, 0xD3, 0xEE) in colors_below_wrist(Handedness.LEFT)
    assert (0xA7, 0x8B, 0xFA) in colors_below_wrist(Handedness.RIGHT)
    window.close()


def test_capture_to_paint_is_recorded_once_per_snapshot(qapp):
    window = MainWindow(CameraSettings())
    window.show()
    view = window.camera_view
    view.set_snapshot(_snapshot(seq=1))
    view.grab()
    view.grab()  # e.g. a resize: same snapshot painted again
    view.grab()
    assert view.capture_to_paint_ms.summary().count == 1
    assert view.result_to_ui_ms.summary().count == 1
    assert view.paint_ms.summary().count >= 3  # paint duration counts every paint
    view.set_snapshot(_snapshot(seq=2))
    view.grab()
    assert view.capture_to_paint_ms.summary().count == 2
    window.close()


def test_debug_view_shows_na_for_unmeasured(qapp):
    window = MainWindow(CameraSettings())
    dv = window.debug_view
    labels = dv.labels
    assert all(label.text().startswith(NA) for label in labels.values())
    assert labels["Latencia audio"].text() == NOT_YET_AVAILABLE
    assert labels["Falsos disparos"].text() == NOT_YET_AVAILABLE

    ui = {k: EMPTY_SUMMARY for k in ("result_to_ui", "ui_to_paint", "capture_to_paint", "paint")}
    dv.update_stats(_empty_stats(tracking_fps=29.7), ui, None)
    assert labels["FPS medido"].text() == NA
    assert labels["FPS tracking"].text() == "29.7 fps"
    assert labels["Inferencia"].text() == NA
    assert labels["Frames descartados"].text() == NA
    assert labels["Sin LEFT"].text() == NA
    assert labels["CPU proceso"].text() == NA
    assert labels["Orden espacial coherente"].text() == NA

    dv.update_camera(CameraMode(0, "AVFoundation", 1920, 1080, 60, 1280, 720, 30.0, False, False))
    assert labels["Resolución solicitada"].text() == "1920×1080"
    assert labels["Resolución entregada"].text() == "1280×720"
    assert labels["FPS solicitado"].text() == "60"
    assert labels["FPS reportado (driver)"].text() == "30.0 fps"

    dv.update_snapshot(_snapshot())
    assert labels["Manos (TRACKED)"].text() == "2"
    assert dv.table.item(0, 1).text() == "TRACKED"
    assert dv.table.item(0, 6).text() == NA  # jitter needs a full window

    dv.reset()
    assert labels["Manos (TRACKED)"].text() == NA
    assert labels["Resolución entregada"].text() == NA
    window.close()


def test_debug_view_cpu_breakdown(qapp):
    window = MainWindow(CameraSettings())
    ui = {k: EMPTY_SUMMARY for k in ("result_to_ui", "ui_to_paint", "capture_to_paint", "paint")}
    breakdown = CpuBreakdown(process_pct=120.0, components_pct={CpuComponent.INFERENCE: 30.0, CpuComponent.UI: 10.0})
    window.debug_view.update_stats(_empty_stats(), ui, breakdown)
    labels = window.debug_view.labels
    assert labels["CPU proceso"].text() == "120 %"
    assert labels["· inferencia (hilo tracking)"].text() == "30 %"
    assert labels["· captura"].text() == NA  # no CPU recorded for this component
    assert labels["· no atribuida"].text() == "80 %"
    window.close()


class _FakePipeline:
    def __init__(self):
        self.state = PipelineState.RUNNING
        self.mode = None
        self.error_message = None
        self.snapshot = None

    def latest_snapshot(self):
        return self.snapshot

    def stop(self):
        return True


def test_ui_always_shows_latest_snapshot(qapp):
    window = MainWindow(CameraSettings())
    app = Application(AppConfig(), window)
    fake = _FakePipeline()
    app.pipeline = fake
    for seq in (3, 4, 9):  # snapshots 5–8 were produced and replaced between two polls
        fake.snapshot = _snapshot(seq)
        app._poll_frame()
        assert window.camera_view._snapshot.seq == seq
    app.pipeline = None
    app.shutdown()
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


def test_banner_is_drawn_over_the_video(qapp):
    window = MainWindow(CameraSettings())
    window.resize(800, 600)
    view = window.camera_view
    view.set_snapshot(_snapshot())
    view.set_banner("Levanta SOLO tu mano IZQUIERDA", ["7 s"], 0.3, "#fbbf24")
    window.show()
    image = view.grab().toImage()
    def near_accent(c):
        return abs(c.red() - 0xFB) < 12 and abs(c.green() - 0xBF) < 12 and abs(c.blue() - 0x24) < 12

    # Banner border (top edge at y≈16) in the accent colour.
    assert any(near_accent(image.pixelColor(x, y)) for x in range(40, view.width() - 40, 5) for y in range(13, 21))
    view.set_banner(None)
    assert view._banner is None
    window.close()


def test_handedness_check_drives_the_banner(qapp):
    window = MainWindow(CameraSettings())
    app = Application(AppConfig(), window)
    fake = _FakePipeline()
    app.pipeline = fake
    app.request_handedness_check()
    fake.snapshot = _snapshot(1)
    app._poll_frame()
    assert app.handedness_check is not None
    title = window.camera_view._banner[0]
    assert title.startswith("Prepárate")
    app.pipeline = None
    app.shutdown()
    window.close()


def test_unchanged_banner_does_not_schedule_repaints(qapp, monkeypatch):
    window = MainWindow(CameraSettings())
    view = window.camera_view
    calls = []
    monkeypatch.setattr(view, "update", lambda: calls.append(1))
    view.set_banner("Paso", ["7 s"], 0.301)
    view.set_banner("Paso", ["7 s"], 0.304)  # same text, progress rounds to the same 1 %
    view.set_banner("Paso", ["7 s"], 0.304)
    assert len(calls) == 1
    view.set_banner("Paso", ["6 s"], 0.40)
    assert len(calls) == 2
    window.close()
