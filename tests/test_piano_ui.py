"""Phase G UI: keyboard widget, projected overlay, calibration flow. Offscreen, no camera."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from dataclasses import replace  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from handpiano.app.application import Application  # noqa: E402
from handpiano.app.config import AppConfig  # noqa: E402
from handpiano.app.pipeline import FrameTimestamps, PipelineState, TrackingSnapshot  # noqa: E402
from handpiano.camera.camera_config import CameraMode, CameraSettings  # noqa: E402
from handpiano.camera.frame import Frame  # noqa: E402
from handpiano.instrument.keyboard import KeyboardLayout  # noqa: E402
from handpiano.mapping.calibration import CameraKey, PianoCalibration  # noqa: E402
from handpiano.mapping.key_mapper import KeyMapper  # noqa: E402
from handpiano.tracking.landmarks import FingerId  # noqa: E402
from handpiano.ui.format import NA  # noqa: E402
from handpiano.ui.main_window import MainWindow  # noqa: E402
from tests.test_mapping import _state_with_tips  # noqa: E402

LAYOUT = KeyboardLayout.from_octaves("C4", 2)
RECT = [(0.2, 0.5), (0.8, 0.5), (0.8, 0.9), (0.2, 0.9)]
CAL = PianoCalibration.from_corners(RECT, LAYOUT.size, CameraKey(0, 320, 240, True))
CYAN, VIOLET = (0x22, 0xD3, 0xEE), (0xA7, 0x8B, 0xFA)


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _cam(name):
    key = next(k for k in LAYOUT.keys if k.name == name)
    return tuple(CAL.to_camera([key.center])[0])


def _snapshot(tips, seq=1, mapper=None):
    state = _state_with_tips(tips)
    image = np.full((240, 320, 3), 40, np.uint8)
    image.setflags(write=False)
    mapping = (mapper or KeyMapper(LAYOUT, CAL)).map(state)
    return TrackingSnapshot(
        seq=seq, frame=Frame(seq, image, 0, 1), state=state, timestamps=FrameTimestamps(0, 1, 2, 3, 4, 5, 6), mapping=mapping
    )


def _dist(color, rgb):
    return sum((a - b) ** 2 for a, b in zip((color.red(), color.green(), color.blue()), rgb, strict=True)) ** 0.5


def _tinted(color, hand_rgb):
    """Hover is a translucent hand colour over the key: closer to this hand's colour than to the other's."""
    other = VIOLET if hand_rgb == CYAN else CYAN
    return _dist(color, hand_rgb) < _dist(color, other) and _dist(color, (0xEC, 0xEB, 0xF5)) > 30


def _pixel_at_key(window, name, dy_fraction=0.7):
    view = window.piano_view
    rect = view.key_rect(next(k for k in LAYOUT.keys if k.name == name))
    image = view.grab().toImage()
    return image.pixelColor(int(rect.center().x()), int(rect.top() + rect.height() * dy_fraction))


def test_piano_widget_highlights_the_key_under_each_hand(qapp):
    window = MainWindow(CameraSettings(), layout=LAYOUT)
    window.resize(900, 700)
    window.show()
    snap = _snapshot({FingerId.LEFT_INDEX: _cam("E4"), FingerId.RIGHT_INDEX: _cam("A5")})
    window.piano_view.set_state(snap.mapping, True)
    assert _tinted(_pixel_at_key(window, "E4"), CYAN)
    assert _tinted(_pixel_at_key(window, "A5"), VIOLET)
    plain = _pixel_at_key(window, "D4")
    assert (plain.red(), plain.green(), plain.blue()) == (0xEC, 0xEB, 0xF5)  # untouched white key
    window.close()


def test_black_key_hover(qapp):
    window = MainWindow(CameraSettings(), layout=LAYOUT)
    window.resize(900, 700)
    window.show()
    snap = _snapshot({FingerId.RIGHT_INDEX: _cam("F#5")})
    window.piano_view.set_state(snap.mapping, True)
    assert _tinted(_pixel_at_key(window, "F#5", 0.3), VIOLET)
    window.close()


def test_two_hands_on_one_key_share_it(qapp):
    window = MainWindow(CameraSettings(), layout=LAYOUT)
    window.resize(900, 700)
    window.show()
    p = _cam("G4")
    window.piano_view.set_state(_snapshot({FingerId.LEFT_INDEX: p, FingerId.RIGHT_INDEX: p}).mapping, True)
    view = window.piano_view
    rect = view.key_rect(LAYOUT.key_by_midi(67))
    image = view.grab().toImage()
    y = int(rect.top() + rect.height() * 0.95)
    left = image.pixelColor(int(rect.left() + rect.width() * 0.2), y)
    right = image.pixelColor(int(rect.left() + rect.width() * 0.8), y)
    assert _tinted(left, CYAN) and _tinted(right, VIOLET)
    window.close()


def test_uncalibrated_piano_shows_notice_and_no_hover(qapp):
    window = MainWindow(CameraSettings(), layout=LAYOUT)
    window.resize(900, 700)
    window.show()
    uncalibrated = _snapshot({FingerId.RIGHT_INDEX: (0.5, 0.7)}, mapper=KeyMapper(LAYOUT))
    window.piano_view.set_state(uncalibrated.mapping, False, "Calibración necesaria")
    assert window.piano_view._notice == "Calibración necesaria"
    pixel = _pixel_at_key(window, "C4")
    # Exactly the dimmed 'uncalibrated' white, with no hover tint at all.
    assert (pixel.red(), pixel.green(), pixel.blue()) == (0x6B, 0x6B, 0x80)
    window.close()


def test_camera_overlay_projects_keys_and_highlights_hover(qapp):
    window = MainWindow(CameraSettings(), layout=LAYOUT)
    window.resize(640, 700)
    window.show()
    view = window.camera_view
    view.set_piano_overlay(LAYOUT, CAL)
    assert len(view._key_polygons) == 24
    # The projected C4 rectangle starts at the calibrated top-left corner.
    c4 = next(poly for key, poly in view._key_polygons if key.name == "C4")
    np.testing.assert_allclose(c4[0], RECT[0], atol=1e-9)
    view.set_snapshot(_snapshot({FingerId.LEFT_INDEX: _cam("D5")}))
    image = view.grab().toImage()
    assert not image.isNull()
    view.set_piano_overlay(LAYOUT, None)
    assert view._key_polygons == []
    window.close()


class _FakePipeline:
    def __init__(self, mapper):
        self.state = PipelineState.RUNNING
        self.mode = CameraMode(0, "fake", 320, 240, 30, 320, 240, 30.0, False, False)
        self.error_message = None
        self.snapshot = None
        self.mapper = mapper

    def latest_snapshot(self):
        return self.snapshot

    def drain_events(self):
        return []

    def stop(self):
        return True

    def stats(self):
        raise AssertionError("not needed")


def _app(tmp_path):
    config = AppConfig()
    config = replace(config, piano=replace(config.piano, calibration_path=tmp_path / "cal.json"))
    window = MainWindow(config.camera, layout=LAYOUT)
    app = Application(config, window)
    fake = _FakePipeline(app.piano.mapper)
    app.pipeline = fake
    return app, window, fake


def _feed(app, fake, tip, seq, now, monkeypatch):
    monkeypatch.setattr("handpiano.app.application.time.monotonic", lambda: now)
    tips = {} if tip is None else {FingerId.RIGHT_INDEX: tip}
    fake.snapshot = _snapshot(tips, seq=seq, mapper=fake.mapper)
    app._poll_frame()


def test_full_calibration_flow_by_dwelling(qapp, tmp_path, monkeypatch):
    app, window, fake = _app(tmp_path)
    _feed(app, fake, None, 1, 0.0, monkeypatch)  # RUNNING → tries to load: nothing saved
    assert not app.piano.calibrated
    assert window.piano_view._notice is not None
    app.start_calibration()
    assert window.camera_view._banner[0].startswith("1/4")
    seq, now = 2, 0.0
    for corner in RECT:
        for _ in range(40):  # ~1.6 s at 24 fps: longer than the 1.2 s dwell
            _feed(app, fake, corner, seq, now, monkeypatch)
            seq, now = seq + 1, now + 1 / 24
    assert app.piano.calibrated
    assert window.camera_view._banner[0] == "Piano calibrado"
    assert (tmp_path / "cal.json").is_file()
    # The new calibration maps the centre of the rectangle to the middle of the keyboard.
    np.testing.assert_allclose(app.piano.mapper.calibration.to_piano([(0.5, 0.7)]), [[7.0, LAYOUT.height / 2]], atol=1e-3)
    app.pipeline = None
    app.shutdown()
    window.close()


def test_calibration_space_capture_cancel_and_reload(qapp, tmp_path, monkeypatch):
    app, window, fake = _app(tmp_path)
    _feed(app, fake, None, 1, 0.0, monkeypatch)
    app.start_calibration()
    seq = 2
    for corner in RECT:
        _feed(app, fake, corner, seq, seq / 24, monkeypatch)
        app.piano.key_command("capture", seq / 24)
        seq += 1
    assert app.piano.calibrated

    # Cancelling a new calibration restores the previous one.
    app.start_calibration()
    assert not app.piano.calibrated
    app.piano.key_command("cancel", 1.0)
    assert app.piano.calibrated

    # A new app instance (same camera) loads it from disk.
    app2, window2, fake2 = _app(tmp_path)
    _feed(app2, fake2, None, 1, 0.0, monkeypatch)
    assert app2.piano.calibrated and "cargada" in app2.piano.status

    app2.piano.forget_calibration()
    assert not app2.piano.calibrated and not (tmp_path / "cal.json").exists()
    for a, w in ((app, window), (app2, window2)):
        a.pipeline = None
        a.shutdown()
        w.close()


def test_invalid_corners_are_rejected_and_explained(qapp, tmp_path, monkeypatch):
    app, window, fake = _app(tmp_path)
    _feed(app, fake, None, 1, 0.0, monkeypatch)
    app.start_calibration()
    for i, corner in enumerate(reversed(RECT)):  # wrong order
        _feed(app, fake, corner, i + 2, i / 24, monkeypatch)
        app.piano.key_command("capture", i / 24)
    assert not app.piano.calibrated
    assert window.camera_view._banner[0] == "Calibración no válida"
    assert "sentido contrario" in window.camera_view._banner[1][0]
    app.pipeline = None
    app.shutdown()
    window.close()


def test_lab_piano_group_is_na_without_calibration(qapp):
    window = MainWindow(CameraSettings(), layout=LAYOUT)
    dv = window.debug_view
    uncalibrated = _snapshot({FingerId.RIGHT_INDEX: (0.5, 0.7)}, mapper=KeyMapper(LAYOUT))
    dv.update_piano(LAYOUT, "NO — calibración necesaria", uncalibrated.mapping)
    assert dv.labels["Dentro del piano"].text() == NA
    assert dv.labels["Fuera del piano"].text() == NA
    assert dv.labels["Teclado"].text() == "C4–B5 · 24 teclas"

    calibrated = _snapshot({FingerId.LEFT_INDEX: _cam("C4"), FingerId.RIGHT_INDEX: _cam("C4"),
                            FingerId.RIGHT_PINKY: (0.05, 0.05)})
    dv.update_piano(LAYOUT, "OK", calibrated.mapping)
    assert dv.labels["Dentro del piano"].text().startswith("2/8") or dv.labels["Dentro del piano"].text().endswith("/8")
    assert "C4: L-IND, R-IND" in dv.labels["Teclas con varios dedos"].text()
    row = list(FingerId).index(FingerId.LEFT_INDEX)
    assert dv.table.item(row, 8).text() == "C4"
    pinky_row = list(FingerId).index(FingerId.RIGHT_PINKY)
    assert dv.table.item(pinky_row, 8).text() == "fuera"
    window.close()


def test_two_visible_hands_block_corner_capture(qapp, tmp_path, monkeypatch):
    """Regression: with both hands visible the resting hand's index was captured."""
    app, window, fake = _app(tmp_path)
    _feed(app, fake, None, 1, 0.0, monkeypatch)
    app.start_calibration()
    seq, now = 2, 0.0
    for _ in range(60):  # 2.5 s holding still, but two hands are visible
        monkeypatch.setattr("handpiano.app.application.time.monotonic", lambda n=now: n)
        state_tips = {FingerId.RIGHT_INDEX: RECT[0], FingerId.LEFT_INDEX: (0.1, 0.2)}
        fake.snapshot = _snapshot(state_tips, seq=seq, mapper=fake.mapper)
        app._poll_frame()
        seq, now = seq + 1, now + 1 / 24
    assert app.piano.session.points == []
    assert "dos manos" in window.camera_view._banner[1][0]
    app.piano.key_command("capture", now)  # Space does not bypass it either
    assert app.piano.session.points == []
    app.pipeline = None
    app.shutdown()
    window.close()
