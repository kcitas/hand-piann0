"""Screen piano: the keyboard drawn at a fixed place of the camera image (no calibration, nothing saved)."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from dataclasses import replace  # noqa: E402

import pytest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from handpiano.app.application import Application  # noqa: E402
from handpiano.app.config import AppConfig  # noqa: E402
from handpiano.app.main import parse_args  # noqa: E402
from handpiano.camera.camera_config import CameraMode  # noqa: E402
from handpiano.instrument.keyboard import KeyboardLayout  # noqa: E402
from handpiano.mapping.calibration import CameraKey  # noqa: E402
from handpiano.ui.main_window import MainWindow  # noqa: E402

LAYOUT = KeyboardLayout.from_octaves("C4", 2)


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _app(tmp_path, screen=True):
    config = AppConfig()
    stored = tmp_path / "calibration.json"
    stored.write_text('{"keep": "this file must not change"}')
    config = replace(config, piano=replace(config.piano, calibration_path=stored, screen=screen))
    window = MainWindow(config.camera, layout=LAYOUT)
    return Application(config, window), window, stored


def _mode(width, height):
    return CameraMode(0, "fake", width, height, 30, width, height, 30.0, False, False)


def test_screen_piano_needs_no_calibration_and_never_touches_the_stored_one(qapp, tmp_path):
    app, window, stored = _app(tmp_path)
    before = stored.read_bytes()
    app.piano.on_camera_running(_mode(1280, 720))
    assert app.piano.calibrated and "pantalla" in app.piano.status
    assert len(window.camera_view._key_polygons) == len(LAYOUT.keys)  # drawn over the camera image
    app.piano.start_calibration(0.0, 720 / 1280)
    assert app.piano.session is None and window.camera_view._banner[0] == "Piano en pantalla"
    app.piano.forget_calibration()
    assert app.piano.calibrated and stored.read_bytes() == before
    app.shutdown()
    window.close()


def test_fingertips_map_to_keys_inside_the_fixed_area(qapp, tmp_path):
    app, window, _ = _app(tmp_path)
    left, top, right, bottom = app.config.piano.screen_area
    cal = app.piano.screen_calibration(CameraKey(0, 1280, 720, True))
    width, height = LAYOUT.size
    # Corners of the area are the corners of the keyboard; the centre is the middle of it.
    assert cal.to_piano([(left, top)])[0] == pytest.approx((0.0, 0.0), abs=1e-9)
    assert cal.to_piano([(right, bottom)])[0] == pytest.approx((width, height), abs=1e-9)
    x, y = cal.to_piano([((left + right) / 2, bottom - 0.01)])[0]
    key = LAYOUT.key_at(x, y)
    assert key is not None and not key.is_black and key.name in ("B4", "C5")
    assert LAYOUT.key_at(*cal.to_piano([(left - 0.05, (top + bottom) / 2)])[0]) is None  # outside the area
    app.shutdown()
    window.close()


def test_same_geometry_at_any_resolution(qapp, tmp_path):
    app, window, _ = _app(tmp_path)
    a = app.piano.screen_calibration(CameraKey(0, 1280, 720, True))
    b = app.piano.screen_calibration(CameraKey(0, 640, 480, False))
    point = [(0.4, 0.6)]
    assert a.to_piano(point)[0] == pytest.approx(b.to_piano(point)[0])
    app.shutdown()
    window.close()


def test_table_mode_still_uses_the_stored_calibration(qapp, tmp_path):
    app, window, _ = _app(tmp_path, screen=False)
    app.piano.on_camera_running(_mode(1280, 720))
    assert not app.piano.calibrated  # the stored file is not a valid calibration: nothing invented
    app.shutdown()
    window.close()


def test_command_line_defaults_to_screen_piano():
    assert parse_args([]).piano_mesa is False
    assert parse_args(["--piano-mesa"]).piano_mesa is True
