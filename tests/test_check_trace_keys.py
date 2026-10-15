"""scripts/check_trace_keys.py: options, mandatory camera, persistent indicator, manual mode end to end.

The script runs unchanged; only the camera capture and MediaPipe are replaced by test doubles.
Keys are Qt key events sent to the window's QWindow (where the platform delivers them), never a
global keyboard injection. The automatic (Cocoa) mode is covered by the opt-in test in
test_trace_keys.py (HANDPIANO_GUI_TESTS=1).
"""

import importlib.util
import os
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402
from PySide6.QtCore import Qt, QTimer  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication, QLabel  # noqa: E402

import handpiano.app.application as appmod  # noqa: E402
from handpiano.camera.camera_manager import CameraManager  # noqa: E402
from handpiano.tracking.landmarks import Handedness  # noqa: E402
from tests.fakes import FakeCapture, make_hand  # noqa: E402

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_trace_keys.py"
KEYS = (Qt.Key.Key_A, Qt.Key.Key_S, Qt.Key.Key_D, Qt.Key.Key_F, Qt.Key.Key_J, Qt.Key.Key_K, Qt.Key.Key_L,
        Qt.Key.Key_Ntilde)


@pytest.fixture
def script():
    QApplication.instance() or QApplication([])
    spec = importlib.util.spec_from_file_location("check_trace_keys", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeDetector:
    def detect(self, rgb, timestamp_ms):
        return [make_hand(Handedness.LEFT, (0.35, 0.6)), make_hand(Handedness.RIGHT, (0.65, 0.6))]

    def close(self):
        pass


def use_camera(monkeypatch, capture_factory):
    monkeypatch.setattr(appmod, "CameraManager", lambda settings: CameraManager(settings, factory=capture_factory))
    monkeypatch.setattr(appmod, "HandTracker", lambda *a, **k: FakeDetector())


def recording_window():
    return next(w for w in QApplication.topLevelWidgets() if type(w).__name__ == "MainWindow" and w.isVisible())


def press_keys_later(delay_ms, seen):
    """Once the test is running: record what the window shows, then send the 8 keys to its QWindow."""
    def go():
        window = next((w for w in QApplication.topLevelWidgets()
                       if type(w).__name__ == "MainWindow" and w.isVisible()), None)
        indicator = None if window is None else window.camera_view.findChild(QLabel, "ManualKeyTestIndicator")
        if indicator is None:  # the test (or the camera) has not started yet: on a busy machine, try again
            QTimer.singleShot(200, go)
            return
        seen["indicator_before"] = None if indicator is None else (indicator.isVisible(), indicator.text())
        seen["protocol_banner"] = window.camera_view._banner[0] if window.camera_view._banner else None
        image = window.camera_view.grab().toImage()  # what is really on screen, banner area included
        seen["banner_area_color"] = image.pixelColor(40, 40).name()
        for key in KEYS:
            QTest.keyClick(window.windowHandle(), key)
        seen["indicator_after"] = indicator.text() if indicator is not None else None
    QTimer.singleShot(delay_ms, go)


def test_unknown_argument_is_rejected(script, capsys):
    with pytest.raises(SystemExit) as exit_info:
        script.main(["--manaul"])
    assert exit_info.value.code == 2
    assert "unrecognized arguments: --manaul" in capsys.readouterr().err


def test_help_lists_the_options(script, capsys):
    with pytest.raises(SystemExit) as exit_info:
        script.main(["--help"])
    assert exit_info.value.code == 0
    out = capsys.readouterr().out
    assert "--manual" in out and "--camera" in out and "--camera-timeout" in out


def test_manual_with_running_camera_records_the_eight_keys(script, monkeypatch, capsys):
    use_camera(monkeypatch, lambda *a, **k: FakeCapture(width=1280, height=720, delay_s=0.033))
    seen = {}
    press_keys_later(1500, seen)
    code = script.main(["--manual", "--camera", "--manual-timeout", "15"])
    out = capsys.readouterr().out
    assert code == 0 and "Cámara: running" in out and out.rstrip().endswith("PASS")
    assert "Eventos registrados: 16" in out and "16 etiquetas válidas" in out
    # The indicator was on screen, above the protocol's banner, with the instructions.
    visible, text = seen["indicator_before"]
    assert visible and "PRUEBA MANUAL DE TECLADO CON CÁMARA" in text and "A S D F J K L Ñ" in text
    assert seen["protocol_banner"] is not None  # the protocol banner was there underneath…
    assert seen["banner_area_color"] == "#22d3ee"  # …but the indicator covered it


def test_indicator_marks_the_keys_received(script):
    from PySide6.QtWidgets import QWidget

    parent = QWidget()
    parent.camera_view = QWidget(parent)
    parent.camera_view.resize(800, 500)
    label = script.show_indicator(parent, with_camera=True)
    script.update_indicator(label, {"A", "Ñ"})
    text = label.text()
    assert "A ✓" in text and "Ñ ✓" in text and "S ✓" not in text and "Ignora los carteles" in text
    assert label.geometry().top() <= 16 and label.geometry().height() >= 150  # covers the banner area


def test_manual_with_failed_camera_never_passes(script, monkeypatch, capsys):
    use_camera(monkeypatch, lambda *a, **k: FakeCapture(opened=False))
    code = script.main(["--manual", "--camera", "--camera-timeout", "5"])
    out = capsys.readouterr().out
    assert code == 3 and "PASS" not in out
    assert "Cámara: failed" in out and "ERROR: la cámara no está en 'running'" in out


def test_manual_with_camera_stuck_in_starting_never_passes(script, monkeypatch, capsys):
    def slow_open(*a, **k):
        time.sleep(2.0)  # the driver does not answer in time
        return FakeCapture()

    use_camera(monkeypatch, slow_open)
    t0 = time.monotonic()
    code = script.main(["--manual", "--camera", "--camera-timeout", "0.5"])
    out = capsys.readouterr().out
    assert code == 3 and "PASS" not in out
    assert "Cámara: starting" in out and "NO se ha realizado" in out
    assert time.monotonic() - t0 < 10


def test_manual_without_camera_shows_the_indicator_and_passes(script, capsys):
    seen = {}
    press_keys_later(800, seen)
    code = script.main(["--manual", "--manual-timeout", "15"])
    out = capsys.readouterr().out
    assert code == 0 and out.rstrip().endswith("PASS") and "Cámara:" not in out
    assert seen["indicator_before"][0] and "SIN CÁMARA" in seen["indicator_before"][1]


def test_missing_keys_fail_after_the_timeout(script, capsys):
    code = script.main(["--manual", "--manual-timeout", "1"])
    out = capsys.readouterr().out
    assert code == 1 and "PASS" not in out and "Tiempo agotado" in out


def test_camera_that_stops_during_the_test_invalidates_it(script, monkeypatch, capsys):
    use_camera(monkeypatch, lambda *a, **k: FakeCapture(width=1280, height=720, delay_s=0.033, frames=60))
    code = script.main(["--manual", "--camera", "--manual-timeout", "15"])
    out = capsys.readouterr().out
    assert code == 3 and "PASS" not in out and "dejó de estar en 'running'" in out
