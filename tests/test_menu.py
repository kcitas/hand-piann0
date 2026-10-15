"""Start menu: one card per mode; a mode opened from it can go back with "☰ Menú"."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

from handpiano.camera.camera_config import CameraSettings  # noqa: E402
from handpiano.ui.main_window import MainWindow  # noqa: E402
from handpiano.ui.menu_window import MODES, MenuWindow  # noqa: E402


def test_menu_offers_every_mode_and_says_which_was_chosen():
    QApplication.instance() or QApplication([])
    menu = MenuWindow()
    chosen = []
    menu.chosen.connect(chosen.append)
    assert [m for m, _, _ in MODES] == ["libre", "juego", "dibujado", "sonido"]
    for mode in ("libre", "juego", "dibujado", "sonido"):
        menu.buttons[mode].click()
    assert chosen == ["libre", "juego", "dibujado", "sonido"]
    assert "dóblalo" in menu.buttons["libre"].accessibleName() and "Piano dibujado" in menu.buttons["dibujado"].accessibleName()
    menu.close()


def test_menu_button_closes_the_mode_and_reports_it():
    QApplication.instance() or QApplication([])
    window = MainWindow(CameraSettings())
    assert not window.menu_button.isVisible()
    window.enable_menu_button()
    window.show()
    closed = []
    window.closed.connect(lambda: closed.append(True))
    window.menu_button.click()
    assert closed == [True] and not window.isVisible()


def test_without_mode_options_the_app_starts_with_the_menu():
    from handpiano.app.main import DIRECT_MODE_ARGS, parse_args

    args = parse_args(["--confianza", "0.3", "--instrument", "upright"])
    assert not any(getattr(args, a) for a in DIRECT_MODE_ARGS)  # → menu
    assert getattr(parse_args(["--piano-mesa"]), "piano_mesa")  # → straight into that mode


def test_closing_a_mode_stops_its_instrument_panel_refresh(tmp_path, monkeypatch):
    """Closing a mode from the menu deleted its panel while a 150 ms timer kept refreshing it
    ("Internal C++ object (QLabel) already deleted", over and over)."""
    import handpiano.app.audio_controller as audio_controller
    from handpiano.app.application import Application
    from handpiano.app.config import AppConfig
    from handpiano.audio.engine import AudioEngine

    QApplication.instance() or QApplication([])
    window = MainWindow(CameraSettings())
    app = Application(AppConfig(), window, audio=AudioEngine(sounds_dir=tmp_path / "none"))
    controller = app.audio_controller
    window.enable_menu_button()
    window.show()
    window.closed.connect(app.shutdown)
    window.menu_button.click()  # back to the menu: the mode shuts down
    assert controller._stopped and not controller._timer.isActive()

    # Even if a refresh still arrives after the panel is gone, it stops instead of touching it.
    other = Application(AppConfig(), MainWindow(CameraSettings()), audio=AudioEngine(sounds_dir=tmp_path / "x"))
    monkeypatch.setattr(audio_controller, "isValid", lambda obj: False)
    other.audio_controller.refresh()
    assert other.audio_controller._stopped
    other.shutdown()


def test_calibration_buttons_only_for_the_table_piano(tmp_path):
    from dataclasses import replace

    from handpiano.app.application import Application
    from handpiano.app.config import AppConfig
    from handpiano.audio.engine import AudioEngine

    QApplication.instance() or QApplication([])
    for screen, visible in ((True, False), (False, True)):
        config = AppConfig()
        config = replace(config, piano=replace(config.piano, screen=screen, calibration_path=tmp_path / "c.json"))
        window = MainWindow(config.camera)
        app = Application(config, window, audio=AudioEngine(sounds_dir=tmp_path / "none"))
        window.show()
        assert window.calibrate_button.isVisible() is visible and window.forget_button.isVisible() is visible
        app.shutdown()
        window.close()
