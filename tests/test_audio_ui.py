"""Instrument panel, audio test window, modes and the rhythm game in the app (offscreen Qt, silent engine)."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import time  # noqa: E402

import pytest  # noqa: E402
from PySide6.QtCore import QEvent, Qt  # noqa: E402
from PySide6.QtGui import QKeyEvent  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from handpiano.app.application import Application  # noqa: E402
from handpiano.app.audio_controller import TEST_NOTE, AudioController, NoteKeys  # noqa: E402
from handpiano.app.config import AppConfig  # noqa: E402
from handpiano.audio.engine import AudioEngine  # noqa: E402
from handpiano.camera.camera_config import CameraSettings  # noqa: E402
from handpiano.game.chart import Chart, ChartNote, practice_pattern  # noqa: E402
from handpiano.interaction.press_detector import PressEvent  # noqa: E402
from handpiano.tracking.landmarks import FingerId  # noqa: E402
from handpiano.ui.audio_test_window import AudioTestWindow  # noqa: E402
from handpiano.ui.instrument_panel import InstrumentPanel  # noqa: E402
from handpiano.ui.main_window import MainWindow  # noqa: E402
from tests.audio_fakes import make_all  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture(scope="module")
def sounds(tmp_path_factory):
    return make_all(tmp_path_factory.mktemp("ui_sounds"), seconds=0.3)


def commands(engine, kinds=("on", "off")):
    return [c for c in engine.mixer.pending if c[0] in kinds]


def key(kind, text):
    return QKeyEvent(kind, Qt.Key.Key_A, Qt.KeyboardModifier.NoModifier, text, False)


def test_panel_shows_three_cards_and_the_active_one(qapp):
    panel = InstrumentPanel()
    assert [c.instrument_id for c in panel.cards.values()] == ["grand", "upright", "electric"]
    assert "GRAND PIANO" in panel.cards["grand"].name.text() and panel.cards["electric"].desc.text()
    selected = []
    panel.instrument_selected.connect(selected.append)
    panel.cards["upright"].clicked.emit("upright")
    assert selected == ["upright"]
    panel.set_active("electric")
    assert panel.active == "electric" and panel.cards["electric"].chip.isVisibleTo(panel.cards["electric"])
    assert not panel.cards["grand"].active
    volumes = []
    panel.volume_changed.connect(volumes.append)
    panel.volume.setValue(30)
    assert volumes == [0.3] and panel.volume_label.text() == "VOLUMEN 30 %"


def test_controller_follows_the_engine_and_plays_a_test_note(qapp, sounds):
    engine = AudioEngine(sounds_dir=sounds)
    panel = InstrumentPanel()
    controller = AudioController(engine, panel)
    controller.select("electric")
    assert engine.wait_ready()
    controller.refresh()
    assert panel.active == "electric" and "sonando" in panel.cards["electric"].state.text()
    engine.mixer._commands.clear()
    controller.test_note("grand")  # selects grand and plays C4 when it is ready
    deadline = time.monotonic() + 5
    while not commands(engine) and time.monotonic() < deadline:
        controller.refresh()
        qapp.processEvents()
        time.sleep(0.01)
    assert commands(engine)[0] == ("on", TEST_NOTE, 96) and engine.status().instrument == "grand"
    controller.change_volume(-0.5)
    assert engine.volume == pytest.approx(0.3) and panel.volume.value() == 30
    assert controller.handle_shortcut("3") and engine.status().instrument == "electric"


def test_missing_samples_are_shown_on_the_card(qapp, tmp_path):
    engine = AudioEngine(sounds_dir=tmp_path)
    panel = InstrumentPanel()
    controller = AudioController(engine, panel)
    controller.select("upright")
    engine.wait_ready(timeout=5)
    controller.refresh()
    assert panel.cards["upright"].state.text().startswith("⚠") and panel.active is None


def test_note_keys_layout_octaves_and_held_keys():
    keys = NoteKeys(base=60)
    assert keys.press("a") == 60 and keys.press("W") == 61 and keys.press("k") == 72
    assert keys.press("a") is None  # already held (auto-repeat / duplicate event)
    keys.press("x")  # octave up while A is held: A still releases ITS note
    assert keys.release("A") == 60 and keys.press("A") == 72
    assert sorted(keys.release_all()) == [61, 72, 72]


def test_audio_test_window_plays_chords_from_the_keyboard(qapp, sounds):
    engine = AudioEngine(sounds_dir=sounds)
    window = AudioTestWindow(engine)
    for text in ("a", "d", "g"):  # C E G at once
        window.keyPressEvent(key(QEvent.Type.KeyPress, text))
    assert [c[1] for c in commands(engine)] == [60, 64, 67] and window.pads.lit == {60, 64, 67}
    for text in ("a", "d", "g"):
        window.keyReleaseEvent(key(QEvent.Type.KeyRelease, text))
    assert [c[0] for c in commands(engine)] == ["on"] * 3 + ["off"] * 3
    window.keyPressEvent(key(QEvent.Type.KeyPress, "2"))
    assert engine.status().instrument == "upright"
    window.pads.resize(800, 300)
    window.pads.grab()
    window.close()


def _app(qapp, sounds):
    window = MainWindow(CameraSettings())
    engine = AudioEngine(sounds_dir=sounds)
    return Application(AppConfig(), window, audio=engine), window, engine


def test_modes_route_hand_presses_to_piano_or_game(qapp, sounds):
    app, window, _ = _app(qapp, sounds)
    assert window.mode == "piano" and app.press_sound.enabled
    window.set_mode("rhythm")
    assert window.mode_stack.currentWidget() is window.rhythm_view and not app.press_sound.enabled
    window.set_mode("piano")
    assert app.press_sound.enabled
    with pytest.raises(ValueError):
        window.set_mode("karaoke")
    app.shutdown()
    window.close()


def test_rhythm_hit_plays_the_chart_note_once_per_press_event(qapp, sounds):
    app, window, engine = _app(qapp, sounds)
    window.set_mode("rhythm")
    layout = app.piano.layout
    left_key = layout.keys[0]
    assert app.lane_for_key(left_key) == 0 and app.lane_for_key(layout.keys[-1]) == 3
    now = time.perf_counter()
    song = Chart("t", (ChartNote(0.0, 0, 67, 0.2), ChartNote(0.3, 0, 69, 0.2)))
    app.game.start(song, now, lead_in=0.0)
    press = PressEvent("press", FingerId.LEFT_INDEX, now + 0.01, left_key, "ok", 0.2, 1.0)
    app._update_game([press])
    app._update_game([])  # later polls: the event was drained once, nothing re-judged
    app._update_game([])
    hits = [j for j in app.game.judgments if j.grade != "MISS"]
    assert len(hits) == 1 and hits[0].note.midi == 67
    assert commands(engine) == [("on", 67, 100)]  # the CHART's note sounds, not the pressed key's
    window.rhythm_view.canvas.resize(700, 400)
    window.rhythm_view.canvas.grab()  # paints a running game without errors
    app.stop_song()
    assert not app.game.running
    app.shutdown()
    window.close()


def test_keyboard_alternatives_in_the_app(qapp, sounds):
    app, window, engine = _app(qapp, sounds)
    first = app.piano.layout.keys[0].midi
    assert app.eventFilter(window, key(QEvent.Type.KeyPress, "a")) is True
    app.eventFilter(window, key(QEvent.Type.KeyRelease, "a"))
    assert [c[:2] for c in commands(engine)] == [("on", first), ("off", first)]
    engine.mixer._commands.clear()
    window.set_mode("rhythm")
    app.start_song(practice_pattern())
    app.eventFilter(window, key(QEvent.Type.KeyPress, "d"))
    app.eventFilter(window.rhythm_view, key(QEvent.Type.KeyPress, "d"))  # same event seen by another widget
    assert app._rhythm_keys_down == {"D"}
    app.eventFilter(window, key(QEvent.Type.KeyRelease, "d"))
    assert commands(engine) == []  # pressing a lane with no note due plays nothing
    app.eventFilter(window, key(QEvent.Type.KeyPress, "+"))
    assert engine.volume == pytest.approx(0.9)
    app.shutdown()
    window.close()


def test_mouse_on_the_piano_view_plays_notes(qapp, sounds):
    app, window, engine = _app(qapp, sounds)
    window.piano_view.key_pressed.emit(62)
    window.piano_view.key_released.emit(62)
    assert [c[:2] for c in commands(engine)] == [("on", 62), ("off", 62)]
    view = window.piano_view
    view.resize(600, 150)
    k = app.piano.layout.keys[0]
    centre = view.key_rect(k).center()
    assert view.key_at_widget(centre).midi == k.midi
    app.shutdown()
    window.close()


def test_press_to_sound_diagnostics_are_visible(qapp, sounds, caplog):
    from tests.test_press_detector import Rig, finger_tap

    app, window, engine = _app(qapp, sounds)
    engine.set_instrument("grand")
    assert engine.wait_ready()
    app.debug_notes = True
    rig = Rig()
    for offsets in finger_tap(FingerId.RIGHT_INDEX):
        out = rig.frame(offsets)
        app._on_presses(out)  # exactly what the pipeline calls on the tracking thread
        engine.mixer.render(256)
    with caplog.at_level("WARNING", logger="handpiano.app.application"):
        app._report_notes()
        app._report_notes()  # reported once
    labels = window.debug_view.labels
    assert labels["Pulsaciones → note_on"].text() == "1 → 1 encoladas"
    assert labels["Motor de audio"].text() == "1 voces iniciadas · 0 sin banco"
    assert labels["Liberaciones"].text() == "1 dedos levantados → 1 note_off"
    lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith("[nota #")]
    assert len(lines) == 1 and "voz iniciada (grand" in lines[0] and "note_off encolada" in lines[0]
    app.shutdown()
    window.close()
