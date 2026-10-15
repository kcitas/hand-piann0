"""Hand placement guide: what it tells the player for each position of the hands over the keyboard."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from handpiano.app.placement import PlacementGuide, keyboard_box  # noqa: E402
from handpiano.tracking.finger_tracker import FingerTracker  # noqa: E402
from handpiano.tracking.landmarks import Handedness  # noqa: E402
from tests.fakes import make_hand  # noqa: E402

BOX = (0.15, 0.30, 0.85, 0.62)  # the screen piano
# make_hand(scale=0.1): fingertips at centre_y − 0.075, wrist at centre_y + 0.1.
GOOD = {Handedness.LEFT: (0.35, 0.575), Handedness.RIGHT: (0.65, 0.575)}


def run(positions, seconds=1.0, guide=None, start=0.0):
    tracker = FingerTracker()
    guide = guide or PlacementGuide()
    placement = None
    t = start
    while t < start + seconds:
        obs = [make_hand(h, c) for h, c in positions.items()]
        placement = guide.update(t, tracker.update(obs, t), BOX)
        t += 1 / 30
    return placement


def test_both_hands_on_the_keyboard_is_ok():
    p = run(GOOD)
    assert p.ok and p.text == "Manos bien colocadas"


@pytest.mark.parametrize("centre, expected", [
    ((0.65, 0.78), "Sube la mano derecha: las puntas están por debajo del piano"),
    ((0.65, 0.86), "Sube la mano derecha: la muñeca se sale de la imagen"),
    ((0.65, 0.28), "Baja la mano derecha: las puntas están por encima del piano"),
    ((0.97, 0.575), "Mueve la mano derecha hacia el piano"),
])
def test_each_problem_has_its_instruction(centre, expected):
    p = run({Handedness.LEFT: GOOD[Handedness.LEFT], Handedness.RIGHT: centre})
    assert not p.ok and p.messages == (expected,)


def test_a_missing_hand_is_reported_after_a_moment():
    p = run({Handedness.LEFT: GOOD[Handedness.LEFT]})
    assert p.messages == ("No veo la mano derecha: ponla entera dentro de la imagen",)


def test_both_problems_are_listed():
    p = run({Handedness.LEFT: (0.35, 0.86), Handedness.RIGHT: (0.65, 0.78)})
    assert len(p.messages) == 2 and "izquierda" in p.messages[0] and "derecha" in p.messages[1]


def test_messages_do_not_flicker():
    guide = PlacementGuide()
    tracker = FingerTracker()
    good = [make_hand(h, c) for h, c in GOOD.items()]
    low = [make_hand(Handedness.LEFT, GOOD[Handedness.LEFT]), make_hand(Handedness.RIGHT, (0.65, 0.78))]

    def feed(frames, t0, seconds):
        shown, t = [], t0
        while t < t0 + seconds - 1e-9:
            shown.append((round(t - t0, 3), guide.update(t, tracker.update(frames, t), BOX).ok))
            t += 1 / 30
        return shown

    assert feed(good, 0.0, 1.0)[-1][1]
    # A brief bad moment (0.2 s) never shows a warning…
    assert all(ok for _, ok in feed(low, 1.0, 0.2))
    feed(good, 1.2, 1.0)
    # …a lasting one does, and stays until the hand is placed again.
    shown = feed(low, 2.2, 2.0)
    first_warning = next(t for t, ok in shown if not ok)
    assert 0.3 <= first_warning <= 1.2 and not any(ok for t, ok in shown if t > first_warning)


def test_no_keyboard_says_so():
    tracker = FingerTracker()
    guide = PlacementGuide()
    state = tracker.update([], 0.0)
    for t in (0.0, 0.5, 1.0):
        p = guide.update(t, state, None)
    assert not p.ok and "Sin piano" in p.text


def test_keyboard_box_from_corners():
    assert keyboard_box([(0.2, 0.4), (0.8, 0.45), (0.78, 0.7), (0.22, 0.72)]) == (0.2, 0.4, 0.8, 0.72)


def test_the_app_shows_the_guide_over_the_video(tmp_path):
    from dataclasses import replace

    import numpy as np

    from handpiano.app.application import Application
    from handpiano.app.config import AppConfig
    from handpiano.app.pipeline import FrameTimestamps, TrackingSnapshot
    from handpiano.audio.engine import AudioEngine
    from handpiano.camera.camera_config import CameraMode
    from handpiano.camera.frame import Frame
    from handpiano.ui.main_window import MainWindow

    QApplication.instance() or QApplication([])
    config = AppConfig()
    config = replace(config, piano=replace(config.piano, calibration_path=tmp_path / "c.json", screen=True))
    window = MainWindow(config.camera)
    app = Application(config, window, audio=AudioEngine(sounds_dir=tmp_path / "none"))
    app.piano.on_camera_running(CameraMode(0, "fake", 640, 360, 30, 640, 360, 30.0, False, False))
    tracker = FingerTracker()
    image = np.zeros((360, 640, 3), np.uint8)
    for i in range(30):  # 1 s with the right hand too low
        t = i / 30
        state = tracker.update([make_hand(Handedness.LEFT, GOOD[Handedness.LEFT]),
                                make_hand(Handedness.RIGHT, (0.65, 0.78))], t)
        snap = TrackingSnapshot(i + 1, Frame(i + 1, image, int(t * 1e9), 0), state, FrameTimestamps(*[0] * 7))
        app.placement.update(t, state, (0.15, 0.30, 0.85, 0.62))
        app._update_placement(snap)
    text, ok = window.camera_view._placement
    assert not ok and "Sube la mano derecha" in text
    window.camera_view.resize(640, 360)
    window.camera_view.set_snapshot(snap)
    assert not window.camera_view.grab().isNull()  # painted with the bar
    app.shutdown()
    window.close()


def test_press_meter_fraction():
    from handpiano.ui.camera_view import METER_TOP, meter_fraction

    assert meter_fraction(None, 0.12) == 0.0 and meter_fraction(-0.05, 0.12) == 0.0
    assert meter_fraction(0.06, 0.12) == pytest.approx(0.5) and meter_fraction(0.12, 0.12) == pytest.approx(1.0)
    assert meter_fraction(5.0, 0.12) == METER_TOP


def test_press_meter_turns_green_on_a_real_press():
    import numpy as np
    from PySide6.QtGui import QColor

    from handpiano.app.pipeline import FrameTimestamps, TrackingSnapshot
    from handpiano.camera.frame import Frame
    from handpiano.tracking.landmarks import FingerId
    from handpiano.ui.camera_view import CameraView
    from tests.test_press_detector import Rig, finger_tap

    QApplication.instance() or QApplication([])
    rig = Rig()
    states = []
    original = rig.tracker.update
    rig.tracker.update = lambda obs, t: states.append(original(obs, t)) or states[-1]
    pressed_frame = None
    for offsets in finger_tap(FingerId.RIGHT_INDEX):
        out = rig.frame(offsets)
        if out.fingers[FingerId.RIGHT_INDEX].phase.value == "PRESSED":
            pressed_frame = (states[-1], out)
            break
    assert pressed_frame is not None
    view = CameraView()
    view.resize(640, 360)
    view.set_press_threshold(0.12)
    image = np.zeros((360, 640, 3), np.uint8)
    view.set_snapshot(TrackingSnapshot(1, Frame(1, image, 0, 0), pressed_frame[0], FrameTimestamps(*[0] * 7),
                                       None, pressed_frame[1]))
    picture = view.grab().toImage()
    tip = pressed_frame[0].fingers[FingerId.RIGHT_INDEX].tip
    target = view._fit_rect(640, 360)
    centre = view._to_widget(target, float(tip[0]), float(tip[1]))
    # Just right of the fingertip, low in the meter: filled in green while the finger is pressed.
    pixel = picture.pixelColor(int(centre.x() + 14), int(centre.y() + 14))
    assert pixel.green() > 150 and pixel.red() < 120, pixel.name()
    assert QColor("#34d399").green() == 0xD3
    view.close()


def test_finger_note_mode_tells_how_to_hold_the_hands():
    import numpy as np

    from handpiano.app.placement import PlacementConfig, assess_posture

    cfg = PlacementConfig()
    hand = make_hand(Handedness.RIGHT, (0.65, 0.5), scale=0.25).landmarks  # palm ≈ 0.11 frame widths, as measured
    assert assess_posture(Handedness.RIGHT, hand, 0.5625, cfg) is None  # palm to the camera, fingers up
    down = hand.copy()
    down[:, 1] = 1.0 - down[:, 1]  # fingers pointing down
    down[:, 1] -= 0.05
    assert "ARRIBA" in assess_posture(Handedness.RIGHT, down, 0.5625, cfg)
    short = hand.copy()
    for tip, base in ((8, 5), (12, 9), (16, 13), (20, 17)):
        short[tip, :2] = short[base, :2] + 0.3 * (short[tip, :2] - short[base, :2])  # foreshortened fingers
    assert "palma" in assess_posture(Handedness.RIGHT, short, 0.5625, cfg)
    far = make_hand(Handedness.RIGHT, (0.65, 0.5), scale=0.03).landmarks
    assert "Acerca" in assess_posture(Handedness.RIGHT, far, 0.5625, cfg)
    low = hand.copy()
    low[:, 1] += 0.25
    assert "muñeca" in assess_posture(Handedness.RIGHT, low, 0.5625, cfg)
    assert np.isfinite(hand).all()
