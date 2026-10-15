"""Microphone taps (synthetic audio) and their fusion with the REAL press detector (Rig): which finger, when."""

import numpy as np
import pytest

from handpiano.audio.tap_input import OnsetDetector, TapOnset
from handpiano.interaction.tap_fusion import TapFusion
from handpiano.tracking.landmarks import FingerId
from tests.test_press_detector import DT, Rig, finger_tap

RATE = 44100


def click(rate=RATE, amp=0.3, ms=4.0, seed=0):
    rng = np.random.default_rng(seed)
    n = int(rate * ms / 1000)
    return amp * rng.standard_normal(n) * np.exp(-np.arange(n) / (n / 4))


def run_detector(signal, block=128, t0=100.0, detector=None):
    detector = detector or OnsetDetector(RATE)
    onsets = []
    for i in range(0, len(signal), block):
        onsets += detector.process(signal[i:i + block], t0 + i / RATE)
    return onsets


def background(seconds, level=0.001, seed=1):
    return level * np.random.default_rng(seed).standard_normal(int(seconds * RATE))


def test_taps_are_detected_once_and_on_time():
    x = background(2.0)
    for at in (0.5, 1.0, 1.5):
        c = click(seed=int(at * 10))
        x[int(at * RATE): int(at * RATE) + len(c)] += c
    onsets = run_detector(x)
    assert [round(o.t - 100.0, 2) for o in onsets] == [0.5, 1.0, 1.5]
    assert all(abs((o.t - 100.0) - at) < 0.003 for o, at in zip(onsets, (0.5, 1.0, 1.5), strict=True))


def test_noise_hum_and_a_piano_tone_are_not_taps():
    t = np.arange(int(2.0 * RATE)) / RATE
    hum = 0.05 * np.sin(2 * np.pi * 120 * t)
    attack = np.minimum(1.0, t / 0.03)  # a piano note through the room: 30 ms attack, decaying
    tone = 0.05 * attack * np.exp(-t * 2) * np.sin(2 * np.pi * 523.25 * t)
    assert run_detector(background(2.0) + hum + tone) == []


def test_our_own_note_raises_the_threshold_briefly():
    detector = OnsetDetector(RATE)
    x = background(1.0)
    soft = click(amp=0.03)
    x[int(0.50 * RATE): int(0.50 * RATE) + len(soft)] += soft  # quiet click right after our note
    x[int(0.80 * RATE): int(0.80 * RATE) + len(soft)] += soft  # same click later
    detector.note_played(100.48)
    onsets = run_detector(x, detector=detector)
    assert [round(o.t - 100.0, 1) for o in onsets] == [0.8]


def test_fast_double_hits_respect_the_refractory_period():
    x = background(1.0)
    for at in (0.30, 0.32, 0.50):
        c = click()
        x[int(at * RATE): int(at * RATE) + len(c)] += c
    assert [round(o.t - 100.0, 2) for o in run_detector(x)] == [0.3, 0.5]


# --- fusion with the real detector ---------------------------------------------------------------


def play(frames, onsets_at_frame, rig=None):
    """onsets_at_frame: frame index → list of fingers' tap (onset placed at that frame's time)."""
    rig = rig or Rig()
    fusion = TapFusion()
    events = []
    for i, offsets in enumerate(frames):
        t = rig.t
        out = rig.frame(offsets)
        if i in onsets_at_frame:
            fusion.add_onsets([TapOnset(t, 20.0)])
        events += [(e.kind, e.finger, None if e.key is None else e.key.midi, round(e.t - t, 3)) for e in
                   fusion.on_frame(t, out)]
    return events, fusion


def test_a_tap_plays_the_finger_that_moved_down_on_its_key():
    tap = finger_tap(FingerId.RIGHT_INDEX)  # rest 10 frames, then the finger goes down
    events, fusion = play(tap, {12: True})  # the table is hit when the finger reaches the bottom
    presses = [e for e in events if e[0] == "press"]
    assert [(e[1]) for e in presses] == [FingerId.RIGHT_INDEX]
    assert any(e[0] == "release" and e[1] == FingerId.RIGHT_INDEX for e in events)  # finger came back up
    assert fusion.stats.unmatched == 0 and fusion.held_keys == {}


def test_a_tap_with_no_moving_finger_plays_nothing():
    events, fusion = play([{}] * 40, {20: True})
    assert events == [] and fusion.stats.unmatched == 1


def test_a_two_finger_chord_plays_both_fingers():
    chord = [{**a, **b} for a, b in zip(finger_tap(FingerId.LEFT_INDEX), finger_tap(FingerId.RIGHT_INDEX),
                                         strict=True)]
    events, _ = play(chord, {12: True})
    assert {e[1] for e in events if e[0] == "press"} == {FingerId.LEFT_INDEX, FingerId.RIGHT_INDEX}


def test_the_moment_of_the_note_is_the_microphone_onset_not_the_camera_frame():
    rig = Rig()
    fusion = TapFusion()
    tap = finger_tap(FingerId.RIGHT_INDEX)
    onset_t = None
    played = []
    for i, offsets in enumerate(tap):
        t = rig.t
        out = rig.frame(offsets)
        if i == 12:
            onset_t = t + DT / 3  # between two frames
            fusion.add_onsets([TapOnset(onset_t, 20.0)])
        played += [e for e in fusion.on_frame(t, out) if e.kind == "press"]
    assert len(played) == 1 and played[0].t == pytest.approx(onset_t)


def test_microphone_mode_in_the_app_plays_taps_and_lights_the_sounding_keys(tmp_path):
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    from handpiano.app.application import Application
    from handpiano.app.config import AppConfig
    from handpiano.audio.engine import AudioEngine
    from handpiano.camera.camera_config import CameraSettings
    from handpiano.ui.main_window import MainWindow

    class FakeMic:
        def __init__(self):
            self.queue, self.ours = [], []

        def drain(self):
            out, self.queue = self.queue, []
            return out

        def note_played(self, t):
            self.ours.append(t)

        def close(self):
            pass

    QApplication.instance() or QApplication([])
    window = MainWindow(CameraSettings())
    engine = AudioEngine(sounds_dir=tmp_path / "none")
    mic = FakeMic()
    app = Application(AppConfig(), window, audio=engine, mic=mic)
    rig = Rig()
    lit = []
    for i, offsets in enumerate(finger_tap(FingerId.RIGHT_INDEX)):
        out = rig.frame(offsets)
        if i == 12:
            mic.queue.append(TapOnset(out.t, 20.0))
        app._on_presses(out)  # what the pipeline calls on the tracking thread
        lit.append(window.piano_view.pressed_source())
    played = [c for c in engine.mixer.pending if c[0] == "on"]
    assert len(played) == 1 and len(mic.ours) == 1  # one note, and the microphone was told about it
    assert any(lit) and lit[-1] == {}  # the key lit while sounding, off after the finger lifted
    assert [c[0] for c in engine.mixer.pending] == ["on", "off"]
    app.shutdown()
    window.close()
