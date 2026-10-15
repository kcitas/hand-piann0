"""Detected presses → notes, end to end with the REAL detector (Rig) and, where it matters, the real engine.

One note per press event (never per frame), note_off on release, chords, tracking loss,
instrument switching, audio failures, and the re-strike of a key held by another finger
(the cause of silent lit keys measured on teclado_1–4).
"""

import pytest

from handpiano.app.press_sound import (
    PressSound,
    describe,
    outcomes_by_tag,
    summary,
    velocity_from_press,
)
from handpiano.audio.engine import AudioEngine
from handpiano.tracking.landmarks import FingerId, Handedness
from tests.audio_fakes import make_all, render
from tests.test_press_detector import Rig, finger_tap


class Sink:
    def __init__(self, fail_on=None):
        self.calls = []
        self.fail_on = fail_on

    def note_on(self, midi, velocity=100, tag=None):
        if self.fail_on == "on":
            raise RuntimeError("audio device lost")
        self.calls.append(("on", midi, velocity))
        return True

    def note_off(self, midi, tag=None):
        if self.fail_on == "off":
            raise RuntimeError("audio device lost")
        self.calls.append(("off", midi))
        return True


def kinds(sink):
    return [c[0] for c in sink.calls]


def play(frames, sound, rig=None, hands=None):
    rig = rig or Rig()
    pressed_frames = 0
    for offsets in frames:
        out = rig.frame(offsets) if hands is None else rig.frame(offsets, hands=hands)
        sound.on_frame(out.events, out.pressed_keys)
        pressed_frames += bool(out.pressed_keys)
    return pressed_frames


def test_one_press_is_exactly_one_note_on_and_holding_does_not_repeat():
    sink = Sink()
    pressed_frames = play(finger_tap(FingerId.RIGHT_INDEX), PressSound(sink))
    assert pressed_frames >= 5  # the finger stayed down for several frames…
    assert kinds(sink) == ["on", "off"]  # …one note_on, and note_off when it lifted
    assert sink.calls[0][1] == sink.calls[1][1]


def test_two_taps_two_notes_and_a_two_finger_chord():
    sink = Sink()
    play(finger_tap(FingerId.RIGHT_INDEX) + finger_tap(FingerId.RIGHT_INDEX), PressSound(sink))
    assert kinds(sink) == ["on", "off", "on", "off"]
    sink = Sink()
    chord = [{**a, **b} for a, b in zip(finger_tap(FingerId.LEFT_INDEX), finger_tap(FingerId.RIGHT_INDEX),
                                         strict=True)]
    play(chord, PressSound(sink))
    ons = [c for c in sink.calls if c[0] == "on"]
    assert len(ons) == 2 and ons[0][1] != ons[1][1]
    assert kinds(sink).count("off") == 2


def test_losing_tracking_releases_the_note():
    sink = Sink()
    sound = PressSound(sink)
    rig = Rig()
    tap = finger_tap(FingerId.RIGHT_INDEX)
    play(tap[:14], sound, rig)  # finger down
    assert kinds(sink) == ["on"]
    play([{}] * 3, sound, rig, hands=(Handedness.LEFT,))  # the right hand disappears
    assert kinds(sink) == ["on", "off"] and sound.sounding == []
    record = sound.records[-1]
    assert record.released_t is not None and "tracking_lost" in record.release_reason


class _Key:
    def __init__(self, midi):
        self.midi = midi


class _Event:
    def __init__(self, kind, finger, midi, reason="ok"):
        self.kind, self.finger, self.key, self.reason, self.t, self.velocity = kind, finger, _Key(midi), reason, 0.0, 1.0


A, B = FingerId.RIGHT_INDEX, FingerId.RIGHT_MIDDLE


def test_a_key_held_by_another_finger_is_struck_again():
    """teclado_1/2: a neighbour finger stuck in PRESSED on key 77; the right finger taps 77 → it must sound."""
    sink = Sink()
    sound = PressSound(sink)
    sound.on_frame([_Event("press", A, 77)], {77: (A,)})  # neighbour, stuck down
    sound.on_frame([_Event("press", B, 77)], {77: (A, B)})  # the intended tap on the same key
    assert kinds(sink) == ["on", "on"] and sound.records[-1].restrike
    sound.on_frame([_Event("release", B, 77)], {77: (A,)})
    assert kinds(sink) == ["on", "on"]  # A still holds the key: it keeps sounding (and lit)
    assert sound.records[-1].note_off.startswith("no enviada")
    sound.on_frame([_Event("release", A, 77)], {})
    assert kinds(sink) == ["on", "on", "off"]


def test_a_lost_release_event_cannot_leave_a_note_stuck():
    sink = Sink()
    sound = PressSound(sink)
    rig = Rig()
    for offsets in finger_tap(FingerId.RIGHT_INDEX):
        out = rig.frame(offsets)
        events = [e for e in out.events if e.kind != "release"]  # the release event never arrives
        sound.on_frame(events, out.pressed_keys)
    assert kinds(sink) == ["on", "off"] and sound.sounding == []
    assert sound.records[-1].release_reason.startswith("reconciliación")


@pytest.mark.parametrize("failing", ["on", "off"])
def test_an_audio_failure_keeps_key_state_coherent_and_does_not_raise(failing):
    sink = Sink(fail_on=failing)
    sound = PressSound(sink)
    play(finger_tap(FingerId.RIGHT_INDEX), sound)  # must not raise (the pipeline would disable the listener)
    assert sound.sounding == [] and sound.errors >= 1
    record = sound.records[-1]
    assert record.released_t is not None
    assert (record.note_on if failing == "on" else record.note_off).startswith("error: RuntimeError")
    sink.fail_on = None  # the device comes back: the next press sounds normally
    play(finger_tap(FingerId.RIGHT_INDEX), sound)
    assert kinds(sink)[-2:] == ["on", "off"]


def test_disabled_router_is_silent_and_stops_sounding_notes():
    sink = Sink()
    sound = PressSound(sink)
    rig = Rig()
    tap = finger_tap(FingerId.RIGHT_INDEX)
    play(tap[:14], sound, rig)
    sound.set_enabled(False)  # e.g. switching to the rhythm game
    assert kinds(sink) == ["on", "off"]
    play(tap, sound, rig)
    assert len(sink.calls) == 2


def test_replayed_frames_without_events_add_nothing():
    sink = Sink()
    sound = PressSound(sink)
    rig = Rig()
    for offsets in finger_tap(FingerId.RIGHT_INDEX):
        out = rig.frame(offsets)
        sound.on_frame(out.events, out.pressed_keys)
        sound.on_frame([], out.pressed_keys)  # same state again, no new events
    assert kinds(sink) == ["on", "off"]


def test_velocity_mapping_is_bounded():
    assert velocity_from_press(None) == 90 and velocity_from_press(float("nan")) == 90
    assert velocity_from_press(0.0) == 50
    assert velocity_from_press(100.0) == 120 and velocity_from_press(-5.0) == 40
    assert velocity_from_press(1.0) < velocity_from_press(2.0)


# --- with the real engine: diagnostics and instrument switching ---------------------------------


@pytest.fixture(scope="module")
def sounds(tmp_path_factory):
    return make_all(tmp_path_factory.mktemp("press_sounds"), seconds=0.4)


def test_each_press_is_traced_to_the_audio_thread(sounds):
    engine = AudioEngine(sounds_dir=sounds)
    engine.set_instrument("upright")
    assert engine.wait_ready()
    sound = PressSound(engine)
    rig = Rig()
    for offsets in finger_tap(FingerId.RIGHT_INDEX):
        out = rig.frame(offsets)
        sound.on_frame(out.events, out.pressed_keys)
        engine.mixer.render(256)  # the audio thread runs between frames
    outcomes = outcomes_by_tag(engine.mixer.outcomes)
    record = sound.records[-1]
    assert outcomes[(record.id, "on")].result == "voz iniciada" and outcomes[(record.id, "on")].instrument == "upright"
    assert outcomes[(record.id, "off")].result == "voces liberadas: 1"
    line = describe(record, outcomes)
    assert line.startswith(f"[nota #{record.id}] RIGHT_INDEX → MIDI") and "voz iniciada (upright" in line
    assert summary(sound.records, outcomes) == {"pulsaciones": 1, "note_on encoladas": 1, "voz iniciada": 1,
                                                "sin banco": 0, "errores": 0, "liberadas": 1,
                                                "note_off encoladas": 1}


def test_switching_instrument_loses_no_press(sounds):
    engine = AudioEngine(sounds_dir=sounds)
    engine.set_instrument("grand")
    assert engine.wait_ready()
    sound = PressSound(engine)
    rig = Rig()
    taps = finger_tap(FingerId.RIGHT_INDEX) * 3
    for i, offsets in enumerate(taps):
        if i == len(taps) // 3:
            engine.set_instrument("electric")  # loads in the background while presses keep coming
        out = rig.frame(offsets)
        sound.on_frame(out.events, out.pressed_keys)
        engine.mixer.render(256)
    engine.wait_ready()
    render(engine, 0.01)
    outcomes = outcomes_by_tag(engine.mixer.outcomes)
    results = [outcomes[(r.id, "on")].result for r in sound.records]
    assert len(results) == 3 and results == ["voz iniciada"] * 3
    assert outcomes[(sound.records[-1].id, "on")].instrument in ("grand", "electric")


def test_no_bank_is_reported_not_claimed_as_played(tmp_path):
    engine = AudioEngine(sounds_dir=tmp_path / "missing")
    engine.set_instrument("grand")
    engine.wait_ready(timeout=5)
    sound = PressSound(engine)
    play(finger_tap(FingerId.RIGHT_INDEX), sound)
    engine.mixer.render(256)
    outcomes = outcomes_by_tag(engine.mixer.outcomes)
    record = sound.records[-1]
    assert record.note_on == "encolada"  # accepted by the queue…
    assert outcomes[(record.id, "on")].result == "sin banco: nota descartada"  # …but nothing was played
    assert summary(sound.records, outcomes)["voz iniciada"] == 0


def test_finger_note_mode_never_plays_inaudibly_soft_notes():
    from handpiano.app.press_sound import FINGER_NOTE_VELOCITY_FLOOR, velocity_from_press

    assert velocity_from_press(0.0) == 50  # drawn piano: a slow press is a soft note
    assert velocity_from_press(0.0, FINGER_NOTE_VELOCITY_FLOOR) == 85  # finger notes: never below 85
    assert velocity_from_press(9.0, FINGER_NOTE_VELOCITY_FLOOR) == 120
    assert velocity_from_press(None, FINGER_NOTE_VELOCITY_FLOOR) == 90
