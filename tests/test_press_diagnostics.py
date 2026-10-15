"""Tracking loss, missing data, stuck PRESSED states and one-to-one event counting.

The two stuck-state tests were ``xfail(strict=True)`` until the detector was
fixed (baseline recovery, relearn after loss, per-hand common mode); they now
pass. The classifier of stuck spans is tested on ``PressConfig.legacy()``,
which reproduces the old defect.
"""

import numpy as np
import pytest  # noqa: F401

from handpiano.interaction.finger_presses import PressEngine
from handpiano.interaction.press_detector import FingerPressMachine, PressConfig, PressInput, PressPhase
from handpiano.tracking.landmarks import PLAYING_FINGERS
from handpiano.interaction.trace import match_cues, pressed_spans, replay, summarize
from handpiano.tracking.landmarks import FingerId, Handedness
from tests.test_press_detector import C4, CFG, DT, LAYOUT, Rig, finger_tap, run, tap
from tests.test_trace import _synthetic_rows, _tap_wave

F = FingerId.RIGHT_INDEX


def feed(machine, values, start=0.0, tracked=None):
    events = []
    for i, v in enumerate(values):
        ok = True if tracked is None else tracked[i]
        events += machine.update(PressInput(start + i * DT, v, C4, ok and v is not None))
    return events


# --- tracking loss and missing data ------------------------------------------------------


def test_hand_leaving_the_camera_releases_a_held_press_at_frame_level():
    rig = Rig()
    events, _ = rig.run(finger_tap(FingerId.RIGHT_INDEX)[:16])  # right index is down
    assert [e.kind for e in events] == ["press"]
    out = rig.frame({}, hands=(Handedness.LEFT,))  # the right hand disappears
    released = [e for e in out.events if e.kind == "release"]
    assert [(e.finger, e.reason) for e in released] == [(FingerId.RIGHT_INDEX, "tracking_lost")]
    assert out.fingers[FingerId.RIGHT_INDEX].phase is PressPhase.IDLE


def test_missing_value_is_not_zero():
    m_missing = FingerPressMachine(F, CFG)
    m_zero = FingerPressMachine(F, CFG)
    pressed = tap(hold=30, after=0)[:18]
    feed(m_missing, pressed)
    feed(m_zero, pressed)
    assert m_missing.phase is PressPhase.PRESSED and m_zero.phase is PressPhase.PRESSED
    nxt = len(pressed) * DT  # the very next frame (no time gap)
    lost = feed(m_missing, [None], start=nxt)  # no data: release now, never hold a key on absent data
    zero = feed(m_zero, [0.0, 0.0], start=nxt)  # a real 0.0 is data: the finger came back up
    assert [(e.kind, e.reason) for e in lost] == [("release", "tracking_lost")]
    assert [(e.kind, e.reason) for e in zero] == [("release", "ok")]


def test_recovery_after_loss_needs_fresh_data_before_pressing():
    m = FingerPressMachine(F, CFG)
    feed(m, [0.0] * 10)
    feed(m, [None] * 5, start=10 * DT)
    # Reacquired with the finger already down: the jump happened during the gap, so there is
    # no velocity across it and nothing fires.
    events = feed(m, [0.3] * 6, start=16 * DT)
    assert [e for e in events if e.kind == "press"] == []
    # A tap before the new reference has been validated by enough still frames is not trusted.
    events = feed(m, tap(rest=1), start=22 * DT)
    assert [e for e in events if e.kind == "press"] == []
    # After a few still frames the reference is valid again and a real tap is detected.
    events = feed(m, tap(rest=10), start=60 * DT)
    assert [e.kind for e in events if e.kind in ("press", "release")] == ["press", "release"]


def test_a_single_missing_frame_releases_and_never_leaves_a_stale_press():
    events, m = run(tap(hold=10, after=0)[:16] + [None] + [0.3] * 20,
                    tracked=[True] * 16 + [False] + [True] * 20)
    assert [(e.kind, e.reason) for e in events] == [("press", "ok"), ("release", "tracking_lost")]
    assert m.phase is PressPhase.IDLE  # back with the finger still down, but no new press without a new gesture


# --- stuck states (production defects demonstrated on traza_v2.csv) ---------------------------


def _hand_inputs(t, values, tracked=True):
    """PressInputs for the four right-hand fingers with the given signal values."""
    fingers = [f for f in PLAYING_FINGERS if f.hand is Handedness.RIGHT]
    return {f: PressInput(t, v, C4, tracked and v is not None) for f, v in zip(fingers, values, strict=True)}


def test_posture_jump_does_not_leave_a_finger_pressed_for_seconds():
    """Whole-hand posture change (seen in traza_v2: 3–4 fingers of a hand jump together and stay):
    no finger may stay PRESSED. (A single finger that jumps and stays is a sustained note, see below.)"""
    engine = PressEngine(PressConfig())
    jump = [0.0] * 20 + [0.25, 0.5, 0.75, 1.0] + [1.0] * (24 * 5)
    gains = (1.0, 0.8, 1.2, 0.9)  # fingers do not move equally
    events = []
    for i, v in enumerate(jump):
        evs, _ = engine.step(_hand_inputs(i * DT, [v * g for g in gains]))
        events += evs
    assert [e for e in events if e.kind == "press"] == []
    assert all(m.phase is PressPhase.IDLE for m in engine.machines.values())


def test_posture_jump_used_to_leave_fingers_pressed_with_the_legacy_detector():
    engine = PressEngine(PressConfig.legacy())
    jump = [0.0] * 20 + [0.25, 0.5, 0.75, 1.0] + [1.0] * (24 * 5)
    for i, v in enumerate(jump):
        engine.step(_hand_inputs(i * DT, [v] * 4))
    right = [m for f, m in engine.machines.items() if f.hand is Handedness.RIGHT]
    assert all(m.phase is PressPhase.PRESSED for m in right)


def test_sustained_note_stays_pressed():
    """One finger presses and holds still for 10 s while the others rest: a legitimate held note."""
    engine = PressEngine(PressConfig())
    events = []
    for i, v in enumerate([0.0] * 20 + [0.1, 0.2, 0.3] + [0.3] * (24 * 10)):
        evs, _ = engine.step(_hand_inputs(i * DT, [v, 0.0, 0.0, 0.0]))
        events += evs
    assert [(e.kind, e.finger) for e in events] == [("press", FingerId.RIGHT_INDEX)]
    assert engine.machines[FingerId.RIGHT_INDEX].phase is PressPhase.PRESSED


def test_posture_change_during_rest_is_learnt_and_taps_still_work():
    """The hand settles into a new posture (slowly enough not to press); later taps are detected."""
    engine = PressEngine(PressConfig())
    events = []
    t = 0
    seq = [0.0] * 20 + list(np.linspace(0.0, 0.6, 48)) + [0.6] * 48  # 2 s drift, then still
    for v in seq:
        evs, _ = engine.step(_hand_inputs(t * DT, [v] * 4))
        events += evs
        t += 1
    for v in tap(rest=10):
        evs, _ = engine.step(_hand_inputs(t * DT, [0.6 + v, 0.6, 0.6, 0.6]))
        events += evs
        t += 1
    assert [(e.kind, e.finger) for e in events] == [("press", FingerId.RIGHT_INDEX), ("release", FingerId.RIGHT_INDEX)]


def test_repeated_taps_each_give_one_press_and_one_release():
    engine = PressEngine(PressConfig())
    events = []
    seq = []
    for _ in range(5):
        seq += tap(rest=8, after=4)
    for i, v in enumerate(seq):
        evs, _ = engine.step(_hand_inputs(i * DT, [v, 0.0, 0.0, 0.0]))
        events += evs
    kinds = [e.kind for e in events if e.finger is FingerId.RIGHT_INDEX]
    assert kinds == ["press", "release"] * 5
    assert all(e.finger is FingerId.RIGHT_INDEX for e in events)


def test_common_mode_keeps_a_two_finger_chord_of_one_hand():
    engine = PressEngine(PressConfig())
    events = []
    for i, v in enumerate(tap()):
        evs, _ = engine.step(_hand_inputs(i * DT, [v, v, 0.0, 0.0]))  # index + middle together
        events += evs
    presses = sorted(e.finger.name for e in events if e.kind == "press")
    assert presses == ["RIGHT_INDEX", "RIGHT_MIDDLE"]


def test_common_mode_is_per_hand():
    """A tap of the right index must not be cancelled by the LEFT hand moving."""
    engine = PressEngine(PressConfig())
    left = [f for f in PLAYING_FINGERS if f.hand is Handedness.LEFT]
    events = []
    for i, v in enumerate(tap()):
        inputs = _hand_inputs(i * DT, [v, 0.0, 0.0, 0.0])
        inputs.update({f: PressInput(i * DT, 0.5 * i / len(tap()), C4, True) for f in left})
        evs, _ = engine.step(inputs)
        events += evs
    assert [e.finger for e in events if e.kind == "press"] == [FingerId.RIGHT_INDEX]


def test_stale_baseline_after_tracking_loss_recovers():
    # Hand returns in another posture (0.8 above the old baseline) and stays still for 4 s,
    # then a small fast movement on top of it, then still again — all frames contiguous.
    values = [0.0] * 20 + [None] * 10 + [0.8] * (24 * 4) + [1.1, 1.1] + [0.8] * 48
    tracked = [v is not None for v in values]
    events, m = run(values, tracked=tracked)
    assert m.phase is PressPhase.IDLE  # baseline re-learnt, no stuck press


def test_max_press_duration_is_off_by_default_and_cuts_notes_when_enabled():
    hold = [0.0] * 20 + [0.1, 0.2, 0.3] + [0.3] * (24 * 6)
    events, m = run(hold)
    assert m.phase is PressPhase.PRESSED  # default: no time limit, sustained notes survive
    events, m = run(hold, config=PressConfig(max_press_s=2.0))
    assert [(e.kind, e.reason) for e in events] == [("press", "ok"), ("release", "max_duration")]
    assert m.phase is PressPhase.IDLE


def test_pressed_spans_classify_both_stuck_mechanisms():
    jump = _synthetic_rows([("taps_RIGHT_INDEX", 6.0, lambda f, t: (0.0 if t < 1.0 else min(1.0, (t - 1.0) * 6))
                             if f == "RIGHT_INDEX" else 0.0)])
    legacy = PressConfig.legacy()  # the classifier is checked on the detector that had the defect
    spans = [s for s in pressed_spans(jump, legacy, LAYOUT) if s.finger == "RIGHT_INDEX"]
    # Only the instructed finger went down and stayed: no other finger moved, so this is
    # NOT a posture change (the old label "posture_jump" was a default, not evidence).
    assert len(spans) == 1 and spans[0].cause == "instructed_hold" and spans[0].ended_by == "trace_end"
    assert spans[0].others_rising == 0 and spans[0].instructed
    assert spans[0].duration > 4 and spans[0].min_depth >= 0.5

    def stale(f, t):
        if f != "RIGHT_INDEX":
            return 0.0
        if t < 1.0:
            return 0.0
        if t < 1.5:
            return None  # tracking lost
        if 3.0 <= t < 3.1:
            return 1.1  # small fast movement on top of the new posture
        return 0.8
    rows = _synthetic_rows([("rest", 6.0, stale)])
    spans = [s for s in pressed_spans(rows, legacy, LAYOUT) if s.finger == "RIGHT_INDEX"]
    assert len(spans) == 1 and spans[0].cause == "stale_baseline_after_gap"
    # The fixed detector does not get stuck in the same scenario.
    assert all(s.duration < 1.0 for s in pressed_spans(rows, PressConfig(), LAYOUT))


# --- one-to-one event counting --------------------------------------------------------------


def _press_rows(press_offsets, cue_every=1.5, seconds=8.0):
    """Right index taps at cue + offset for each listed offset (one entry per cue, None = no tap)."""
    def fn(f, t):
        if f != "RIGHT_INDEX":
            return 0.0
        for k, off in enumerate(press_offsets):
            if off is None:
                continue
            start = 1.0 + k * cue_every + off - 0.1
            if start <= t < start + 0.3:
                return 0.4
        return 0.0
    return _synthetic_rows([("taps_RIGHT_INDEX", seconds, fn)], cue_every=cue_every, cue_finger="RIGHT_INDEX")


def test_each_press_counts_once_and_each_cue_at_most_once():
    rows = _press_rows([0.0, 0.0, None, 0.0])
    r = {x.finger: x for x in match_cues(rows, CFG, LAYOUT)}["RIGHT_INDEX"]
    presses = sum(res.presses.get("RIGHT_INDEX", 0) for res in replay(rows, CFG, LAYOUT))
    assert r.hits + r.extra == presses  # every press is either a hit or extra, never both or twice
    assert r.hits == 3 and r.missed == r.cues - 3


def test_anticipating_the_cue_still_counts():
    rows = _press_rows([-0.6, -0.6, -0.6])  # pressing 0.6 s early (anticipation seen in traza_v2)
    r = {x.finger: x for x in match_cues(rows, CFG, LAYOUT)}["RIGHT_INDEX"]
    assert r.hits == 3 and all(-0.75 <= o <= 0 for o in r.offsets)


def test_two_presses_near_one_cue_give_one_hit_and_one_extra():
    def fn(f, t):
        if f != "RIGHT_INDEX":
            return 0.0
        return 0.4 if (0.95 <= t < 1.2 or 1.45 <= t < 1.7) else 0.0  # two taps around the first cue (t=1.0)
    rows = _synthetic_rows([("taps_RIGHT_INDEX", 3.0, fn)], cue_every=1.5, cue_finger="RIGHT_INDEX")
    r = {x.finger: x for x in match_cues(rows, PressConfig(refractory_s=0.0), LAYOUT)}["RIGHT_INDEX"]
    assert r.hits == 1 and r.extra == 1


def test_summary_press_totals_equal_replay_totals():
    rows = _synthetic_rows([("taps_RIGHT_INDEX", 8.0, lambda f, t: _tap_wave()(t) if f == "RIGHT_INDEX" else 0.0)],
                           cue_every=1.5, cue_finger="RIGHT_INDEX")
    from_summary = sum(s.presses for s in summarize(rows, CFG, LAYOUT))
    from_replay = sum(sum(r.presses.values()) for r in replay(rows, CFG, LAYOUT))
    assert from_summary == from_replay > 0


def test_span_cause_requires_evidence_of_a_whole_hand_movement():
    def hand_jump(f, t):  # all four right fingers jump together and stay
        return (0.0 if t < 1.0 else min(1.0, (t - 1.0) * 6)) if f.startswith("RIGHT") else 0.0
    spans = pressed_spans(_synthetic_rows([("rest", 5.0, hand_jump)]), PressConfig.legacy(), LAYOUT)
    assert spans and all(s.cause == "hand_comovement" and s.others_rising >= 2 for s in spans)

    def neighbour(f, t):  # during middle taps, the (not instructed) index alone goes down and stays
        if f == "RIGHT_INDEX":
            return 0.0 if t < 1.0 else min(0.8, (t - 1.0) * 6)
        return 0.0
    spans = [s for s in pressed_spans(_synthetic_rows([("taps_RIGHT_MIDDLE", 5.0, neighbour)]), PressConfig(), LAYOUT)
             if s.finger == "RIGHT_INDEX"]
    assert len(spans) == 1 and spans[0].cause == "uninstructed_single" and not spans[0].instructed


# --- idle_min common mode (alternative, evaluated on traza_v2/v3) --------------------------------

IDLE_MIN = PressConfig(common_mode="idle_min")


def _engine_run(config, sequences):
    """sequences: per right-hand finger (index, middle, ring, pinky), a list of values."""
    engine = PressEngine(config)
    events = []
    for i in range(len(sequences[0])):
        evs, _ = engine.step(_hand_inputs(i * DT, [seq[i] for seq in sequences]))
        events += evs
    return events, engine


def test_idle_min_keeps_a_three_finger_chord_that_hand_low_suppresses():
    chord = tap()
    rest = [0.0] * len(chord)
    seq = [chord, chord, chord, rest]  # index + middle + ring together, pinky still
    presses_low = sorted(e.finger.name for e in _engine_run(PressConfig(), seq)[0] if e.kind == "press")
    presses_idle = sorted(e.finger.name for e in _engine_run(IDLE_MIN, seq)[0] if e.kind == "press")
    assert presses_low == []  # documented limit of hand_low
    assert presses_idle == ["RIGHT_INDEX", "RIGHT_MIDDLE", "RIGHT_RING"]


def test_idle_min_ignores_held_neighbours_when_another_finger_taps():
    """traza_v3: index and middle were held down (pressed); the ring tap was cancelled by hand_low."""
    held = [0.0] * 10 + [0.1, 0.3, 0.6, 0.8] + [0.8] * 60
    ring = [0.0] * 40 + tap(rest=0)
    ring += [0.0] * (len(held) - len(ring))
    rest = [0.0] * len(held)
    seq = [held, held, ring, rest]
    low = [e.finger for e in _engine_run(PressConfig(), seq)[0] if e.kind == "press"]
    idle = [e.finger for e in _engine_run(IDLE_MIN, seq)[0] if e.kind == "press"]
    assert FingerId.RIGHT_RING not in low  # the defect, reproduced
    assert FingerId.RIGHT_RING in idle


def test_idle_min_still_suppresses_closing_the_hand():
    jump = [0.0] * 20 + [0.25, 0.5, 0.75, 1.0] + [1.0] * 60
    seq = [[v * g for v in jump] for g in (1.0, 0.85, 1.2, 0.9)]
    events, engine = _engine_run(IDLE_MIN, seq)
    assert [e for e in events if e.kind == "press"] == []
    assert all(m.phase is PressPhase.IDLE for f, m in engine.machines.items() if f.hand is Handedness.RIGHT)


# --- cue diagnostics and offsets ---------------------------------------------------------------

from handpiano.interaction.trace import MISS_REASONS, cue_diagnostics, press_offsets  # noqa: E402


def test_cue_diagnostics_hit_and_no_visible_movement():
    tap_wave = _tap_wave(period=1.5, start=1.0)
    rows = _synthetic_rows([("taps_RIGHT_INDEX", 8.0, lambda f, t: tap_wave(t) if f == "RIGHT_INDEX" else 0.0)],
                           cue_every=1.5, cue_finger="RIGHT_INDEX")
    diag = cue_diagnostics(rows, CFG, LAYOUT)
    assert diag and all(d.hit and d.reason is None for d in diag)

    still = _synthetic_rows([("taps_RIGHT_INDEX", 8.0, lambda f, t: 0.0)], cue_every=1.5, cue_finger="RIGHT_INDEX")
    diag = cue_diagnostics(still, CFG, LAYOUT)
    assert diag and all(not d.hit and d.reason == "no_visible_movement" for d in diag)
    assert set(MISS_REASONS) >= {d.reason for d in diag}


def test_cue_diagnostics_reports_common_mode_removal():
    def fn(f, t):
        if f in ("RIGHT_INDEX", "RIGHT_MIDDLE"):
            return 0.0 if t < 0.5 else min(0.8, (t - 0.5) * 6)  # two neighbours held down
        if f == "RIGHT_RING":
            phase = (t - 1.0) % 1.5
            return 0.6 if t >= 1.0 and 0.05 <= phase < 0.35 else 0.0
        return 0.0
    rows = _synthetic_rows([("taps_RIGHT_RING", 6.0, fn)], cue_every=1.5, cue_finger="RIGHT_RING")
    reasons = {d.reason for d in cue_diagnostics(rows, CFG, LAYOUT)}
    assert "common_mode_removed" in reasons
    assert all(d.hit for d in cue_diagnostics(rows, IDLE_MIN, LAYOUT))


def test_press_offsets_are_signed_and_window_free():
    def fn(f, t):  # taps 0.5 s after each cue (cues at 1.0, 2.5, 4.0, ...)
        if f != "RIGHT_INDEX" or t < 1.4:
            return 0.0
        phase = (t - 1.4) % 1.5
        return 0.4 if phase < 0.3 else 0.0
    rows = _synthetic_rows([("taps_RIGHT_INDEX", 7.0, fn)], cue_every=1.5, cue_finger="RIGHT_INDEX")
    offs = press_offsets(rows, CFG, LAYOUT)["RIGHT_INDEX"]
    assert offs and all(0.3 < o < 0.7 for o in offs)
