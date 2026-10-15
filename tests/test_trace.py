"""Press traces: recording rows from snapshots and replaying them offline with the same detector."""

import csv
import subprocess
import sys
from dataclasses import replace

import numpy as np

from handpiano.app.pipeline import FrameTimestamps, TrackingSnapshot
from handpiano.camera.frame import Frame
from handpiano.interaction.press_detector import PressConfig
from handpiano.interaction.trace import COLUMNS, TraceRecorder, load_trace, replay
from handpiano.tracking.landmarks import FingerId
from tests.test_press_detector import LAYOUT, Rig, finger_tap


def _record(tmp_path):
    rig = Rig()
    recorder = TraceRecorder()
    image = np.zeros((4, 4, 3), np.uint8)
    seq = 0

    def feed(step, frames):
        nonlocal seq
        recorder.step = step
        for offsets in frames:
            t_ns = int(rig.t * 1e9)
            out = rig.frame(offsets)
            seq += 1
            snap = TrackingSnapshot(seq, Frame(seq, image, t_ns, t_ns), _last_state[0], FrameTimestamps(*[0] * 7),
                                    _last_mapping[0], out)
            recorder.on_frame(snap)

    # Capture the state/mapping the rig computed for each frame.
    _last_state, _last_mapping = [None], [None]
    original = rig.mapper.map

    def spy(state):
        _last_state[0] = state
        _last_mapping[0] = original(state)
        return _last_mapping[0]

    rig.mapper.map = spy
    feed("rest", [{}] * 40)
    feed("right_index_taps", finger_tap(FingerId.RIGHT_INDEX) + finger_tap(FingerId.RIGHT_MIDDLE, rest=4))
    path = tmp_path / "trace.csv"
    recorder.write(path)
    return path, recorder


def test_recorder_writes_one_row_per_finger_per_frame_numbers_only(tmp_path):
    path, recorder = _record(tmp_path)
    rows = list(csv.DictReader(path.open()))
    assert tuple(rows[0].keys()) == COLUMNS
    assert len(rows) == recorder.row_count and len(rows) % 8 == 0
    assert {r["step"] for r in rows} == {"rest", "right_index_taps"}
    assert any(r["events"].startswith("press:ok:") for r in rows if r["finger"] == "RIGHT_INDEX")
    assert all(r["tip_y"] for r in rows if r["tracked"] == "True")


def test_replay_separates_intended_from_unintended_presses(tmp_path):
    path, _ = _record(tmp_path)
    results = {r.step: r for r in replay(load_trace(path), PressConfig(), LAYOUT)}
    assert results["rest"].intended == 0 and results["rest"].unintended == 0
    taps = results["right_index_taps"]
    assert taps.presses.get("RIGHT_INDEX") == 1
    assert taps.intended == 1 and taps.unintended == 1  # the middle finger's tap is not the instructed one


def test_replay_with_another_config_changes_the_outcome(tmp_path):
    path, _ = _record(tmp_path)
    strict = replace(PressConfig(), press_on=5.0, release_off=2.0)  # unreachable threshold
    results = replay(load_trace(path), strict, LAYOUT)
    assert sum(r.intended + r.unintended for r in results) == 0


def test_analyze_script_runs_on_a_trace(tmp_path):
    path, _ = _record(tmp_path)
    out = subprocess.run([sys.executable, "scripts/analyze_trace.py", str(path), "--sweep"],
                         capture_output=True, text=True, check=True)
    assert "Per step × finger" in out.stdout and "rejected" in out.stdout
    assert "Sweep" in out.stdout and "release_off=0.06" in out.stdout and "tip_z" in out.stdout
    assert "RIGHT_INDEX" in out.stdout


# --- diagnostics (summary, common mode, cues) ------------------------------------------------

from handpiano.interaction.trace import (  # noqa: E402
    V1_STEPS,
    V2_STEPS,
    match_cues,
    remove_common_mode,
    summarize,
)
from handpiano.tracking.landmarks import PLAYING_FINGERS  # noqa: E402


def _synthetic_rows(step_signals, fps=24, cue_every=None, cue_finger=None):
    """step_signals: list of (step, seconds, fn(finger_name, t) -> tip_y or None)."""
    rows, t = [], 0.0
    next_cue = None
    for step, seconds, fn in step_signals:
        start = t
        next_cue = start + 1.0 if cue_every else None
        while t < start + seconds:
            cue = ""
            if next_cue is not None and t >= next_cue:
                cue, next_cue = cue_finger, next_cue + cue_every
            for f in PLAYING_FINGERS:
                v = fn(f.name, t - start)
                rows.append({
                    "t": f"{t:.4f}", "step": step, "finger": f.name, "tracked": str(v is not None),
                    "discontinuity": "False", "hand_speed": "0.0",
                    "tip_y": "" if v is None else f"{v:.5f}", "flex": "" if v is None else f"{v:.5f}",
                    "tip_z": "" if v is None else "0.0", "piano_x": "", "piano_y": "", "raw_key": "",
                    "stable_key": "C4", "phase": "IDLE", "depth": "", "velocity": "", "events": "", "cue": cue,
                })
            t += 1 / fps
    return rows


def _tap_wave(period=1.5, start=1.0, depth=0.4):
    def fn(t):
        if t < start:
            return 0.0
        phase = (t - start) % period
        return depth if 0.05 <= phase < 0.35 else 0.0
    return fn


def test_v2_protocol_has_cued_blocks_for_every_finger_and_no_press_steps():
    names = [s.name for s in V2_STEPS]
    assert all(f"taps_{f.name}" in names for f in PLAYING_FINGERS)
    assert {"rest", "posture", "slide", "chord"} <= set(names)
    for s in V2_STEPS:
        if s.name.startswith("taps_"):
            assert len(s.cue_times()) == 8
    assert len(next(s for s in V2_STEPS if s.name == "chord").cue_times()) == 6
    assert 120 < sum(s.duration_s for s in V2_STEPS) < 200  # under ~3 minutes
    assert [s.name for s in V1_STEPS][:2] == ["prepare", "rest"]  # old traces stay analyzable


def test_common_mode_removal_cancels_whole_hand_motion_but_keeps_a_single_finger_tap():
    import numpy as np

    n = 200
    common = np.sin(np.linspace(0, 6, n))  # all four fingers move together (posture)
    tap = np.zeros(n)
    tap[100:110] = 1.0
    exc = {"LEFT_INDEX": common + tap, "LEFT_MIDDLE": common.copy(), "LEFT_RING": common.copy(),
           "LEFT_PINKY": common.copy()}
    out = remove_common_mode(exc)
    assert np.nanmax(np.abs(out["LEFT_MIDDLE"])) < 1e-9
    assert np.nanmax(out["LEFT_INDEX"]) > 0.99


def test_summary_distinguishes_missing_data_from_zero():
    rows = _synthetic_rows([
        ("rest", 3.0, lambda f, t: 0.0),  # tracked, perfectly still: zeros, not missing
        ("right_index_taps", 3.0, lambda f, t: None if f.startswith("LEFT") else 0.0),  # left hand not seen
    ])
    summary = {(s.step, s.finger): s for s in summarize(rows, PressConfig(), LAYOUT)}
    still = summary[("rest", "LEFT_INDEX")]
    assert still.valid == still.frames and still.missing == 0
    assert still.signal_max["tip_y"] == 0.0 and still.signal_range["tip_y"] == 0.0
    absent = summary[("right_index_taps", "LEFT_INDEX")]
    assert absent.valid == 0 and absent.missing == absent.frames
    assert absent.signal_max["tip_y"] is None and absent.signal_range["tip_y"] is None
    assert absent.excursion_p95["tip_y"] is None
    assert summary[("right_index_taps", "RIGHT_INDEX")].expected


def test_summary_counts_presses_transition_presses_and_stuck_spans():
    tap = _tap_wave()
    rows = _synthetic_rows([
        ("rest", 3.0, lambda f, t: 0.0),
        # Right index taps; the left index goes down at the very start and stays down (stuck).
        ("right_index_taps", 6.0, lambda f, t: tap(t) if f == "RIGHT_INDEX"
         else (min(0.5, t * 2) if f == "LEFT_INDEX" else 0.0)),
    ])
    s = {(x.step, x.finger): x for x in summarize(rows, PressConfig(), LAYOUT)}
    taps = s[("right_index_taps", "RIGHT_INDEX")]
    assert taps.presses >= 3
    stuck = s[("right_index_taps", "LEFT_INDEX")]
    assert stuck.presses == 1 and stuck.transition_presses == 1
    assert stuck.longest_pressed_s == 0.0  # never released within the trace: no completed span
    assert taps.longest_pressed_s > 0


def test_cue_matching_counts_hits_misses_and_extra_presses():
    tap = _tap_wave(period=1.5, start=1.0)
    rows = _synthetic_rows(
        [("taps_RIGHT_INDEX", 8.0, lambda f, t: tap(t) if f == "RIGHT_INDEX" else 0.0)],
        cue_every=1.5, cue_finger="RIGHT_INDEX",
    )
    results = {r.finger: r for r in match_cues(rows, PressConfig(), LAYOUT)}
    r = results["RIGHT_INDEX"]
    assert r.cues >= 4 and r.hits == r.cues and r.missed == 0 and r.extra == 0
    assert results["LEFT_INDEX"].cues == 0 and results["LEFT_INDEX"].hit_rate is None

    # Same cues, but the finger never moves: every cue is missed, no extra presses.
    still = _synthetic_rows([("taps_RIGHT_INDEX", 8.0, lambda f, t: 0.0)], cue_every=1.5, cue_finger="RIGHT_INDEX")
    r2 = {r.finger: r for r in match_cues(still, PressConfig(), LAYOUT)}["RIGHT_INDEX"]
    assert r2.hits == 0 and r2.missed == r2.cues


def test_v1_trace_without_cue_column_has_no_cue_results(tmp_path):
    path, _ = _record(tmp_path)
    rows = load_trace(path)
    for r in rows:
        r.pop("cue", None)
    assert match_cues(rows, PressConfig(), LAYOUT) == []


def test_recorder_marks_a_cue_on_exactly_one_frame(tmp_path):
    rig = Rig()
    recorder = TraceRecorder()
    image = np.zeros((4, 4, 3), np.uint8)
    for i in range(5):
        if i == 2:
            recorder.mark_cue((FingerId.RIGHT_INDEX,))
        out = rig.frame({})
        snap = TrackingSnapshot(i + 1, Frame(i + 1, image, int(rig.t * 1e9), 0), None, FrameTimestamps(*[0] * 7),
                                None, out)
        recorder.on_frame(snap)
    path = tmp_path / "t.csv"
    recorder.write(path)
    cues = [r["cue"] for r in csv.DictReader(path.open())]
    assert cues.count("RIGHT_INDEX") == 8  # one frame × 8 finger rows


def test_compare_script_reports_legacy_and_fixed_detectors(tmp_path):
    path, _ = _record(tmp_path)
    out = subprocess.run([sys.executable, "scripts/compare_detectors.py", str(path)],
                         capture_output=True, text=True, check=True).stdout
    assert "previous (legacy)" in out and "all fixes (new default)" in out and "max PRESSED" in out


def test_replay_runs_the_production_engine_frame_by_frame(tmp_path):
    """Same trace, same config: replay() and the online detector that wrote the trace agree."""
    path, _ = _record(tmp_path)
    rows = load_trace(path)
    online = sum(1 for r in rows for e in r["events"].split("|") if e.startswith("press:"))
    offline = sum(sum(r.presses.values()) for r in replay(rows, PressConfig(), LAYOUT))
    assert online == offline > 0


def test_focus_protocol_targets_problem_fingers_with_unambiguous_cues():
    from handpiano.interaction.trace import FOCUS_STEPS, PROTOCOLS

    assert PROTOCOLS["focus"] is FOCUS_STEPS
    tap_steps = [s for s in FOCUS_STEPS if s.cue_every_s]
    assert [s.expected[0].name for s in tap_steps] == ["LEFT_INDEX", "LEFT_RING", "RIGHT_MIDDLE", "RIGHT_RING"]
    for s in tap_steps:
        assert len(s.cue_times()) == 6 and s.cue_every_s == 3.0  # matching window ±1.5 s: no overlap
    assert {"rest", "posture"} <= {s.name for s in FOCUS_STEPS}
    assert 90 <= sum(s.duration_s for s in FOCUS_STEPS) <= 120


def test_cli_accepts_trace_protocol():
    from handpiano.app.main import parse_args

    args = parse_args(["--record-trace", "x.csv", "--trace-protocol", "focus"])
    assert args.trace_protocol == "focus" and str(args.record_trace) == "x.csv"
    assert parse_args([]).trace_protocol == "v2"


def test_play_protocol_is_short_free_playing_with_every_finger():
    from handpiano.interaction.trace import PROTOCOLS
    from handpiano.tracking.landmarks import PLAYING_FINGERS

    steps = PROTOCOLS["play"]
    assert sum(s.duration_s for s in steps) == 35.0
    assert steps[-1].name == "play_free" and steps[-1].expected == PLAYING_FINGERS and not steps[-1].cue_times()


def test_finger_count_protocol_asks_five_taps_per_finger():
    from handpiano.interaction.trace import PROTOCOLS
    from handpiano.tracking.landmarks import PLAYING_FINGERS

    steps = PROTOCOLS["dedos"]
    taps = [s for s in steps if s.expected]
    assert [s.expected[0] for s in taps] == list(PLAYING_FINGERS)
    assert all(len(s.cue_times()) == 5 and s.cue_every_s == 2.0 for s in taps)
    assert 95 <= sum(s.duration_s for s in steps) <= 110


def test_index_protocol_asks_ten_taps_per_index_finger():
    from handpiano.interaction.trace import ALL_STEPS, PROTOCOLS
    from handpiano.tracking.landmarks import FingerId

    steps = PROTOCOLS["indices"]
    taps = [s for s in steps if s.expected]
    assert [s.expected for s in taps] == [(FingerId.LEFT_INDEX,), (FingerId.RIGHT_INDEX,)]
    assert all(len(s.cue_times()) == 10 and s.cue_every_s == 2.0 and ALL_STEPS[s.name] is s for s in taps)


def test_cue_report_says_how_many_taps_were_recognised_per_finger():
    from handpiano.interaction.trace import CueResult, cue_report

    results = [CueResult("LEFT_INDEX", 5, 4, 1, 2, ()), CueResult("RIGHT_INDEX", 5, 1, 4, 0, ()),
               CueResult("LEFT_RING", 0, 0, 0, 3, ())]
    assert cue_report(results) == ["Toques reconocidos: 5/10 (50%) · notas de más: 5",
                                   "índice izq 4/5 · índice der 1/5"]
    assert cue_report([]) == []


def test_finger_note_recordings_ask_to_bend_with_the_hands_up_and_keep_the_same_cues():
    from handpiano.interaction.trace import PROTOCOLS, finger_note_steps

    plain = PROTOCOLS["dedos"]
    bent = finger_note_steps(plain)
    assert [(s.name, s.duration_s, s.expected, s.cue_every_s) for s in bent] == [
        (s.name, s.duration_s, s.expected, s.cue_every_s) for s in plain]
    assert "LEVANTA" in bent[0].instruction and "piano dibujado" not in bent[0].instruction
    assert bent[1].instruction.startswith("DOBLA el ÍNDICE IZQUIERDO (Fa)")
    assert all("¡TOCA!" not in s.instruction for s in bent)
