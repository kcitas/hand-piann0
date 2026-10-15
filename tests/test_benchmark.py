"""Threshold-free signal benchmark and the PianoVAM adapter (synthetic fixtures, no downloads)."""

import csv
import json
import subprocess
import sys

import numpy as np
import pytest

from handpiano.interaction.benchmark import (
    Event,
    Series,
    auc,
    excursion,
    hand_of_finger,
    score_signals,
    series_from_trace,
    trace_events,
)
from handpiano.interaction.pianovam import (
    detect_label_swap,
    events_from_recording,
    key_x,
    load_recording,
    parse_duration,
    series_from_recording,
)
from tests.test_trace import _synthetic_rows

FPS = 30.0


# --- measures ---------------------------------------------------------------------------------


def test_auc_basic_properties():
    assert auc([2, 3], [0, 1]) == 1.0
    assert auc([0, 1], [2, 3]) == 0.0
    assert auc([1, 1], [1, 1]) == 0.5
    assert auc([], [1]) is None and auc([1], []) is None
    assert auc([1, float("nan")], [0]) == 1.0  # NaN ignored


def _series(fn, seconds=6.0, features=("tip_y", "flex", "tip_z")):
    t = np.arange(0, seconds, 1 / FPS)
    return Series(t, {f: np.array([fn(x) for x in t]) for f in features})


def test_excursion_level_and_velocity():
    s = _series(lambda t: 0.5 if 2.0 <= t < 2.3 else 0.0)
    assert excursion(s, "tip_y", 2.0, (-0.8, -0.3), (-0.3, 1.0)) == pytest.approx(0.5)
    assert excursion(s, "tip_y", 2.0, (-0.8, -0.3), (-0.3, 1.0), use_velocity=True) == pytest.approx(0.5 * FPS, rel=0.05)
    gap = Series(s.t, {"tip_y": np.full(len(s.t), np.nan)})
    assert np.isnan(excursion(gap, "tip_y", 2.0, (-0.8, -0.3), (-0.3, 1.0)))


def _hand(press_finger_fn, common_fn, fingers=("RIGHT_INDEX", "RIGHT_MIDDLE", "RIGHT_RING", "RIGHT_PINKY")):
    return {f: _series(lambda t, f=f: common_fn(t) + press_finger_fn(f, t), seconds=20) for f in fingers}


def test_score_signals_separates_the_pressing_finger():
    times = [2.0, 5.0, 8.0, 11.0]
    series = _hand(lambda f, t: 0.6 if f == "RIGHT_INDEX" and any(e <= t < e + 0.3 for e in times) else 0.0,
                   lambda t: 0.0)
    scores = score_signals(series, [Event("RIGHT_INDEX", e) for e in times], [15.0, 17.0], hand_of_finger)
    pooled = next(s for s in scores if s.variant == "tip_y" and s.finger == "ALL")
    assert pooled.auc_neighbours == 1.0 and pooled.rank1 == 1.0 and pooled.events == 4
    assert pooled.auc_background == 1.0


def test_common_mode_variant_helps_when_the_whole_hand_moves_at_events():
    rng = np.random.default_rng(0)
    times = [2.0 + 2.5 * k for k in range(6)]
    amps = {e: rng.uniform(0.0, 1.0) for e in times}  # whole-hand movement of random size at each event

    def common(t):
        return next((amps[e] for e in times if e <= t < e + 0.3), 0.0)

    # The pressing finger moves 0.3 more than the hand; other fingers follow the hand with different gains.
    gains = {"RIGHT_INDEX": 1.0, "RIGHT_MIDDLE": 1.1, "RIGHT_RING": 0.9, "RIGHT_PINKY": 1.05}
    fingers = list(gains)
    def signal(f, t):
        pressing = f == "RIGHT_INDEX" and any(e <= t < e + 0.3 for e in times)
        return gains[f] * common(t) + (0.3 if pressing else 0.0)

    series = {f: _series(lambda t, f=f: signal(f, t), seconds=20) for f in fingers}
    scores = score_signals(series, [Event("RIGHT_INDEX", e) for e in times], [18.0], hand_of_finger)
    raw = next(s for s in scores if s.variant == "tip_y" and s.finger == "ALL").auc_neighbours
    cm = next(s for s in scores if s.variant == "tip_y:cm" and s.finger == "ALL").auc_neighbours
    assert cm >= raw


def test_trace_adapters_use_cues_and_no_press_steps():
    rows = _synthetic_rows([("rest", 2.0, lambda f, t: 0.0),
                            ("taps_RIGHT_INDEX", 6.0, lambda f, t: 0.0)], cue_every=1.5, cue_finger="RIGHT_INDEX")
    events, background = trace_events(rows)
    assert events and all(e.finger == "RIGHT_INDEX" for e in events)
    assert background and max(background) < 2.0  # only from the rest step
    series = series_from_trace(rows)
    assert set(series) == {"LEFT_INDEX", "LEFT_MIDDLE", "LEFT_RING", "LEFT_PINKY",
                           "RIGHT_INDEX", "RIGHT_MIDDLE", "RIGHT_RING", "RIGHT_PINKY"}


def test_benchmark_script_runs_on_a_trace(tmp_path):
    rows = _synthetic_rows([("rest", 2.0, lambda f, t: 0.0),
                            ("taps_RIGHT_INDEX", 6.0, lambda f, t: 0.4 if f == "RIGHT_INDEX" and (t % 1.5) < 0.3 else 0.0)],
                           cue_every=1.5, cue_finger="RIGHT_INDEX")
    path = tmp_path / "t.csv"
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    out = subprocess.run([sys.executable, "scripts/benchmark_signals.py", str(path)],
                         capture_output=True, text=True, check=True).stdout
    assert "AUC vs neighbours" in out and "tip_y:cm" in out


# --- PianoVAM adapter -----------------------------------------------------------------------


def _hand_points(tip_x, wrist=(0.5, 0.8)):
    """A 21-point hand whose index tip (landmark 8) is at tip_x (normalized)."""
    pts = [[wrist[0], wrist[1], 0.0] for _ in range(21)]
    for i in range(21):
        pts[i] = [wrist[0] + 0.001 * i, wrist[1] - 0.01 * i, -0.001 * i]
    pts[9] = [wrist[0], wrist[1] - 0.1, 0.0]  # middle MCP: hand size
    pts[8] = [tip_x, 0.6, -0.02]
    return pts


def _write_pianovam(tmp_path, swap_labels: bool):
    lt, rt = (100.0, 400.0), (1820.0, 400.0)
    c4 = key_x(60, lt, rt) / 1920  # normalized x of middle C
    a5 = key_x(81, lt, rt) / 1920
    frames = {}
    for i in range(300):  # 10 s at 30 fps
        left, right = _hand_points(c4), _hand_points(a5)
        frames[str(i)] = {"Left": right, "Right": left} if swap_labels else {"Left": left, "Right": right}
    (tmp_path / "Handskeleton").mkdir()
    (tmp_path / "Fingering_GT").mkdir()
    (tmp_path / "Handskeleton" / "rec.json").write_text(json.dumps(frames))
    notes = [("1.0", 60, "L", "2"), ("3.0", 81, "R", "2"), ("3.1", 79, "R", "3"), ("6.0", 60, "L", "1")]
    with (tmp_path / "Fingering_GT" / "rec.tsv").open("w") as f:
        f.write("onset\tkey_offset\tframe_offset\tnote\tvelocity\thand\tfinger\n")
        for onset, note, hand, finger in notes:
            f.write(f"{onset}\t0\t0\t{note}\t64\t{hand}\t{finger}\n")
    meta = {"0": {"record_time": "rec", "duration": "0 days 00:00:10.000.000000",
                  "Point_LT": "100, 400", "Point_RT": "1820, 400", "Point_RB": "1820, 600", "Point_LB": "100, 600"}}
    (tmp_path / "metadata.json").write_text(json.dumps(meta))
    return load_recording(tmp_path / "Handskeleton" / "rec.json", tmp_path / "Fingering_GT" / "rec.tsv",
                          tmp_path / "metadata.json")


def test_parse_duration_and_key_positions():
    assert parse_duration("0 days 00:02:57.835.000000") == pytest.approx(177.835)
    lt, rt = (0.0, 0.0), (52.0, 0.0)
    assert key_x(21, lt, rt) == pytest.approx(0.5)  # A0: first white key centre
    assert key_x(22, lt, rt) == pytest.approx(1.0)  # A#0: boundary between the first two whites
    assert key_x(108, lt, rt) == pytest.approx(51.5)  # C8: last white key


@pytest.mark.parametrize("swapped", [False, True])
def test_label_swap_is_measured_not_assumed(tmp_path, swapped):
    rec = _write_pianovam(tmp_path, swap_labels=swapped)
    assert rec.fps == pytest.approx(30.0)
    assert detect_label_swap(rec) is swapped


def test_pianovam_events_skip_thumbs_and_mark_isolated_onsets(tmp_path):
    rec = _write_pianovam(tmp_path, swap_labels=True)
    events, isolated = events_from_recording(rec, isolation_s=0.25)
    assert [(e.finger, e.t) for e in events] == [("LEFT_INDEX", 1.0), ("RIGHT_INDEX", 3.0), ("RIGHT_MIDDLE", 3.1)]
    assert [(e.finger, e.t) for e in isolated] == [("LEFT_INDEX", 1.0)]  # 3.0 and 3.1 are 0.1 s apart


def test_pianovam_series_use_handpiano_features_on_physical_hands(tmp_path):
    rec = _write_pianovam(tmp_path, swap_labels=True)
    series = series_from_recording(rec, swapped=True)
    assert set(series) == {f"{h}_{f}" for h in ("LEFT", "RIGHT") for f in ("INDEX", "MIDDLE", "RING", "PINKY")}
    assert np.all(np.isfinite(series["LEFT_INDEX"].values["tip_y"]))
    assert len(series["LEFT_INDEX"].t) == 300
