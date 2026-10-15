"""--check-keys: deterministic probe points, sample categories, metrics and the real app flow."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import csv  # noqa: E402
from dataclasses import replace  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from handpiano.app.application import Application  # noqa: E402
from handpiano.app.config import AppConfig  # noqa: E402
from handpiano.app.pipeline import FrameTimestamps, PipelineState, TrackingSnapshot  # noqa: E402
from handpiano.camera.camera_config import CameraMode  # noqa: E402
from handpiano.camera.frame import Frame  # noqa: E402
from handpiano.instrument.keyboard import KeyboardLayout  # noqa: E402
from handpiano.mapping.calibration import CalibrationStore, CameraKey, PianoCalibration  # noqa: E402
from handpiano.mapping.key_accuracy import (  # noqa: E402
    ADJACENT,
    EXACT,
    FAR,
    MISS,
    OUTSIDE_CORRECT,
    OUTSIDE_FALSE_POSITIVE,
    AccuracyConfig,
    KeyAccuracyCheck,
    categorize,
    default_probe_points,
    keys_adjacent,
)
from handpiano.mapping.key_mapper import KeyMapper  # noqa: E402
from handpiano.tracking.landmarks import FingerId  # noqa: E402
from handpiano.ui.main_window import MainWindow  # noqa: E402
from tests.test_mapping import _state_with_tips  # noqa: E402

LAYOUT = KeyboardLayout.from_octaves("C4", 2)
CAM = CameraKey(0, 320, 240, True)
RECT = [(0.2, 0.5), (0.8, 0.5), (0.8, 0.9), (0.2, 0.9)]
CAL = PianoCalibration.from_corners(RECT, LAYOUT.size, CAM)
FAST = AccuracyConfig(prepare_s=0.5, min_settle_s=0.2, stable_s=0.3, timeout_s=3.0)
DT = 1 / 24
F = FingerId.RIGHT_INDEX


def key(name):
    return next(k for k in LAYOUT.keys if k.name == name)


def cam_of(piano_xy):
    return tuple(float(v) for v in CAL.to_camera([piano_xy])[0])


# --- probe points & categories -------------------------------------------------------


def test_probe_points_are_deterministic_and_cover_all_kinds():
    points = default_probe_points(LAYOUT)
    assert points == default_probe_points(LAYOUT)
    kinds = [p.kind for p in points]
    assert kinds.count("white") == 7 and kinds.count("black") == 5
    assert kinds.count("border") == 6 and kinds.count("outside") == 4
    assert [p.sample_id for p in points] == list(range(1, len(points) + 1))
    for p in points:
        # The expected key is exactly what the existing key_at says for the target point.
        assert p.expected is LAYOUT.key_at(*p.piano_xy)
        assert (p.expected is None) == (p.kind == "outside")
    whites = [p.expected.name for p in points if p.kind == "white"]
    assert whites == ["C4", "E4", "G4", "B4", "D5", "F5", "A5"]
    assert all(p.expected.is_black for p in points if p.kind == "black")
    borders = [p.expected.name for p in points if p.kind == "border"]
    assert borders[:4] == ["E4", "F4", "B4", "C5"]
    assert key(borders[4]).is_black and not key(borders[5]).is_black  # just inside / just below a black key


def test_adjacency_is_geometric():
    assert keys_adjacent(key("E4"), key("F4"))  # white neighbours
    assert keys_adjacent(key("C#4"), key("C4")) and keys_adjacent(key("C#4"), key("D4"))
    assert keys_adjacent(key("D4"), key("E4"))  # touch below D#4
    assert not keys_adjacent(key("C4"), key("E4"))
    assert not keys_adjacent(key("C#4"), key("D#4"))  # black keys do not touch
    assert not keys_adjacent(key("C4"), key("C4"))


@pytest.mark.parametrize(
    ("expected", "detected", "had_position", "category"),
    [
        ("E4", "E4", True, EXACT),
        ("E4", "F4", True, ADJACENT),
        ("E4", "D#4", True, ADJACENT),
        ("E4", "A5", True, FAR),
        ("E4", None, True, MISS),  # finger left the keyboard
        ("E4", None, False, MISS),  # no fingertip at all
        (None, None, True, OUTSIDE_CORRECT),
        (None, "C4", True, OUTSIDE_FALSE_POSITIVE),
        (None, None, False, MISS),  # absence of a finger is not evidence of "outside"
    ],
)
def test_categories(expected, detected, had_position, category):
    e = None if expected is None else key(expected)
    d = None if detected is None else key(detected)
    assert categorize(e, d, had_position) == category


# --- sampling ---------------------------------------------------------------------------


def _check(points=None):
    check = KeyAccuracyCheck(KeyMapper(LAYOUT, CAL), config=FAST)
    if points is not None:
        check.points = points
    check.start(0.0)
    return check


def _hold(check, xy, t0, frames=40, finger=F):
    t = t0
    for _ in range(frames):
        sample = check.update(t, finger, xy)
        t += DT
        if sample is not None:
            return sample, t
    return None, t


def test_nothing_is_sampled_while_preparing():
    check = _check()
    assert check.update(0.1, F, cam_of(check.current.piano_xy)) is None
    assert check.samples == []


def test_exact_sample_uses_the_shared_homography_and_key_at():
    check = _check()
    target = check.current
    sample, _ = _hold(check, cam_of(target.piano_xy), 0.6)
    assert sample is not None
    assert sample.error_category == EXACT and sample.correct
    assert sample.expected_key == sample.detected_key == "C4"
    assert sample.piano_xy == pytest.approx(target.piano_xy, abs=1e-6)
    assert sample.distance_to_expected_center == pytest.approx(0.0, abs=1e-6)
    assert sample.finger is F and sample.sample_id == 1
    assert check.current.sample_id == 2


def test_incorrect_samples_adjacent_and_far():
    check = _check()
    c4 = check.current
    sample, t = _hold(check, cam_of((c4.piano_xy[0] + 1.0, c4.piano_xy[1])), 0.6)  # on D4
    assert (sample.detected_key, sample.error_category, sample.correct) == ("D4", ADJACENT, False)
    sample, _ = _hold(check, cam_of(key("A5").center), t)  # target E4, finger on A5
    assert (sample.expected_key, sample.detected_key, sample.error_category) == ("E4", "A5", FAR)


def test_outside_points_correct_and_false_positive():
    points = tuple(p for p in default_probe_points(LAYOUT) if p.kind == "outside")
    check = _check(points)
    sample, t = _hold(check, cam_of(points[0].piano_xy), 0.6)
    assert (sample.expected_key, sample.detected_key, sample.error_category) == (None, None, OUTSIDE_CORRECT)
    assert sample.correct
    sample, _ = _hold(check, cam_of(key("G4").center), t)  # asked for outside, finger is on the keyboard
    assert sample.error_category == OUTSIDE_FALSE_POSITIVE and sample.detected_key == "G4"


def test_moving_finger_is_not_sampled_and_timeout_is_a_miss():
    check = _check()
    target_start = FAST.prepare_s  # the first target starts when preparation ends
    t = target_start
    while t < target_start + FAST.timeout_s - DT:
        x = 0.3 + 0.02 * np.sin(t * 20)  # never stable
        assert check.update(t, F, (x, 0.7)) is None
        t += DT
    sample = check.update(target_start + FAST.timeout_s + 0.01, F, (0.3, 0.7))
    assert sample.error_category == MISS and sample.detected_key is None and not sample.correct


def test_switching_finger_restarts_the_stability_window():
    check = _check()
    xy = cam_of(check.current.piano_xy)
    t = 0.6
    for i in range(30):
        finger = FingerId.RIGHT_INDEX if i % 2 else FingerId.LEFT_INDEX
        assert check.update(t, finger, xy) is None
        t += DT


def test_metrics_are_computed_from_samples():
    points = default_probe_points(LAYOUT)
    inside = [p for p in points if p.expected is not None][:4]
    outside = [p for p in points if p.expected is None][:2]
    check = _check(tuple(inside + outside))
    t = 0.6
    # inside: exact, adjacent (one white to the right), far, miss (timeout)
    _, t = _hold(check, cam_of(inside[0].piano_xy), t)
    _, t = _hold(check, cam_of((inside[1].piano_xy[0] + 1.0, inside[1].piano_xy[1])), t)
    _, t = _hold(check, cam_of(key("B5").center), t)
    check.update(t + FAST.timeout_s + 0.01, None, None)
    t += FAST.timeout_s + 0.02
    # outside: correct, false positive
    _, t = _hold(check, cam_of(outside[0].piano_xy), t)
    _, t = _hold(check, cam_of(key("C4").center), t)
    assert check.done
    r = check.report()
    assert (r.inside, r.exact, r.adjacent, r.far, r.miss_inside) == (4, 1, 1, 1, 1)
    assert r.exact_accuracy == pytest.approx(0.25)
    assert (r.outside, r.outside_correct, r.outside_false_positive) == (2, 1, 1)
    assert r.outside_accuracy == pytest.approx(0.5)
    lines = r.lines()
    assert "Exact accuracy: 25.0%" in lines and "Outside accuracy: 50.0%" in lines


def test_report_is_na_without_samples():
    r = _check().report()
    assert r.exact_accuracy is None and r.outside_accuracy is None
    assert "Exact accuracy: N/A" in r.lines()


def test_csv_contains_the_real_samples(tmp_path):
    check = _check(default_probe_points(LAYOUT)[:2])
    _, t = _hold(check, cam_of(check.current.piano_xy), 0.6)
    _hold(check, cam_of(key("A5").center), t)
    path = tmp_path / "acc.csv"
    check.write_csv(path)
    rows = list(csv.DictReader(path.open()))
    assert len(rows) == 2
    assert rows[0]["expected_key"] == "C4" and rows[0]["error_category"] == EXACT and rows[0]["correct"] == "True"
    assert rows[1]["detected_key"] == "A5" and rows[1]["error_category"] == FAR
    assert {"sample_id", "expected_key", "detected_key", "correct", "error_category", "finger", "timestamp"} <= set(rows[0])


# --- the real --check-keys flow in the app --------------------------------------------------


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


class _FakePipeline:
    def __init__(self, mapper):
        self.state = PipelineState.RUNNING
        self.mode = CameraMode(0, "fake", 320, 240, 30, 320, 240, 30.0, False, False)
        self.error_message = None
        self.snapshot = None
        self.mapper = mapper

    def latest_snapshot(self):
        return self.snapshot

    def drain_events(self):
        return []

    def stop(self):
        return True


class _Run:
    def __init__(self, tmp_path, monkeypatch, saved=True, saved_camera=CAM):
        store_path = tmp_path / "cal.json"
        if saved:
            CalibrationStore(store_path).save(PianoCalibration.from_corners(RECT, LAYOUT.size, saved_camera))
        config = AppConfig()
        config = replace(config, piano=replace(config.piano, calibration_path=store_path))
        self.window = MainWindow(config.camera, layout=LAYOUT)
        self.app = Application(config, self.window)
        self.exits = []
        self.app.piano.on_exit = self.exits.append
        self.fake = _FakePipeline(self.app.piano.mapper)
        self.app.pipeline = self.fake
        self.monkeypatch = monkeypatch
        self.seq = 0
        self.now = 0.0

    def poll(self, tip=None, two_hands=False, dt=DT):
        self.monkeypatch.setattr("handpiano.app.application.time.monotonic", lambda: self.now)
        tips = {} if tip is None else {F: tip}
        if two_hands:
            tips[FingerId.LEFT_INDEX] = (0.05, 0.05)
        self.seq += 1
        state = _state_with_tips(tips)
        image = np.zeros((240, 320, 3), np.uint8)
        self.fake.snapshot = TrackingSnapshot(self.seq, Frame(self.seq, image, 0, 1), state,
                                              FrameTimestamps(0, 1, 2, 3, 4, 5, 6), self.fake.mapper.map(state))
        self.app._poll_frame()
        self.now += dt

    def close(self):
        self.app.pipeline = None
        self.app.shutdown()
        self.window.close()


def test_valid_calibration_measures_without_calibrating(qapp, tmp_path, monkeypatch, capsys):
    run = _Run(tmp_path, monkeypatch)
    run.app.request_key_check(tmp_path / "out.csv")
    assert not run.window.calibrate_button.isEnabled() and not run.window.forget_button.isEnabled()
    run.poll()  # camera RUNNING → saved calibration loaded → test starts
    piano = run.app.piano
    assert piano.calibrated and piano.session is None
    assert piano.accuracy is not None
    # Clicking "Calibrar piano" during the test is ignored and does not touch the test.
    run.app.start_calibration()
    piano.forget_calibration()
    assert piano.session is None and piano.accuracy is not None and piano.calibrated

    # Point exactly at every target until the test finishes.
    while piano.accuracy is not None:
        point = piano.accuracy.current if not piano.accuracy.preparing(run.now) else None
        run.poll(None if point is None else cam_of(point.piano_xy))
        assert run.now < 400, "test did not finish"
    report = piano.last_report
    assert report.inside == 18 and report.exact == 18 and report.exact_accuracy == 1.0
    assert report.outside >= 1 and report.outside_false_positive == 0
    assert piano.session is None  # never calibrated
    assert run.exits == []  # stays open with the results
    out = capsys.readouterr().out
    assert "=== KEY ACCURACY ===" in out and "Exact accuracy: 100.0%" in out
    rows = list(csv.DictReader((tmp_path / "out.csv").open()))
    assert len(rows) == report.inside + report.outside
    assert run.window.camera_view._banner[0] == "Prueba de precisión terminada"
    assert run.window.calibrate_button.isEnabled()
    run.close()


def test_missing_calibration_ends_with_clear_message(qapp, tmp_path, monkeypatch, capsys):
    run = _Run(tmp_path, monkeypatch, saved=False)
    run.app.request_key_check(None)
    run.poll()
    piano = run.app.piano
    assert piano.accuracy is None and piano.session is None
    banner = run.window.camera_view._banner
    assert "No valid calibration found." in banner[1] and "Run the normal calibration first." in banner[1]
    assert "No valid calibration found." in capsys.readouterr().err
    for _ in range(int(6.5 / DT)):
        run.poll()
    assert run.exits == [1]
    assert piano.session is None  # never started a calibration
    run.close()


def test_calibration_for_another_camera_is_reported(qapp, tmp_path, monkeypatch):
    run = _Run(tmp_path, monkeypatch, saved_camera=CameraKey(0, 1920, 1080, True))
    run.app.request_key_check(None)
    run.poll()
    banner = run.window.camera_view._banner
    assert "No valid calibration found." in banner[1]
    assert any("otra cámara, resolución o modo espejo" in line for line in banner[1])
    run.close()


def test_two_hands_block_sampling_and_esc_gives_partial_results(qapp, tmp_path, monkeypatch, capsys):
    run = _Run(tmp_path, monkeypatch)
    run.app.request_key_check(None)
    run.poll()
    piano = run.app.piano
    while piano.accuracy.preparing(run.now):
        run.poll()
    target = piano.accuracy.current
    for _ in range(48):  # 2 s holding still on target, but with two hands visible
        run.poll(cam_of(target.piano_xy), two_hands=True)
    assert piano.accuracy.samples == []
    assert "dos manos" in run.window.camera_view._banner[1][1]
    for _ in range(30):
        run.poll(cam_of(target.piano_xy))
    assert len(piano.accuracy.samples) == 1
    piano.key_command("cancel", run.now)
    assert piano.accuracy is None and piano.last_report.inside == 1
    assert "interrumpida" in capsys.readouterr().out
    run.close()


# --- regressions from the first real --check-keys run (precision.csv, 2026-10-08) -------------


def test_finger_resting_on_the_previous_target_is_not_recorded_for_the_next():
    """Real run: samples 11–12 recorded A#4 for D#5 and G#5 ~1 s apart because the finger never moved."""
    check = _check()
    first = check.current
    _, t = _hold(check, cam_of(first.piano_xy), 0.6)  # exact sample on C4
    assert len(check.samples) == 1 and check.waiting_to_move
    # Stay on C4 for the whole time of the next target (E4).
    while len(check.samples) == 1:
        check.update(t, F, cam_of(first.piano_xy))
        t += DT
    sample = check.samples[1]
    assert sample.expected_key == "E4" and sample.error_category == MISS
    assert sample.miss_reason == "did_not_move" and sample.frames_waiting_move > 0
    assert sample.detected_key is None  # not the resting finger's key


def test_moving_to_the_new_target_is_recorded_normally():
    check = _check()
    _, t = _hold(check, cam_of(check.current.piano_xy), 0.6)
    target = check.current
    sample, _ = _hold(check, cam_of(target.piano_xy), t)
    assert sample is not None and sample.error_category == EXACT and sample.expected_key == "E4"


def test_no_movement_needed_when_the_next_target_is_right_there():
    points = tuple(p for p in default_probe_points(LAYOUT) if p.kind == "border")[:2]  # E4 then F4, 0.4 keys apart
    check = _check(points)
    _, t = _hold(check, cam_of(points[0].piano_xy), 0.6)
    assert not check.waiting_to_move


@pytest.mark.parametrize(
    ("feed", "reason"),
    [
        (lambda t: (None, None, False), "no_index"),
        (lambda t: (None, None, True), "two_hands"),
        (lambda t: (F, (0.3 + 0.03 * np.sin(t * 25), 0.7), False), "unstable"),
    ],
)
def test_miss_reason_records_why_no_sample_was_taken(feed, reason):
    check = _check()
    t = FAST.prepare_s
    while not check.samples:
        finger, xy, two = feed(t)
        check.update(t, finger, xy, two)
        t += DT
    sample = check.samples[0]
    assert sample.error_category == MISS and sample.miss_reason == reason
    assert sample.frames_no_index + sample.frames_two_hands + sample.stability_resets > 0


def test_csv_includes_miss_diagnostics(tmp_path):
    check = _check(default_probe_points(LAYOUT)[:1])
    check.update(FAST.prepare_s + 0.1, None, None, True)
    check.update(FAST.prepare_s + FAST.timeout_s + 0.01, None, None)
    path = tmp_path / "acc.csv"
    check.write_csv(path)
    row = next(csv.DictReader(path.open()))
    assert row["miss_reason"] == "two_hands" and row["frames_two_hands"] == "1"
