"""Trace format v4: landmark sidecar, metadata, key labels, compatibility with older traces."""

import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

from handpiano.app.pipeline import FrameTimestamps, TrackingSnapshot
from handpiano.camera.frame import Frame
from handpiano.interaction.finger_presses import PressDetector
from handpiano.interaction.labels import load_labels
from handpiano.interaction.landmark_features import landmark_features
from handpiano.interaction.press_detector import PressConfig
from handpiano.interaction.trace import (
    COLUMNS,
    PROTOCOLS,
    TraceRecorder,
    load_trace,
    replay,
)
from handpiano.interaction.trace_landmarks import (
    LANDMARK_COLUMNS,
    load_landmarks,
    load_meta,
    series_from_landmarks,
    sidecar_paths,
)
from handpiano.mapping.key_mapper import KeyMapper
from handpiano.tracking.finger_tracker import FingerTracker, FingerTrackerConfig
from handpiano.tracking.landmarks import (
    FingerId,
    Handedness,
    HandObservation,
    observations_from_result,
)
from handpiano.tracking.smoothing import SmoothingConfig
from tests.fakes import make_hand
from tests.test_press_detector import ASPECT, CAL, DT, LAYOUT, finger_tap

W, H = 16, 9  # image size with the same aspect as the rig (720/1280)
assert abs(H / W - ASPECT) < 1e-9


def _record_v4(tmp_path, keys=(), name="rec.csv", smoothing=True):
    """Real tracker (One Euro on) + mapper + detector; the LEFT hand disappears for a while."""
    tracker = FingerTracker(FingerTrackerConfig() if smoothing else FingerTrackerConfig(smoothing=SmoothingConfig()))
    mapper = KeyMapper(LAYOUT, CAL)
    detector = PressDetector(LAYOUT)
    recorder = TraceRecorder(PROTOCOLS["keys"], protocol="keys", config=PressConfig(), mirrored=True)
    image = np.zeros((H, W, 3), np.uint8)
    t = 1000.0  # perf_counter-like origin
    frames = [({}, True)] * 30 + [(o, True) for o in finger_tap(FingerId.RIGHT_INDEX)] + [({}, False)] * 20 \
        + [({}, True)] * 20
    for seq, (offsets, left_visible) in enumerate(frames, start=1):
        obs = []
        for hand in (Handedness.LEFT, Handedness.RIGHT):
            if hand is Handedness.LEFT and not left_visible:
                continue
            o = make_hand(hand, (0.35 if hand is Handedness.LEFT else 0.65, 0.75))
            pts = o.landmarks.copy()
            for finger, dy in offsets.items():
                if finger.hand is hand:
                    pts[finger.finger.tip, 1] += dy
            obs.append(HandObservation(hand, 0.9, pts, world_landmarks=(pts - pts[0]) * 0.1))
        state = tracker.update(obs, t)
        mapping = mapper.map(state)
        presses = detector.update(t, state, mapping, ASPECT)
        recorder.step = "rest" if seq < 30 else "key_taps_RIGHT_INDEX"
        snap = TrackingSnapshot(seq, Frame(seq, image, int(t * 1e9), int(t * 1e9)), state,
                                FrameTimestamps(*[0] * 7), mapping, presses)
        recorder.on_frame(snap)
        t += DT
    for key, down, kt in keys:
        recorder.record_key(key, down, kt)
    path = tmp_path / name
    written = recorder.write(path)
    return path, written


def _sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def test_v4_writes_the_file_set_and_never_overwrites(tmp_path):
    path, written = _record_v4(tmp_path, keys=[("J", True, 1001.2), ("J", False, 1001.4)])
    files = sidecar_paths(path)
    assert set(written) == {"trace", "landmarks", "meta", "labels"} and all(p.is_file() for p in written.values())
    before = {k: _sha(p) for k, p in written.items()}
    # A second recording with the same name is refused and nothing changes.
    with pytest.raises(FileExistsError):
        _record_v4(tmp_path)
    assert {k: _sha(p) for k, p in written.items()} == before
    # Also when only a sidecar exists.
    other = tmp_path / "other.csv"
    sidecar_paths(other)["meta"].write_text("{}")
    with pytest.raises(FileExistsError):
        _record_v4(tmp_path, name="other.csv")
    assert not other.exists()
    meta = load_meta(files["meta"])
    assert meta["version"] == 4 and meta["protocol"] == "keys" and meta["mirrored"] is True
    assert meta["aspect"] == pytest.approx(ASPECT) and meta["press_config"]["feature"] == "tip_y"


def test_no_labels_file_without_key_events(tmp_path):
    path, written = _record_v4(tmp_path)
    assert "labels" not in written and not sidecar_paths(path)["labels"].exists()


def test_per_finger_trace_keeps_old_columns_first(tmp_path):
    path, _ = _record_v4(tmp_path)
    rows = load_trace(path)
    assert tuple(rows[0]) == COLUMNS and COLUMNS[:18] == (
        "t", "step", "finger", "tracked", "discontinuity", "hand_speed", "tip_y", "flex", "tip_z",
        "piano_x", "piano_y", "raw_key", "stable_key", "phase", "depth", "velocity", "events", "cue")
    left = [r for r in rows if r["finger"] == "LEFT_INDEX"]
    assert {r["hand_status"] for r in left} == {"TRACKED", "HELD", "LOST"}
    assert all(r["baseline"] for r in left[10:25])


def test_landmark_rows_cover_both_hands_every_frame_including_loss(tmp_path):
    path, _ = _record_v4(tmp_path)
    with sidecar_paths(path)["landmarks"].open() as f:
        reader = csv.DictReader(f)
        assert tuple(reader.fieldnames or ()) == LANDMARK_COLUMNS
        raw_rows = list(reader)
    loaded = load_landmarks(sidecar_paths(path)["landmarks"])
    assert loaded.issues == []
    frames = len(load_trace(path)) // 8
    assert len(loaded.samples) == 2 * frames
    left = [s for s in loaded.samples if s.hand is Handedness.LEFT]
    statuses = [s.status for s in left]
    assert "HELD" in statuses and "LOST" in statuses
    lost = [s for s in left if s.status == "LOST"]
    assert all(s.smoothed is None and s.raw is None and not s.detected and s.world is None for s in lost)
    held = [s for s in left if s.status == "HELD"]
    assert all(s.smoothed is not None and s.raw is None for s in held)  # stale smoothing, no fresh detection
    # Visibility / presence were not provided by the (fake) detector: the cells stay empty.
    assert all(r["vis0"] == "" and r["pres20"] == "" for r in raw_rows)
    tracked = [s for s in loaded.samples if s.tracked]
    assert all(s.world is not None and s.raw is not None for s in tracked)
    assert all(np.all(np.diff([s.t for s in loaded.samples if s.hand is h]) > 0) for h in Handedness)


def test_smoothed_landmarks_reproduce_the_recorded_signals(tmp_path):
    """The landmark file must allow rebuilding the detector's input exactly (up to CSV rounding)."""
    path, _ = _record_v4(tmp_path)
    loaded = load_landmarks(sidecar_paths(path)["landmarks"])
    meta = load_meta(sidecar_paths(path)["meta"])
    series = series_from_landmarks(loaded.samples, meta["aspect"], features=("tip_y", "flex", "tip_z"))
    rows = load_trace(path)
    for finger in ("RIGHT_INDEX", "LEFT_RING"):
        recorded = {r["t"]: r for r in rows if r["finger"] == finger}
        s = series[finger]
        compared = 0
        for t, v in zip(s.t, s.values["tip_y"], strict=True):
            r = recorded[f"{t:.4f}"]
            if r["tracked"] == "True":
                assert v == pytest.approx(float(r["tip_y"]), abs=2e-4)
                compared += 1
            else:
                assert np.isnan(v)
        assert compared > 50


def test_old_traces_still_replay_without_the_new_columns(tmp_path):
    path, _ = _record_v4(tmp_path)
    rows = load_trace(path)
    old_rows = [{k: v for k, v in r.items() if k in COLUMNS[:18]} for r in rows]
    config = PressConfig()
    assert replay(old_rows, config, LAYOUT) == replay(rows, config, LAYOUT)


def test_key_events_become_independent_labels(tmp_path):
    _, written = _record_v4(tmp_path, keys=[("J", True, 1001.2), ("J", False, 1001.4), ("x", True, 1002.0),
                                              ("X", False, 1002.1), ("ñ", True, 1003.0)])
    labels = load_labels(written["labels"])
    got = [(lb.label, lb.target, lb.source) for lb in labels]
    assert ("contact", "RIGHT_INDEX", "keyboard") in got and ("release", "RIGHT_INDEX", "keyboard") in got
    assert ("ambiguous", "ALL", "keyboard") in got  # X is not a protocol key
    assert ("contact", "RIGHT_PINKY", "keyboard") in got  # ñ upper-cased, no release (still down at the end)
    assert sum(1 for g in got if g[0] == "release") == 1


def test_landmark_loader_validates_timestamps_and_coordinates(tmp_path):
    path, _ = _record_v4(tmp_path)
    src = sidecar_paths(path)["landmarks"]
    lines = src.read_text().splitlines()
    header = lines[0].split(",")

    def write(name, rows):
        p = tmp_path / name
        p.write_text("\n".join([lines[0], *[",".join(r) for r in rows]]) + "\n")
        return p

    rows = [line.split(",") for line in lines[1:]]
    # Timestamps going back for the same hand → error naming the line.
    swapped = [rows[2], rows[3], rows[0], rows[1], *rows[4:]]
    with pytest.raises(ValueError, match=":4: timestamp"):
        load_landmarks(write("back.csv", swapped))
    bad_t = [r.copy() for r in rows]
    bad_t[0][0] = "nan"
    with pytest.raises(ValueError, match="non-finite"):
        load_landmarks(write("nan_t.csv", bad_t))
    bad_t[0][0] = "abc"
    with pytest.raises(ValueError, match="bad timestamp"):
        load_landmarks(write("abc_t.csv", bad_t))
    # Invalid coordinates are dropped and reported; a TRACKED row without smoothed data becomes LOST.
    bad_xy = [r.copy() for r in rows]
    i_sm, i_raw, i_w = header.index("sm_x8"), header.index("raw_y0"), header.index("w_z3")
    tracked_row = next(i for i, r in enumerate(bad_xy) if r[header.index("status")] == "TRACKED")
    bad_xy[tracked_row][i_sm] = "inf"
    bad_xy[tracked_row][i_raw] = "7.5"
    bad_xy[tracked_row][i_w] = "nan"
    loaded = load_landmarks(write("coords.csv", bad_xy))
    assert len(loaded.issues) == 4
    s = loaded.samples[tracked_row]
    assert s.smoothed is None and s.raw is None and s.world is None and s.status == "LOST"
    # Missing columns.
    p = tmp_path / "cols.csv"
    p.write_text("t,hand\n1.0,LEFT\n")
    with pytest.raises(ValueError, match="missing columns"):
        load_landmarks(p)


def test_raw_source_uses_unsmoothed_detections(tmp_path):
    path, _ = _record_v4(tmp_path)
    loaded = load_landmarks(sidecar_paths(path)["landmarks"])
    raw = series_from_landmarks(loaded.samples, ASPECT, source="raw")
    sm = series_from_landmarks(loaded.samples, ASPECT, source="smoothed")
    assert np.nanmax(raw["RIGHT_INDEX"].values["tip_y"]) >= np.nanmax(sm["RIGHT_INDEX"].values["tip_y"]) - 1e-9
    assert np.isfinite(raw["RIGHT_INDEX"].values["pip_bend_w"]).any()
    with pytest.raises(ValueError):
        series_from_landmarks(loaded.samples, ASPECT, source="video")


def test_observations_keep_world_landmarks_and_only_real_scores():
    def lm(x, vis=None):
        return SimpleNamespace(x=x, y=0.5, z=0.0, visibility=vis, presence=vis)

    result = SimpleNamespace(
        hand_landmarks=[[lm(0.1 * i / 21) for i in range(21)], [lm(0.5, vis=0.9) for _ in range(21)]],
        handedness=[[SimpleNamespace(category_name="Left", score=0.9)], [SimpleNamespace(category_name="Right",
                                                                                         score=0.8)]],
        hand_world_landmarks=[[lm(0.01) for _ in range(21)], [lm(0.02) for _ in range(21)]],
    )
    a, b = observations_from_result(result, mirrored_input=True)
    assert a.world_landmarks is not None and a.world_landmarks.shape == (21, 3)
    assert a.visibility is None and a.presence is None  # not provided → not invented
    assert b.visibility is not None and b.visibility[0] == pytest.approx(0.9)
    no_world = SimpleNamespace(hand_landmarks=result.hand_landmarks[:1], handedness=result.handedness[:1])
    assert observations_from_result(no_world, mirrored_input=False)[0].world_landmarks is None


def test_landmark_benchmark_script(tmp_path):
    # Contacts aligned with the recorded tap of RIGHT_INDEX (frames 30–55 at 1000 + n·DT).
    path, _ = _record_v4(tmp_path, keys=[("J", True, 1000.0 + 34 * DT), ("J", False, 1000.0 + 44 * DT)])
    out = subprocess.run([sys.executable, "scripts/benchmark_landmarks.py", str(path)],
                         capture_output=True, text=True, check=True).stdout
    assert "detector presses vs contacts" in out and "dip_bend:vel:cm" in out
    assert "detector releases vs key releases" in out and "spurious" in out
    old = tmp_path / "old.csv"
    old.write_text(",".join(COLUMNS[:18]) + "\n")
    res = subprocess.run([sys.executable, "scripts/benchmark_landmarks.py", str(old)], capture_output=True, text=True, check=False)
    assert res.returncode == 1 and "before v4" in res.stdout


def test_meta_is_valid_json_with_conventions(tmp_path):
    path, _ = _record_v4(tmp_path)
    meta = json.loads(sidecar_paths(path)["meta"].read_text())
    assert meta["format"] == "handpiano-trace" and "perf_counter" in meta["clock"]
    assert [s["name"] for s in meta["steps"]][:3] == ["prepare", "rest", "lift"]


def test_landmark_features_match_press_features_for_shared_names():
    hand = make_hand(Handedness.RIGHT, (0.5, 0.6))
    feats = landmark_features(hand.landmarks.astype(np.float64), FingerId.RIGHT_MIDDLE.finger, ASPECT)
    from handpiano.interaction.press_features import press_features
    base = press_features(hand.landmarks, FingerId.RIGHT_MIDDLE.finger, ASPECT)
    assert base is not None
    for k, v in base.items():
        assert feats[k] == pytest.approx(v)
    assert np.isnan(feats["pip_bend_w"])  # no world landmarks given


def test_physical_keys_are_labels_while_recording_and_notes_otherwise(tmp_path):
    from PySide6.QtCore import QEvent, Qt
    from PySide6.QtGui import QKeyEvent
    from PySide6.QtWidgets import QApplication

    from handpiano.app.application import Application
    from handpiano.app.config import AppConfig
    from handpiano.audio.engine import AudioEngine
    from handpiano.camera.camera_config import CameraSettings
    from handpiano.ui.main_window import MainWindow

    QApplication.instance() or QApplication([])
    window = MainWindow(CameraSettings())
    engine = AudioEngine(sounds_dir=tmp_path / "no_sounds")  # silent (NullOutput), nothing loaded
    app = Application(AppConfig(), window, audio=engine)

    def key(kind, text, repeat=False):
        code = getattr(Qt.Key, f"Key_{text.upper()}")
        return QKeyEvent(kind, code, Qt.KeyboardModifier.NoModifier, text, repeat)

    def notes_sent():
        return [c for c in engine.mixer.pending if c[0] in ("on", "off")]

    # Not recording: F plays a note (Free Piano) and nothing is logged.
    assert app.eventFilter(window, key(QEvent.Type.KeyPress, "f")) is True
    app.eventFilter(window, key(QEvent.Type.KeyRelease, "f"))
    assert [c[0] for c in notes_sent()] == ["on", "off"]
    engine.mixer._commands.clear()

    app.request_trace(tmp_path / "k.csv", "keys")
    trace = app.trace
    assert trace is not None
    assert app.eventFilter(window, key(QEvent.Type.KeyPress, "f")) is True  # consumed
    assert app.eventFilter(window.camera_view, key(QEvent.Type.KeyPress, "f")) is True  # same event, other widget
    assert app.eventFilter(window, key(QEvent.Type.KeyPress, "f", repeat=True)) is True  # auto-repeat
    app.eventFilter(window, key(QEvent.Type.KeyRelease, "f"))
    app.eventFilter(window, key(QEvent.Type.KeyPress, "q"))  # outside the map: logged, not consumed
    assert [(e.key, e.down) for e in trace._keys] == [("F", True), ("F", False), ("Q", True)]
    assert notes_sent() == []  # while recording, keys are labels only: no sound
    app.trace = None
    app.shutdown()
    window.close()


def test_keys_protocol_warns_when_no_key_reaches_the_window(tmp_path):
    from PySide6.QtWidgets import QApplication

    from handpiano.app.application import Application
    from handpiano.app.config import AppConfig
    from handpiano.app.pipeline import PipelineState
    from handpiano.audio.engine import AudioEngine
    from handpiano.camera.camera_config import CameraSettings
    from handpiano.ui.main_window import MainWindow

    QApplication.instance() or QApplication([])
    window = MainWindow(CameraSettings())
    app = Application(AppConfig(), window, audio=AudioEngine(sounds_dir=tmp_path / "none"))
    app.request_trace(tmp_path / "k.csv", "keys")
    window.isActiveWindow = lambda: True  # focus is checked separately (test_trace_keys.py)

    class RunningPipeline:
        state = PipelineState.RUNNING
        frame_listener = None

    steps = app.trace.steps
    # Jump to the first cue of the first key-tap step, plus a little.
    first_tap = next(i for i, s in enumerate(steps) if s.name.startswith("key_taps"))
    offset = sum(s.duration_s for s in steps[:first_tap]) + steps[first_tap].cue_times()[0] + 0.5
    app._trace_started_at = time.monotonic() - offset
    app._update_trace(RunningPipeline())  # marks the first cue
    app._cue_shown_at = None
    app._update_trace(RunningPipeline())
    assert window.camera_view._banner[0] == "⚠ NO LLEGA NINGUNA TECLA"
    app.trace.record_key("J", True)
    app._update_trace(RunningPipeline())
    assert window.camera_view._banner[0].startswith("Pulsa la tecla")
    app.trace = None
    app.shutdown()
    window.close()
