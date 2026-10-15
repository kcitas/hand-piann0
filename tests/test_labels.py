"""Independent labels, landmark features and the benchmark additions (no camera, no downloads)."""

import math

import numpy as np
import pytest

from handpiano.interaction.benchmark import (
    Event,
    Series,
    events_from_labels,
    hand_of_finger,
    match_detections,
    parse_variant,
    score_signals,
    session_folds,
    variants_for,
)
from handpiano.interaction.labels import (
    KeyEvent,
    Label,
    labels_from_keys,
    load_labels,
    write_labels,
)
from handpiano.interaction.landmark_features import bend_deg, landmark_features
from handpiano.tracking.landmarks import Finger, Handedness
from tests.fakes import make_hand

# --- label format ---------------------------------------------------------------------------


@pytest.mark.parametrize("label, message", [
    (Label(1.0, None, "RIGHT_INDEX", "contact", "detector"), "not independent"),
    (Label(1.0, None, "RIGHT_INDEX", "contact", "cue"), "not independent"),
    (Label(1.0, 2.0, "RIGHT_INDEX", "contact", "keyboard"), "point label"),
    (Label(2.0, 1.0, "RIGHT_INDEX", "rest", "manual"), "interval"),
    (Label(1.0, None, "RIGHT_INDEX", "rest", "manual"), "interval"),
    (Label(1.0, None, "RIGHT_THUMB", "contact", "keyboard"), "target"),
    (Label(1.0, None, "RIGHT_INDEX", "pressed", "keyboard"), "unknown label"),
    (Label(1.0, None, "RIGHT_INDEX", "contact", "tracking"), "only supports"),
    (Label(1.0, None, "RIGHT_INDEX", "contact", "keyboard", "sure"), "confidence"),
    (Label(float("nan"), None, "RIGHT_INDEX", "contact", "keyboard"), "finite"),
])
def test_invalid_labels_are_rejected(label, message):
    with pytest.raises(ValueError, match=message):
        label.validate()


def test_labels_round_trip_and_never_overwrite(tmp_path):
    labels = [
        Label(3.0, None, "LEFT_RING", "contact", "video_review", "uncertain", "blurred"),
        Label(1.0, 2.5, "ALL", "rest", "manual"),
        Label(4.0, 4.5, "LEFT", "insufficient_tracking", "tracking"),
        Label(3.0, 3.2, "LEFT_PINKY", "neighbour_motion", "video_review"),
    ]
    path = tmp_path / "x.labels.csv"
    write_labels(path, labels)
    assert load_labels(path) == sorted(labels, key=lambda lb: (lb.t_start, lb.target, lb.label))
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        write_labels(path, labels[:1])
    assert path.read_bytes() == before
    with pytest.raises(ValueError):
        write_labels(tmp_path / "bad.csv", [Label(1.0, None, "RIGHT_INDEX", "contact", "detector")])
    assert not (tmp_path / "bad.csv").exists()


def test_load_labels_names_the_bad_line(tmp_path):
    path = tmp_path / "l.csv"
    path.write_text("t_start,t_end,target,label,source,confidence,note\n1.0,,RIGHT_INDEX,contact,keyboard,,\n"
                    "2.0,,RIGHT_INDEX,contact,cue,confirmed,\n")
    with pytest.raises(ValueError, match=":3: source 'cue'"):
        load_labels(path)


def test_keys_to_labels_handles_unknown_and_held_keys():
    events = [KeyEvent(1.0, "F", True), KeyEvent(1.2, "F", False), KeyEvent(2.0, "F", True),
              KeyEvent(2.05, "F", True), KeyEvent(3.0, "Z", True)]
    labels = labels_from_keys(events, end_t=5.0)
    assert [(lb.label, lb.target, lb.t_start) for lb in labels] == [
        ("contact", "LEFT_INDEX", 1.0), ("release", "LEFT_INDEX", 1.2), ("contact", "LEFT_INDEX", 2.0),
        ("ambiguous", "ALL", 3.0)]
    assert labels[-1].t_end == 5.0 and labels[-1].confidence == "uncertain"
    for lb in labels:
        lb.validate()


# --- landmark features ----------------------------------------------------------------------


def test_bend_angles():
    a, b = np.array([0.0, 0, 0]), np.array([1.0, 0, 0])
    assert bend_deg(a, b, np.array([2.0, 0, 0])) == pytest.approx(0.0)
    assert bend_deg(a, b, np.array([1.0, 1, 0])) == pytest.approx(90.0)
    assert math.isnan(bend_deg(a, a, b))


def test_landmark_features_are_invariant_to_hand_position_and_size():
    hand = make_hand(Handedness.RIGHT, (0.5, 0.6)).landmarks.astype(np.float64)
    moved = hand.copy()
    moved[:, :2] = (moved[:, :2] - hand[0, :2]) * 1.5 + (0.3, 0.4)
    moved[:, 2] *= 1.5
    a = landmark_features(hand, Finger.INDEX, 1.0, world=hand * 0.1)
    b = landmark_features(moved, Finger.INDEX, 1.0, world=moved * 0.2)
    for k in a:
        assert a[k] == pytest.approx(b[k], abs=1e-6), k


def test_landmark_features_grow_when_the_finger_curls_and_handle_bad_input():
    hand = make_hand(Handedness.RIGHT, (0.5, 0.6)).landmarks.astype(np.float64)
    curled = hand.copy()
    tip = Finger.MIDDLE.tip
    curled[tip] = (curled[tip] + curled[Finger.MIDDLE.base]) / 2 + (0.0, 0.0, -0.02)
    a = landmark_features(hand, Finger.MIDDLE, 1.0, world=hand)
    b = landmark_features(curled, Finger.MIDDLE, 1.0, world=curled)
    assert b["dip_bend"] > a["dip_bend"] and b["palm_dist"] > a["palm_dist"] and b["dip_bend_w"] > a["dip_bend_w"]
    broken = hand.copy()
    broken[3, 0] = np.inf
    assert all(math.isnan(v) for v in landmark_features(broken, Finger.MIDDLE, 1.0).values())
    assert all(math.isnan(v) for v in landmark_features(None, Finger.MIDDLE, 1.0).values())


# --- benchmark additions --------------------------------------------------------------------


def test_variant_parsing():
    assert parse_variant("flex:vel:cm") == ("flex", frozenset({"vel", "cm"}))
    assert variants_for(["a"]) == ("a", "a:cm", "a:vel", "a:vel:cm")
    with pytest.raises(ValueError):
        parse_variant("flex:acc")


def test_velocity_common_mode_variant_scores_a_separable_finger():
    t = np.arange(0, 20, 1 / 30)
    times = [2.0, 6.0, 10.0, 14.0]

    def sig(f):
        return np.array([0.5 if f == "RIGHT_INDEX" and any(e <= x < e + 0.3 for e in times) else 0.0 for x in t])

    fingers = ("RIGHT_INDEX", "RIGHT_MIDDLE", "RIGHT_RING", "RIGHT_PINKY")
    series = {f: Series(t, {"dip_bend": sig(f)}) for f in fingers}
    scores = score_signals(series, [Event("RIGHT_INDEX", e) for e in times], [17.0], hand_of_finger,
                           variants=("dip_bend:vel:cm", "missing_feature"))
    pooled = {s.variant: s for s in scores if s.finger == "ALL"}
    assert pooled["dip_bend:vel:cm"].auc_neighbours == 1.0
    assert pooled["missing_feature"].events == 0  # a feature absent from the data is reported, not invented


def test_events_from_labels_use_only_confirmed_contacts_and_quiet_background():
    labels = [
        Label(5.0, None, "RIGHT_INDEX", "contact", "keyboard"),
        Label(5.2, None, "RIGHT_INDEX", "release", "keyboard"),
        Label(9.0, None, "LEFT_RING", "contact", "video_review", "uncertain"),
        Label(12.0, 14.0, "ALL", "ambiguous", "manual"),
    ]
    events, background = events_from_labels(labels, 0.0, 20.0)
    assert events == [Event("RIGHT_INDEX", 5.0)]
    assert background and all(abs(t - 5.0) > 1.0 and abs(t - 9.0) > 1.0 for t in background)
    assert not any(11.0 <= t <= 15.0 for t in background)


def test_match_detections_is_one_to_one_with_offsets():
    ref = [Event("R", 1.0), Event("R", 2.0), Event("L", 1.0)]
    det = [Event("R", 1.1), Event("R", 1.15), Event("R", 5.0), Event("L", 0.5)]
    m = {x.finger: x for x in match_detections(ref, det)}
    assert (m["R"].hits, m["R"].misses, m["R"].false_presses) == (1, 1, 2)
    assert m["R"].offsets == pytest.approx((0.1,))
    assert (m["L"].hits, m["L"].false_presses) == (0, 1)
    assert (m["ALL"].reference, m["ALL"].detected, m["ALL"].hits) == (3, 4, 1)


def test_session_folds_never_share_a_session():
    folds = session_folds(["s1", "s2", "s1", "s3"])
    assert len(folds) == 3
    for train, held in folds:
        assert held not in train and len(train) == 2
    with pytest.raises(ValueError):
        session_folds(["only"])
