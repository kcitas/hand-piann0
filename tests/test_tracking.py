from types import SimpleNamespace

import numpy as np
import pytest

from handpiano.tracking.finger_tracker import FingerTracker, FingerTrackerConfig, TrackStatus
from handpiano.tracking.hand_identity import HandIdentityResolver, IdentityConfig
from handpiano.tracking.landmarks import (
    HAND_CONNECTIONS,
    PLAYING_FINGERS,
    Finger,
    FingerId,
    Handedness,
    Landmark,
    observations_from_result,
)
from handpiano.tracking.smoothing import SmoothingConfig, SmoothingKind
from tests.fakes import make_hand, shifted

L, R = Handedness.LEFT, Handedness.RIGHT
DT = 1 / 30


# --- landmark model -----------------------------------------------------------


def test_eight_playing_fingers_without_thumbs():
    assert len(PLAYING_FINGERS) == 8
    assert {f.hand for f in PLAYING_FINGERS} == {L, R}
    assert Finger.THUMB not in {f.finger for f in PLAYING_FINGERS}


def test_finger_tip_indices_match_mediapipe_topology():
    assert [f.tip for f in Finger] == [4, 8, 12, 16, 20]
    assert FingerId.RIGHT_RING.finger.tip == Landmark.RING_TIP == 16


def test_hand_connections_form_a_tree_over_21_landmarks():
    nodes = {i for pair in HAND_CONNECTIONS for i in pair}
    assert nodes == set(range(21))


def _mp_result(*hands):
    def lm(x, y, z):
        return SimpleNamespace(x=x, y=y, z=z)

    return SimpleNamespace(
        hand_landmarks=[[lm(*p) for p in pts] for pts, _ in hands],
        handedness=[cats for _, cats in hands],
    )


def test_converts_mediapipe_result():
    pts = [(i / 21, 0.5, 0.0) for i in range(21)]
    result = _mp_result((pts, [SimpleNamespace(category_name="Left", score=0.9)]))
    (obs,) = observations_from_result(result)
    assert obs.handedness is L
    assert obs.score == pytest.approx(0.9)
    assert obs.landmarks.shape == (21, 3)
    assert obs.landmarks[20, 0] == pytest.approx(20 / 21)


def test_conversion_skips_malformed_hands():
    short = [(0.5, 0.5, 0.0)] * 5
    good = [(0.5, 0.5, 0.0)] * 21
    result = _mp_result(
        (short, [SimpleNamespace(category_name="Left", score=0.9)]),
        (good, [SimpleNamespace(category_name="Unknown", score=0.9)]),
        (good, []),
    )
    assert observations_from_result(result) == []


# --- hand identity ------------------------------------------------------------


def test_two_hands_with_distinct_labels_are_trusted():
    r = HandIdentityResolver()
    left, right = make_hand(L, (0.3, 0.5)), make_hand(R, (0.7, 0.5))
    assigned = r.resolve([right, left], 0.0)
    assert assigned[L] is left and assigned[R] is right


def test_duplicate_labels_are_split_by_position():
    r = HandIdentityResolver()
    a, b = make_hand(R, (0.7, 0.5)), make_hand(R, (0.3, 0.5))
    assigned = r.resolve([a, b], 0.0)
    assert assigned[L] is b and assigned[R] is a


def test_single_hand_label_flicker_is_ignored():
    r = HandIdentityResolver(IdentityConfig(label_switch_frames=4))
    hand = make_hand(R, (0.6, 0.5))
    r.resolve([hand], 0.0)
    for k in range(1, 4):  # 3 flipped frames, below the threshold
        assigned = r.resolve([shifted(hand, label=L)], k * DT)
        assert R in assigned
    assert R in r.resolve([hand], 4 * DT)


def test_persistent_label_change_is_accepted():
    r = HandIdentityResolver(IdentityConfig(label_switch_frames=3))
    hand = make_hand(R, (0.6, 0.5))
    r.resolve([hand], 0.0)
    results = [r.resolve([shifted(hand, label=L)], k * DT) for k in range(1, 4)]
    assert L in results[-1]


def test_returning_hand_after_memory_uses_label():
    r = HandIdentityResolver(IdentityConfig(memory_s=0.3))
    r.resolve([make_hand(R, (0.6, 0.5))], 0.0)
    assert L in r.resolve([make_hand(L, (0.6, 0.5))], 1.0)


# --- finger tracker -----------------------------------------------------------


def _tracker(**overrides):
    return FingerTracker(FingerTrackerConfig(smoothing=SmoothingConfig(kind=SmoothingKind.NONE), **overrides))


def test_all_eight_fingers_tracked_with_two_hands():
    tracker = _tracker()
    state = tracker.update([make_hand(L, (0.3, 0.5)), make_hand(R, (0.7, 0.5))], 0.0)
    assert state.hands_tracked == 2
    assert all(state.fingers[f].status is TrackStatus.TRACKED for f in PLAYING_FINGERS)
    tip = state.fingers[FingerId.LEFT_INDEX].tip
    np.testing.assert_allclose(tip, make_hand(L, (0.3, 0.5)).landmarks[Landmark.INDEX_TIP], atol=1e-6)


def test_fingers_of_missing_hand_are_lost():
    state = _tracker().update([make_hand(R, (0.7, 0.5))], 0.0)
    assert state.fingers[FingerId.LEFT_PINKY].status is TrackStatus.LOST
    assert state.fingers[FingerId.LEFT_PINKY].tip is None
    assert state.fingers[FingerId.RIGHT_PINKY].status is TrackStatus.TRACKED


def test_short_dropout_is_held_then_lost():
    tracker = _tracker(lost_grace_s=0.1)
    hand = make_hand(R)
    tracker.update([hand], 0.0)
    held = tracker.update([], 0.05)
    assert held.hands[R].status is TrackStatus.HELD
    assert held.fingers[FingerId.RIGHT_INDEX].tip is not None
    assert held.fingers[FingerId.RIGHT_INDEX].velocity is None
    lost = tracker.update([], 0.2)
    assert lost.hands[R].status is TrackStatus.LOST


def test_velocity_of_moving_finger():
    tracker = _tracker()
    hand = make_hand(R)
    tracker.update([hand], 0.0)
    state = tracker.update([shifted(hand, dy=0.03)], DT)  # 0.9 heights per second downward
    v = state.fingers[FingerId.RIGHT_MIDDLE].velocity
    assert v is not None
    assert v[1] == pytest.approx(0.9, rel=1e-3)
    assert v[0] == pytest.approx(0.0, abs=1e-6)


def test_single_frame_teleport_is_rejected_as_outlier():
    tracker = _tracker(max_speed=8.0, max_consecutive_outliers=2)
    hand = make_hand(R, (0.6, 0.5))
    tracker.update([hand], 0.0)
    glitch = tracker.update([shifted(hand, dx=0.5)], DT)  # 15 widths/s
    assert glitch.hands[R].status is TrackStatus.HELD
    assert glitch.hands[R].rejected_outliers == 1
    back = tracker.update([hand], 2 * DT)
    assert back.hands[R].status is TrackStatus.TRACKED


def test_persistent_jump_is_eventually_accepted():
    # Speed is measured from the last ACCEPTED position, so it decays with time:
    # 0.4 units -> 12, 6, 4 units/s over 1, 2, 3 frames, all above max_speed=3.
    tracker = _tracker(max_speed=3.0, max_consecutive_outliers=2)
    hand = make_hand(R, (0.6, 0.5))
    tracker.update([hand], 0.0)
    moved = shifted(hand, dy=0.4)
    statuses = [tracker.update([moved], k * DT).hands[R].status for k in range(1, 5)]
    assert statuses[:2] == [TrackStatus.HELD, TrackStatus.HELD]
    assert statuses[2] is TrackStatus.TRACKED


def test_jitter_is_measured_once_enough_samples():
    tracker = FingerTracker(FingerTrackerConfig())
    rng = np.random.default_rng(0)
    hand = make_hand(R)
    state = None
    for k in range(40):
        noisy = shifted(hand, dx=float(rng.normal(0, 0.003)), dy=float(rng.normal(0, 0.003)))
        state = tracker.update([noisy], k * DT)
    assert state is not None
    jitter = state.fingers[FingerId.RIGHT_INDEX].jitter
    assert jitter is not None and 0 < jitter < 0.01
