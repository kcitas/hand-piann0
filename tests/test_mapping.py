import json

import numpy as np
import pytest

from handpiano.instrument.keyboard import KeyboardLayout
from handpiano.mapping.calibration import (
    CalibrationError,
    CalibrationStore,
    CameraKey,
    PianoCalibration,
    validate_corners,
)
from handpiano.mapping.calibration_session import CalibrationSession, SessionConfig
from handpiano.mapping.homography import DegenerateHomographyError, apply_homography, homography_from_points
from handpiano.mapping.key_mapper import KeyMapper
from handpiano.tracking.finger_tracker import FingerTracker
from handpiano.tracking.landmarks import PLAYING_FINGERS, FingerId, Handedness, Landmark
from tests.fakes import make_hand

LAYOUT = KeyboardLayout.from_octaves("C4", 2)
W, H = LAYOUT.size
CAM = CameraKey(0, 1280, 720, True)
# An axis-aligned rectangle in the camera image.
RECT = [(0.2, 0.5), (0.8, 0.5), (0.8, 0.9), (0.2, 0.9)]
# A trapezoid, as a flat keyboard seen in perspective (far edge shorter).
TRAPEZOID = [(0.3, 0.5), (0.7, 0.5), (0.9, 0.9), (0.1, 0.9)]


def _cal(points=RECT):
    return PianoCalibration.from_corners(points, LAYOUT.size, CAM)


# --- homography -----------------------------------------------------------------


def test_homography_maps_the_four_reference_points_exactly():
    src = np.array(TRAPEZOID)
    dst = np.array([[0, 0], [W, 0], [W, H], [0, H]])
    h = homography_from_points(src, dst)
    np.testing.assert_allclose(apply_homography(h, src), dst, atol=1e-9)


def test_rectangle_center_maps_to_piano_center():
    cal = _cal(RECT)
    np.testing.assert_allclose(cal.to_piano([(0.5, 0.7)]), [[W / 2, H / 2]], atol=1e-9)


def test_perspective_center_is_the_diagonal_intersection_not_the_average():
    """Under perspective the keyboard centre is where the diagonals cross; an affine fit would get this wrong."""
    cal = _cal(TRAPEZOID)
    p1, p2, p3, p4 = (np.array(p) for p in TRAPEZOID)
    # Intersection of diagonals P1–P3 and P2–P4.
    d1, d2 = p3 - p1, p4 - p2
    def cross2(a, b):
        return a[0] * b[1] - a[1] * b[0]

    t = cross2(p2 - p1, d2) / cross2(d1, d2)
    center = p1 + t * d1
    np.testing.assert_allclose(cal.to_piano([center]), [[W / 2, H / 2]], atol=1e-9)
    assert not np.allclose(center, np.mean(TRAPEZOID, axis=0))


def test_round_trip_camera_piano_camera():
    cal = _cal(TRAPEZOID)
    pts = np.random.default_rng(0).uniform([0.15, 0.55], [0.85, 0.85], (50, 2))
    np.testing.assert_allclose(cal.to_camera(cal.to_piano(pts)), pts, atol=1e-9)


def test_opposite_corners_map_to_opposite_corners():
    cal = _cal(TRAPEZOID)
    np.testing.assert_allclose(cal.to_piano([TRAPEZOID[0], TRAPEZOID[2]]), [[0, 0], [W, H]], atol=1e-9)
    np.testing.assert_allclose(cal.to_piano([TRAPEZOID[1], TRAPEZOID[3]]), [[W, 0], [0, H]], atol=1e-9)


def test_degenerate_homography_rejected():
    collinear = np.array([[0, 0], [1, 0], [2, 0], [3, 0]])
    with pytest.raises(DegenerateHomographyError):
        homography_from_points(collinear, np.array([[0, 0], [1, 0], [1, 1], [0, 1]]))
    with pytest.raises(DegenerateHomographyError):
        homography_from_points(np.zeros((3, 2)), np.zeros((3, 2)))


# --- calibration validation ------------------------------------------------------


def test_valid_corners_accepted():
    validate_corners(np.array(RECT))
    validate_corners(np.array(TRAPEZOID))


@pytest.mark.parametrize(
    ("points", "fragment"),
    [
        (RECT[:3], "exactamente 4"),
        ([(0.2, 0.5), (0.21, 0.5), (0.8, 0.9), (0.2, 0.9)], "demasiado cerca"),
        (list(reversed(RECT)), "sentido contrario"),
        ([(0.2, 0.5), (0.8, 0.9), (0.8, 0.5), (0.2, 0.9)], "convexo"),  # bow-tie: edges cross
        ([(0.2, 0.5), (0.5, 0.6), (0.8, 0.7), (0.2, 0.9)], "convexo"),  # three corners aligned
        ([(0.4, 0.5), (0.5, 0.5), (0.5, 0.56), (0.4, 0.56)], "demasiado pequeña"),
        ([(0.2, float("nan")), (0.8, 0.5), (0.8, 0.9), (0.2, 0.9)], "posición válida"),
    ],
)
def test_invalid_corners_rejected_with_reason(points, fragment):
    with pytest.raises(CalibrationError) as err:
        PianoCalibration.from_corners(points, LAYOUT.size, CAM)
    assert fragment in err.value.user_message


def test_calibration_persists_and_is_invalidated_for_other_camera(tmp_path):
    store = CalibrationStore(tmp_path / "cal.json")
    assert store.load(CAM, LAYOUT.size)[0] is None
    store.save(_cal(TRAPEZOID))
    loaded, reason = store.load(CAM, LAYOUT.size)
    assert loaded is not None and "cargada" in reason
    np.testing.assert_allclose(loaded.to_piano([TRAPEZOID[2]]), [[W, H]], atol=1e-9)
    for other in (CameraKey(1, 1280, 720, True), CameraKey(0, 1920, 1080, True), CameraKey(0, 1280, 720, False)):
        cal, reason = store.load(other, LAYOUT.size)
        assert cal is None and "otra cámara" in reason
    store.clear()
    assert store.load(CAM, LAYOUT.size)[0] is None


def test_corrupt_or_invalid_saved_calibration_is_ignored(tmp_path):
    path = tmp_path / "cal.json"
    store = CalibrationStore(path)
    path.write_text("{not json")
    assert store.load(CAM, LAYOUT.size) == (None, "la calibración guardada está dañada")
    data = json.loads(_cal().to_json())
    data["camera_points"] = [[0.2, 0.5]] * 4  # degenerate
    path.write_text(json.dumps(data))
    assert store.load(CAM, LAYOUT.size)[0] is None


def test_calibration_saves_only_corners_not_images(tmp_path):
    data = json.loads(_cal().to_json())
    assert set(data) == {"version", "camera_points", "camera", "created_at"}


def test_same_corners_for_another_keyboard_size():
    one_octave = KeyboardLayout.from_octaves("C4", 1)
    cal = _cal(RECT).with_piano_size(one_octave.size)
    np.testing.assert_allclose(cal.to_piano([RECT[2]]), [one_octave.size], atol=1e-9)


# --- calibration session --------------------------------------------------------


def test_dwell_captures_mean_position_after_holding_still():
    s = CalibrationSession(SessionConfig(dwell_s=1.0, dwell_radius=0.01))
    captured = None
    t = 0.0
    rng = np.random.default_rng(0)
    while captured is None and t < 3:
        captured = s.update(t, (0.3 + rng.normal(0, 0.001), 0.4 + rng.normal(0, 0.001)))
        t += 1 / 24
    assert captured is not None
    assert t == pytest.approx(1.0 + 1 / 24, abs=0.05)
    assert captured == pytest.approx((0.3, 0.4), abs=0.002)
    assert s.step == 2 and s.corner_name == "superior derecha"


def test_moving_finger_does_not_capture():
    s = CalibrationSession(SessionConfig(dwell_s=1.0, dwell_radius=0.01))
    for k in range(100):
        assert s.update(k / 24, (0.1 + k * 0.005, 0.5)) is None
    assert s.points == []


def test_must_move_away_before_next_corner():
    s = CalibrationSession(SessionConfig(dwell_s=0.5, dwell_radius=0.01, min_move=0.05))
    t = 0.0
    while not s.points:
        s.update(t, (0.3, 0.4))
        t += 0.04
    for _ in range(50):  # staying at the same place never captures corner 2
        assert s.update(t, (0.3, 0.4)) is None
        t += 0.04
    assert s.waiting_to_move_away
    while len(s.points) < 2:
        s.update(t, (0.7, 0.4))
        t += 0.04
    assert s.points[1] == pytest.approx((0.7, 0.4))


def test_capture_now_undo_and_lost_finger():
    s = CalibrationSession()
    assert s.capture_now(None) is None
    for p in RECT:
        s.capture_now(p)
    assert s.done and s.corner_name is None
    s.undo()
    assert len(s.points) == 3 and not s.done
    s.update(0.0, (0.2, 0.9))
    s.update(0.5, None)  # finger lost resets the dwell
    assert s.dwell_progress(0.6) == 0.0


# --- key mapper -----------------------------------------------------------------


def _state_with_tips(tips: dict[FingerId, tuple[float, float]]):
    """A TrackingState whose fingertips are at the given camera positions."""
    hands = []
    for hand in Handedness:
        mine = {f: p for f, p in tips.items() if f.hand is hand}
        if not mine:
            continue
        obs = make_hand(hand, (0.5, 0.5))
        pts = obs.landmarks.copy()
        for finger, (x, y) in mine.items():
            pts[finger.finger.tip, :2] = (x, y)
        hands.append(type(obs)(handedness=hand, score=obs.score, landmarks=pts))
    return FingerTracker().update(hands, 0.0)


def _camera_point_for(cal, key):
    return tuple(cal.to_camera([key.center])[0])


def test_uncalibrated_maps_nothing():
    mapping = KeyMapper(LAYOUT).map(_state_with_tips({FingerId.RIGHT_INDEX: (0.5, 0.7)}))
    assert not mapping.calibrated
    fk = mapping.fingers[FingerId.RIGHT_INDEX]
    assert fk.has_position and fk.key is None and fk.piano_xy is None and fk.inside is None
    assert mapping.inside is None and mapping.outside is None
    assert mapping.with_position >= 1


def test_eight_fingers_two_hands_each_on_its_own_key():
    cal = _cal(TRAPEZOID)
    targets = dict(zip(PLAYING_FINGERS, ["C4", "E4", "G4", "B4", "D5", "F#5", "A5", "C#4"], strict=True))
    by_name = {k.name: k for k in LAYOUT.keys}
    tips = {f: _camera_point_for(cal, by_name[n]) for f, n in targets.items()}
    mapping = KeyMapper(LAYOUT, cal).map(_state_with_tips(tips))
    assert mapping.calibrated and mapping.inside == 8 and mapping.outside == 0
    for finger, name in targets.items():
        fk = mapping.fingers[finger]
        assert fk.key.name == name
        assert fk.distance_to_center == pytest.approx(0.0, abs=1e-6)
    assert len(mapping.keys_under) == 8


def test_two_fingers_on_the_same_key_do_not_create_a_second_note():
    cal = _cal(RECT)
    c4 = LAYOUT.key_by_midi(60)
    p = _camera_point_for(cal, c4)
    mapping = KeyMapper(LAYOUT, cal).map(_state_with_tips({FingerId.LEFT_INDEX: p, FingerId.RIGHT_INDEX: p}))
    assert mapping.keys_under == {60: (FingerId.LEFT_INDEX, FingerId.RIGHT_INDEX)}


def test_fingers_outside_the_keyboard():
    cal = _cal(RECT)
    mapping = KeyMapper(LAYOUT, cal).map(
        _state_with_tips({FingerId.LEFT_INDEX: (0.1, 0.2), FingerId.RIGHT_INDEX: (0.5, 0.7)})
    )
    left, right = mapping.fingers[FingerId.LEFT_INDEX], mapping.fingers[FingerId.RIGHT_INDEX]
    assert left.inside is False and left.key is None and left.distance_to_center is None
    assert right.inside is True
    # Fingers without a position are neither inside nor outside.
    assert mapping.fingers[FingerId.LEFT_RING].inside in (None, False)


def test_same_finger_maps_to_different_keys_as_it_moves():
    cal = _cal(RECT)
    mapper = KeyMapper(LAYOUT, cal)
    names = []
    for key in (LAYOUT.key_by_midi(60), LAYOUT.key_by_midi(67)):
        names.append(mapper.map(_state_with_tips({FingerId.LEFT_INDEX: _camera_point_for(cal, key)})).fingers[
            FingerId.LEFT_INDEX].key.name)
    assert names == ["C4", "G4"]


def test_black_key_priority_through_the_full_chain():
    cal = _cal(TRAPEZOID)
    c_sharp = LAYOUT.key_by_midi(61)
    # Inside C#'s rectangle but over the area it shares with C.
    point = cal.to_camera([(c_sharp.x + 0.05, 1.0)])[0]
    fk = KeyMapper(LAYOUT, cal).map(_state_with_tips({FingerId.RIGHT_INDEX: tuple(point)})).fingers[FingerId.RIGHT_INDEX]
    assert fk.key.name == "C#4"


def test_calibration_can_be_swapped_and_removed():
    mapper = KeyMapper(LAYOUT)
    state = _state_with_tips({FingerId.RIGHT_INDEX: (0.5, 0.7)})
    assert not mapper.map(state).calibrated
    mapper.calibration = _cal(RECT)
    assert mapper.map(state).fingers[FingerId.RIGHT_INDEX].key is not None
    mapper.calibration = None
    assert mapper.map(state).fingers[FingerId.RIGHT_INDEX].key is None


def test_mapping_uses_the_smoothed_tracker_tip():
    state = _state_with_tips({FingerId.RIGHT_INDEX: (0.5, 0.7)})
    tip = state.fingers[FingerId.RIGHT_INDEX].tip
    fk = KeyMapper(LAYOUT, _cal(RECT)).map(state).fingers[FingerId.RIGHT_INDEX]
    assert fk.camera_xy == pytest.approx((float(tip[0]), float(tip[1])))
    assert Landmark.INDEX_TIP == FingerId.RIGHT_INDEX.finger.tip


def test_corner_too_close_to_a_previous_one_is_rejected_immediately():
    """Regression from the first real calibration: corners 2 and 3 were captured ~0.01 apart."""
    s = CalibrationSession()
    assert s.capture_now((0.222, 0.388)) is not None
    assert s.capture_now((0.288, 0.415)) is not None
    assert s.capture_now((0.283, 0.426)) is None
    assert len(s.points) == 2
    assert "Demasiado cerca de la esquina 2" in s.rejection
    assert s.capture_now((0.8, 0.9)) is not None and s.rejection is None


def test_dwell_near_any_previous_corner_never_captures():
    s = CalibrationSession(SessionConfig(dwell_s=0.3, dwell_radius=0.01, min_move=0.05))
    s.capture_now((0.2, 0.5))
    s.capture_now((0.8, 0.5))
    t = 0.0
    for _ in range(60):  # resting back at corner 1 for 2.4 s
        assert s.update(t, (0.2, 0.5)) is None
        t += 0.04
    assert len(s.points) == 2


def test_tests_never_touch_the_real_user_calibration(tmp_path):
    """conftest.py redirects the default store; a default CalibrationStore must live under tmp."""
    from handpiano.mapping.calibration import default_store_path

    store = CalibrationStore()
    assert store.path == default_store_path()
    assert str(store.path).startswith(str(tmp_path))
    assert "Application Support" not in str(store.path)
