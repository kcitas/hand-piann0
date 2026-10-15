"""Press detection: per-finger state machine and the frame-level detector, on synthetic sequences."""

import numpy as np
import pytest

from handpiano.instrument.keyboard import KeyboardLayout
from handpiano.interaction.finger_presses import PressDetector
from handpiano.interaction.press_detector import FingerPressMachine, PressConfig, PressInput, PressPhase
from handpiano.interaction.press_features import hand_scale, press_features
from handpiano.interaction.stable_keys import StableKeys
from handpiano.mapping.calibration import CameraKey, PianoCalibration
from handpiano.mapping.key_mapper import KeyMapper
from handpiano.tracking.finger_tracker import FingerTracker, FingerTrackerConfig
from handpiano.tracking.landmarks import Finger, FingerId, Handedness, HandObservation
from handpiano.tracking.smoothing import SmoothingConfig, SmoothingKind
from tests.fakes import make_hand

LAYOUT = KeyboardLayout.from_octaves("C4", 2)
C4, CS4, D4, E4 = (LAYOUT.key_by_midi(m) for m in (60, 61, 62, 64))
FPS = 24
DT = 1 / FPS
CFG = PressConfig()


def run(values, keys=None, tracked=None, discontinuity=None, hand_speed=None, calibrated=True, config=CFG):
    """Feeds a sequence to one machine; returns (events, machine)."""
    m = FingerPressMachine(FingerId.RIGHT_INDEX, config)
    n = len(values)
    keys = keys if keys is not None else [C4] * n
    tracked = tracked if tracked is not None else [True] * n
    discontinuity = discontinuity if discontinuity is not None else [False] * n
    hand_speed = hand_speed if hand_speed is not None else [0.0] * n
    events = []
    for i in range(n):
        events += m.update(PressInput(i * DT, values[i], keys[i], tracked[i], discontinuity[i], hand_speed[i], calibrated))
    return events, m


def tap(rest=10, down=3, hold=6, up=3, after=10, depth=0.3):
    """Rest at 0, press to ``depth`` in ``down`` frames, hold, come back up."""
    return ([0.0] * rest + list(np.linspace(depth / down, depth, down)) + [depth] * hold
            + list(np.linspace(depth * 2 / 3, 0.0, up)) + [0.0] * after)


def kinds(events):
    return [(e.kind, e.reason) for e in events]


# 1. still finger -------------------------------------------------------------------------


def test_still_finger_with_jitter_never_presses():
    rng = np.random.default_rng(0)
    values = list(rng.normal(0, 0.01, 24 * 30))  # 30 s of tracking noise
    events, m = run(values)
    assert [e for e in events if e.kind == "press"] == []
    assert m.phase is PressPhase.IDLE


def test_slow_posture_drift_is_absorbed_by_the_baseline():
    values = list(np.linspace(0, 0.5, 24 * 10))  # half a hand size over 10 s: drift, not a tap
    events, _ = run(values)
    assert [e for e in events if e.kind == "press"] == []


# 2. fast press ----------------------------------------------------------------------------


def test_fast_tap_gives_exactly_one_press_and_one_release():
    events, m = run(tap())
    assert kinds(events) == [("press", "ok"), ("release", "ok")]
    assert events[0].key is C4 and events[1].key is C4
    assert m.phase is PressPhase.IDLE


def test_press_latency_is_two_frames_after_crossing_the_threshold():
    values = tap(down=3, depth=0.3)
    events, _ = run(values)
    first_above = next(i for i, v in enumerate(values) if v > CFG.press_on)
    assert events[0].t == pytest.approx((first_above + CFG.min_press_frames - 1) * DT)


# 3. release and press again; holding does not repeat ------------------------------------------


def test_holding_still_does_not_repeat_and_a_new_tap_needs_a_release():
    values = tap(hold=24 * 5, after=6) + tap(rest=0)
    events, _ = run(values)
    assert [k for k, _ in kinds(events)] == ["press", "release", "press", "release"]


# 4. tremor around the thresholds ----------------------------------------------------------------


def test_tremor_around_press_threshold_never_presses():
    values = [0.0] * 10 + [0.10, 0.14] * 40  # flickers across press_on every frame
    events, _ = run(values)
    assert [e for e in events if e.kind == "press"] == []


def test_tremor_around_release_threshold_while_pressed_does_not_chatter():
    values = [0.0] * 10 + [0.1, 0.2, 0.3, 0.3] + [0.05, 0.08] * 30 + [0.0] * 6
    events, _ = run(values)
    assert [k for k, _ in kinds(events)] == ["press", "release"]


# 5. boundary between two keys ----------------------------------------------------------------------


def test_key_is_latched_before_the_tap_moves_the_fingertip():
    values = tap(rest=10, down=3)
    keys = [C4] * 10 + [D4] * (len(values) - 10)  # the tap motion itself drags the tip onto D4
    events, _ = run(values, keys)
    assert events[0].kind == "press" and events[0].key is C4
    assert events[1].key is C4  # released on the same key it pressed


def test_stable_keys_stop_boundary_flicker_but_follow_real_moves():
    sk = StableKeys(LAYOUT, margin=0.08)
    finger = FingerId.RIGHT_INDEX
    # Front of the keyboard, oscillating ±0.03 around the E|F boundary (x = 3).
    keys = [sk.update(finger, (3.0 + (0.03 if i % 2 else -0.03), 5.5)).name for i in range(20)]
    assert set(keys) == {"E4"}
    assert sk.update(finger, (3.2, 5.5)).name == "F4"  # a real move beyond the margin switches
    assert sk.update(finger, (-0.05, 5.5)) is None  # outside is outside, never snapped to C4


# 6–7. one finger vs others, simultaneous fingers ----------------------------------------------------


CAL = PianoCalibration.from_corners([(0.2, 0.5), (0.8, 0.5), (0.8, 0.9), (0.2, 0.9)], LAYOUT.size,
                                    CameraKey(0, 1280, 720, True))
ASPECT = 720 / 1280


class Rig:
    """Real tracker (no smoothing) + key mapper + detector, driven by synthetic hands."""

    def __init__(self):
        self.tracker = FingerTracker(FingerTrackerConfig(smoothing=SmoothingConfig(kind=SmoothingKind.NONE)))
        self.mapper = KeyMapper(LAYOUT, CAL)
        self.detector = PressDetector(LAYOUT)
        self.t = 0.0

    def frame(self, offsets=None, hands=(Handedness.LEFT, Handedness.RIGHT), label_flip=False):
        """offsets: FingerId → downward tip offset (normalized image y)."""
        offsets = offsets or {}
        obs = []
        for hand in hands:
            o = make_hand(hand, (0.35 if hand is Handedness.LEFT else 0.65, 0.75))
            pts = o.landmarks.copy()
            for finger, dy in offsets.items():
                if finger.hand is hand:
                    pts[finger.finger.tip, 1] += dy
            label = hand.opposite if (label_flip and hand is Handedness.RIGHT) else hand
            obs.append(HandObservation(handedness=label, score=0.95, landmarks=pts))
        state = self.tracker.update(obs, self.t)
        mapping = self.mapper.map(state)
        out = self.detector.update(self.t, state, mapping, ASPECT)
        self.t += DT
        return out

    def run(self, frames):
        events = []
        last = None
        for offsets in frames:
            last = self.frame(offsets)
            events += last.events
        return events, last


def finger_tap(finger, depth=0.02, rest=10):
    seq = [{}] * rest
    seq += [{finger: depth * k / 3} for k in (1, 2, 3)] + [{finger: depth}] * 6
    seq += [{finger: depth * k / 3} for k in (2, 1, 0)] + [{}] * 10
    return seq


def test_feature_is_hand_relative():
    hand = make_hand(Handedness.RIGHT, (0.5, 0.5))
    moved = hand.landmarks.copy()
    moved[:, :2] += (0.2, 0.1)  # the whole hand moves
    a = press_features(hand.landmarks, Finger.INDEX, ASPECT)
    b = press_features(moved, Finger.INDEX, ASPECT)
    assert a == pytest.approx(b)
    assert hand_scale(hand.landmarks, ASPECT) > 0


def test_one_finger_tapping_does_not_trigger_the_others():
    rig = Rig()
    events, _ = rig.run(finger_tap(FingerId.RIGHT_INDEX))
    presses = [e for e in events if e.kind == "press"]
    assert len(presses) == 1 and presses[0].finger is FingerId.RIGHT_INDEX
    assert {e.finger for e in events} == {FingerId.RIGHT_INDEX}


def test_moving_the_whole_hand_does_not_press():
    rig = Rig()
    events = []
    for i in range(60):
        rig.t += 0  # keep cadence
        state_shift = 0.002 * i  # slow slide of the whole hand to the right
        obs = []
        for hand in Handedness:
            o = make_hand(hand, (0.3 + state_shift if hand is Handedness.LEFT else 0.6 + state_shift, 0.75))
            obs.append(o)
        state = rig.tracker.update(obs, rig.t)
        events += rig.detector.update(rig.t, state, rig.mapper.map(state), ASPECT).events
        rig.t += DT
    assert [e for e in events if e.kind == "press"] == []


def test_two_fingers_pressing_at_once_give_two_independent_presses():
    rig = Rig()
    seq = [{}] * 10 + [{FingerId.LEFT_INDEX: 0.02 * k / 3, FingerId.RIGHT_RING: 0.02 * k / 3} for k in (1, 2, 3)]
    seq += [{FingerId.LEFT_INDEX: 0.02, FingerId.RIGHT_RING: 0.02}] * 4
    events, last = rig.run(seq)
    presses = sorted((e.finger.name, e.key.name) for e in events if e.kind == "press")
    assert [f for f, _ in presses] == ["LEFT_INDEX", "RIGHT_RING"]
    assert len({k for _, k in presses}) == 2
    assert len(last.pressed_keys) == 2


# 8. tracking loss -------------------------------------------------------------------------------


def test_tracking_loss_releases_and_stale_velocity_cannot_fire():
    values = tap(hold=20, after=0)[:18]  # pressed, still down
    n = len(values)
    tracked = [True] * n
    events, m = run(values + [None] * 3 + [0.3] * 10, tracked=tracked + [False] * 3 + [True] * 10)
    assert kinds(events) == [("press", "ok"), ("release", "tracking_lost")]
    # Re-acquired already "down" (0.3): no new press from a jump over the gap.
    assert m.phase is PressPhase.IDLE


def test_time_gap_is_a_discontinuity():
    m = FingerPressMachine(FingerId.RIGHT_INDEX, CFG)
    for i in range(10):
        m.update(PressInput(i * DT, 0.0, C4, True))
    # Next sample 0.5 s later already deep: velocity across the gap must not be used.
    events = m.update(PressInput(10 * DT + 0.5, 0.3, C4, True))
    events += m.update(PressInput(10 * DT + 0.5 + DT, 0.3, C4, True))
    assert [e for e in events if e.kind == "press"] == []


# 9. outside the keyboard --------------------------------------------------------------------------


def test_press_outside_the_keyboard_is_rejected_not_snapped():
    values = tap()
    events, _ = run(values, keys=[None] * len(values))
    assert kinds(events) == [("rejected", "outside")]


def test_uncalibrated_press_is_rejected():
    events, _ = run(tap(), calibrated=False)
    assert kinds(events) == [("rejected", "uncalibrated")]


# 10. identity changes / discontinuities ---------------------------------------------------------


def test_discontinuity_flag_releases_and_resets():
    values = tap(hold=20, after=0)[:18] + [0.3] * 8
    disc = [False] * 18 + [True] + [False] * 7
    events, _ = run(values, discontinuity=disc)
    assert kinds(events) == [("press", "ok"), ("release", "tracking_lost")]


def test_hand_label_flicker_resets_that_hand_only():
    rig = Rig()
    rig.run(finger_tap(FingerId.LEFT_INDEX)[:16])  # left index is down now
    out = rig.frame({FingerId.LEFT_INDEX: 0.02}, label_flip=True)  # MediaPipe briefly calls the right hand LEFT
    released = [e for e in out.events if e.kind == "release"]
    # Only the right hand's fingers are reset; the left index press is not affected.
    assert all(e.finger.hand is Handedness.RIGHT for e in released)
    assert out.fingers[FingerId.LEFT_INDEX].phase.value in ("PRESSED", "RELEASE_CANDIDATE")


# refractory & whole-hand motion -------------------------------------------------------------------


def test_refractory_period_is_per_finger():
    cfg = PressConfig(refractory_s=0.5)
    values = tap(after=2) + tap(rest=0)
    events, _ = run(values, config=cfg)
    assert [k for k, r in kinds(events)] == ["press", "release", "rejected"]
    assert kinds(events)[-1] == ("rejected", "refractory")


def test_fast_hand_motion_suppresses_new_presses():
    values = tap()
    speed = [5.0] * len(values)
    events, _ = run(values, hand_speed=speed)
    assert kinds(events) == [("rejected", "hand_moving")]  # reported once per gesture


def test_only_index_mode_plays_the_index_fingers_and_nothing_else():
    from dataclasses import replace

    def presses(config, finger):
        rig = Rig()
        rig.detector = PressDetector(LAYOUT, config)
        return [e.finger for offsets in finger_tap(finger) for e in rig.frame(offsets).events if e.kind == "press"]

    only = replace(PressConfig(), only_index=True)
    assert presses(PressConfig(), FingerId.RIGHT_RING) == [FingerId.RIGHT_RING]
    assert presses(only, FingerId.RIGHT_RING) == []
    assert presses(only, FingerId.RIGHT_INDEX) == [FingerId.RIGHT_INDEX]
    assert presses(only, FingerId.LEFT_INDEX) == [FingerId.LEFT_INDEX]
