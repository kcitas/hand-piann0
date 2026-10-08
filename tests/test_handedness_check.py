from handpiano.app.handedness_check import CheckConfig, CheckStep, HandednessCheck, StepSpec
from handpiano.tracking.finger_tracker import FingerTracker
from handpiano.tracking.landmarks import Handedness
from tests.fakes import make_hand

L, R = Handedness.LEFT, Handedness.RIGHT
STEPS = (
    StepSpec(CheckStep.PREPARE, 1.0, "prep"),
    StepSpec(CheckStep.LEFT_ONLY, 2.0, "left"),
    StepSpec(CheckStep.RIGHT_ONLY, 2.0, "right"),
    StepSpec(CheckStep.BOTH, 2.0, "both"),
)
FPS = 30


def _check(**kw):
    check = HandednessCheck(config=CheckConfig(steps=STEPS, settle_s=0.5, min_frames=10, **kw))
    check.start(0.0)
    return check


def _run(check, hands_for_step):
    tracker = FingerTracker()
    t = 0.0
    while t < 7.0:
        spec, _, _ = check.current(t)
        hands = hands_for_step(spec.step if spec else None)
        check.update(t, tracker.update(hands, t))
        t += 1 / FPS


def _left():
    return make_hand(L, (0.3, 0.5))


def _right():
    return make_hand(R, (0.7, 0.5))


def test_steps_follow_the_schedule():
    check = _check()
    assert check.current(0.5)[0].step is CheckStep.PREPARE
    assert check.current(1.5)[0].step is CheckStep.LEFT_ONLY
    spec, remaining, _ = check.current(4.0)
    assert spec.step is CheckStep.RIGHT_ONLY and abs(remaining - 1.0) < 1e-9
    assert check.current(7.5)[0] is None
    assert check.is_done(7.5)


def test_correct_handedness_passes():
    check = _check()
    _run(check, lambda s: {CheckStep.LEFT_ONLY: [_left()], CheckStep.RIGHT_ONLY: [_right()],
                           CheckStep.BOTH: [_left(), _right()]}.get(s, []))
    assert check.passed()
    assert check.results[CheckStep.LEFT_ONLY].rate == 1.0
    assert check.results[CheckStep.BOTH].rate == 1.0


def test_inverted_labels_fail():
    """What the app did before the fix: the left hand came back as RIGHT."""
    check = _check()
    _run(check, lambda s: {CheckStep.LEFT_ONLY: [make_hand(R, (0.3, 0.5))],
                           CheckStep.RIGHT_ONLY: [make_hand(L, (0.7, 0.5))],
                           CheckStep.BOTH: [make_hand(R, (0.3, 0.5)), make_hand(L, (0.7, 0.5))]}.get(s, []))
    assert not check.passed()
    assert check.verdict(CheckStep.LEFT_ONLY) == "FALLA"
    assert check.verdict(CheckStep.BOTH) == "FALLA"


def test_no_hands_gives_insufficient_data_not_a_pass():
    check = _check()
    _run(check, lambda s: [])
    assert check.verdict(CheckStep.LEFT_ONLY) == "datos insuficientes"
    assert check.results[CheckStep.LEFT_ONLY].rate is None
    assert check.results[CheckStep.LEFT_ONLY].missing > 0
    assert not check.passed()


def test_settle_period_is_not_scored():
    check = _check()
    _run(check, lambda s: {CheckStep.LEFT_ONLY: [_left()]}.get(s, []))
    # 2 s step minus 0.5 s settle at 30 fps ≈ 45 frames, not 60.
    assert 40 <= check.results[CheckStep.LEFT_ONLY].evaluated <= 47


def test_records_what_was_seen_for_diagnosis():
    check = _check()
    # During the RIGHT step the left hand stays up: not a label error, a protocol error.
    _run(check, lambda s: {CheckStep.LEFT_ONLY: [_left()], CheckStep.RIGHT_ONLY: [_left(), _right()]}.get(s, []))
    seen = check.results[CheckStep.RIGHT_ONLY].seen
    assert seen["LEFT+RIGHT"] > 0 and seen["RIGHT"] == 0
    assert "LEFT+RIGHT" in check.summary_lines()[1]
