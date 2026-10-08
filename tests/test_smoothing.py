import numpy as np
import pytest

from handpiano.tracking.smoothing import OneEuroFilter, SmoothingConfig, SmoothingKind

FPS = 30.0
DT = 1.0 / FPS
NOISE = 0.004  # ~2.5 px at 640 px wide: comparable to real landmark jitter


def _run(filt, signal):
    return np.array([filt(np.array([v]), i * DT)[0] for i, v in enumerate(signal)])


def _ema(signal, alpha):
    out, y = [], signal[0]
    for v in signal:
        y = alpha * v + (1 - alpha) * y
        out.append(y)
    return np.array(out)


def _lag_frames(response, target, start):
    """Frames after ``start`` until the response gets within 10 % of the step."""
    step = target - response[start - 1]
    for k in range(start, len(response)):
        if abs(target - response[k]) <= 0.1 * abs(step):
            return k - start
    return len(response)


def test_first_sample_passes_through():
    f = OneEuroFilter(SmoothingConfig())
    out = f(np.array([0.3, 0.7]), 0.0)
    np.testing.assert_allclose(out, [0.3, 0.7])


def test_reduces_jitter_on_still_hand():
    rng = np.random.default_rng(0)
    signal = 0.5 + rng.normal(0, NOISE, 300)
    out = _run(OneEuroFilter(SmoothingConfig()), signal)
    assert np.std(out[30:]) < 0.5 * np.std(signal[30:])


def test_follows_fast_movement_with_less_lag_than_equally_smooth_ema():
    """One Euro: as stable as a strong EMA at rest, but much faster on a strike."""
    rng = np.random.default_rng(1)
    still = 0.5 + rng.normal(0, NOISE, 300)
    one_euro_still = _run(OneEuroFilter(SmoothingConfig()), still)
    jitter_one_euro = np.std(one_euro_still[30:])

    # Find the EMA alpha that gives the same (or better) jitter at rest.
    alpha = next(a for a in np.linspace(0.9, 0.02, 200) if np.std(_ema(still, a)[30:]) <= jitter_one_euro)

    # A sudden 0.1 jump (a quick finger strike), noise-free to isolate lag.
    step = np.concatenate([np.full(30, 0.5), np.full(30, 0.6)])
    lag_one_euro = _lag_frames(_run(OneEuroFilter(SmoothingConfig()), step), 0.6, 30)
    lag_ema = _lag_frames(_ema(step, alpha), 0.6, 30)
    assert lag_one_euro < lag_ema


def test_higher_beta_reduces_lag():
    step = np.concatenate([np.full(30, 0.5), np.full(30, 0.6)])
    slow = _lag_frames(_run(OneEuroFilter(SmoothingConfig(beta=0.0)), step), 0.6, 30)
    fast = _lag_frames(_run(OneEuroFilter(SmoothingConfig(beta=20.0)), step), 0.6, 30)
    assert fast < slow


def test_restarts_after_gap():
    f = OneEuroFilter(SmoothingConfig(max_gap_s=0.25))
    f(np.array([0.1]), 0.0)
    f(np.array([0.1]), DT)
    out = f(np.array([0.9]), 1.0)  # hand reappears elsewhere after 1 s
    assert out[0] == pytest.approx(0.9)


def test_none_kind_is_passthrough():
    f = OneEuroFilter(SmoothingConfig(kind=SmoothingKind.NONE))
    f(np.array([0.1]), 0.0)
    assert f(np.array([0.9]), DT)[0] == pytest.approx(0.9)


def test_filters_landmark_arrays_elementwise():
    f = OneEuroFilter(SmoothingConfig())
    pts = np.full((21, 3), 0.5)
    f(pts, 0.0)
    out = f(pts + 0.01, DT)
    assert out.shape == (21, 3)
    assert np.all((out > 0.5) & (out < 0.51))
