import pytest

from handpiano.metrics.latency import RollingStats
from handpiano.metrics.performance import CpuAccounting, CpuComponent, CpuSampler, RateCounter
from handpiano.metrics.tracking_metrics import RollingFraction


def test_rolling_stats_empty_reports_none():
    s = RollingStats().summary()
    assert s.count == 0
    assert s.p50 is None and s.p95 is None and s.mean is None


def test_rolling_stats_percentiles():
    stats = RollingStats(window=100)
    for v in range(1, 101):
        stats.add(v)
    s = stats.summary()
    assert s.count == 100
    assert s.p50 == pytest.approx(50.5)
    assert s.p95 == pytest.approx(95.05)
    assert s.max == 100
    assert s.last == 100


def test_rolling_stats_window_discards_old_samples():
    stats = RollingStats(window=3)
    for v in (1000, 1, 2, 3):
        stats.add(v)
    assert stats.summary().max == 3


def test_rolling_stats_clear():
    stats = RollingStats()
    stats.add(5)
    stats.clear()
    assert stats.summary().count == 0


def test_rate_counter_needs_two_ticks():
    rc = RateCounter(window_s=2.0)
    assert rc.rate(now=0.0) is None
    rc.tick(0.0)
    assert rc.rate(now=0.0) is None


def test_rate_counter_measures_regular_ticks():
    rc = RateCounter(window_s=2.0)
    for i in range(31):
        rc.tick(i / 30)
    assert rc.rate(now=1.0) == pytest.approx(30.0)


def test_rate_counter_forgets_old_ticks():
    rc = RateCounter(window_s=1.0)
    for i in range(10):
        rc.tick(i * 0.1)
    assert rc.rate(now=5.0) is None


def test_cpu_sampler_first_sample_is_unknown():
    assert CpuSampler(clock=lambda: 0.0, process_clock=lambda: 0.0).sample() is None


def test_cpu_sampler_breakdown_and_unattributed():
    acc = CpuAccounting()
    wall = iter([0.0, 2.0])
    proc = iter([0.0, 3.0])  # 3 CPU-s in 2 wall-s = 150 % of one core
    sampler = CpuSampler(acc, clock=lambda: next(wall), process_clock=lambda: next(proc))
    sampler.sample()
    acc.add(CpuComponent.INFERENCE, 1.0)
    acc.add(CpuComponent.UI, 0.4)
    b = sampler.sample()
    assert b.process_pct == pytest.approx(150.0)
    assert b.components_pct[CpuComponent.INFERENCE] == pytest.approx(50.0)
    assert b.components_pct[CpuComponent.UI] == pytest.approx(20.0)
    # 150 − 70: CPU of threads HandPiano cannot see (MediaPipe workers, Qt, OS).
    assert b.unattributed_pct == pytest.approx(80.0)


def test_cpu_accounting_ignores_non_positive():
    acc = CpuAccounting()
    acc.add(CpuComponent.UI, 0.0)
    acc.add(CpuComponent.UI, -1.0)
    assert acc.totals() == {}


def test_rolling_fraction():
    rf = RollingFraction(window=4)
    assert rf.fraction() is None
    for v in (True, False, True, True):
        rf.add(v)
    assert rf.fraction() == pytest.approx(0.75)
    rf.add(False)  # window drops the first True
    assert rf.fraction() == pytest.approx(0.5)
    assert rf.count() == 4
