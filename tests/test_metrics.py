import pytest

from handpiano.metrics.latency import RollingStats
from handpiano.metrics.performance import ProcessCpuMonitor, RateCounter
from handpiano.metrics.tracking_metrics import PresenceRate


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


def test_cpu_monitor_uses_cpu_over_wall_time():
    wall = iter([0.0, 1.0, 2.0])
    cpu = iter([0.0, 0.5, 2.0])
    mon = ProcessCpuMonitor(clock=lambda: next(wall), cpu_clock=lambda: next(cpu))
    assert mon.sample() is None
    assert mon.sample() == pytest.approx(50.0)
    assert mon.sample() == pytest.approx(150.0)  # more than one core in use


def test_presence_rate():
    pr = PresenceRate(window=4)
    assert pr.lost_rate() is None
    for present in (True, False, True, True):
        pr.add(present)
    assert pr.lost_rate() == pytest.approx(0.25)
    pr.add(False)  # window drops the first True
    assert pr.lost_rate() == pytest.approx(0.5)
