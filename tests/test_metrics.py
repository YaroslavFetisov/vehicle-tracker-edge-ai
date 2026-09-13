from __future__ import annotations

from vehicle_tracker.metrics import Metrics


def test_repeated_calls_accumulate():
    metrics = Metrics()
    metrics.record("detect", 0.010)
    metrics.record("detect", 0.020)
    metrics.frame_done(0.030, had_vehicles=True)

    assert metrics.ms_per_frame("detect") == 30.0


def test_an_expensive_stage_that_runs_rarely_is_cheap_per_frame():
    metrics = Metrics()
    for frame in range(10):
        if frame == 0:
            metrics.record("plate", 0.200)
        metrics.frame_done(0.020, had_vehicles=True)

    assert metrics.ms_per_frame("plate") == 20.0


def test_an_unused_stage_costs_nothing():
    metrics = Metrics()
    metrics.frame_done(0.010, had_vehicles=True)

    assert metrics.ms_per_frame("plate") == 0.0


def test_the_stage_context_manager_records_time():
    metrics = Metrics()
    with metrics.stage("detect"):
        pass
    metrics.frame_done(0.010, had_vehicles=True)

    assert metrics.ms_per_frame("detect") >= 0.0


def test_a_failing_stage_is_still_recorded():
    metrics = Metrics()
    try:
        with metrics.stage("detect"):
            raise RuntimeError("inference failed")
    except RuntimeError:
        pass
    metrics.frame_done(0.010, had_vehicles=True)

    assert "detect" in metrics.status_line()


def test_frames_without_vehicles_are_counted():
    metrics = Metrics()
    metrics.frame_done(0.010, had_vehicles=True)
    metrics.frame_done(0.010, had_vehicles=False)
    metrics.frame_done(0.010, had_vehicles=False)

    assert metrics.frames == 3
    assert metrics.frames_without_vehicles == 2


def test_rate_follows_the_frame_time():
    metrics = Metrics(smoothing=1.0)
    metrics.frame_done(0.020, had_vehicles=True)

    assert metrics.fps == 50.0


def test_a_zero_length_frame_does_not_break_the_rate():
    metrics = Metrics()
    metrics.frame_done(0.0, had_vehicles=True)

    assert metrics.fps == 0.0
    assert metrics.frames == 1


def test_a_nested_stage_is_kept_out_of_the_frame_budget():
    metrics = Metrics()
    metrics.record("plate", 0.100)
    metrics.record("plate read", 0.090, nested=True)
    metrics.frame_done(0.110, had_vehicles=True)

    status = metrics.status_line()
    assert "plate 100.0ms" in status
    assert "plate read" not in status
    assert "plate read" in "\n".join(metrics.summary())


def test_summary_reports_every_stage():
    metrics = Metrics()
    metrics.record("detect", 0.010)
    metrics.record("plate", 0.100)
    metrics.frame_done(0.110, had_vehicles=True)

    summary = "\n".join(metrics.summary())
    assert "detect" in summary
    assert "plate" in summary
    assert "1 frames" in summary
