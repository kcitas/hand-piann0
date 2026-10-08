"""Runs the real app (camera + MediaPipe + UI with the Tracking Lab) for a fixed time and
reports where CPU and time go. No frames are saved.

Usage:
    python scripts/profile_cpu.py --seconds 30                 # CPU breakdown + internal latencies
    python scripts/profile_cpu.py --seconds 30 --cprofile out.prof   # + cProfile of all Python threads

CPU per component comes from time.thread_time() measured inside each HandPiano
thread. "No atribuida" is process CPU that no HandPiano thread reports: mainly
MediaPipe/XNNPACK worker threads plus Qt and OS threads.

cProfile (Python 3.12+) records every Python thread but measures WALL time per
function, not CPU, and cannot see inside MediaPipe's C++ code. Use it to find
which Python functions take time; use the CPU breakdown to know who burns CPU.

By default the UI runs offscreen (no window). Painting offscreen uses a software
raster that may cost differently than an on-screen window; pass --window to
profile with a real window instead.
"""

from __future__ import annotations

import argparse
import cProfile
import os
import pstats
import sys
import time

os.environ.setdefault("GLOG_minloglevel", "2")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--cprofile", metavar="FILE", help="also write a cProfile .prof file and print the top")
    parser.add_argument("--window", action="store_true", help="show a real window instead of offscreen")
    parser.add_argument(
        "--null-detector",
        action="store_true",
        help="control run: same camera and UI but MediaPipe is replaced by a detector that finds nothing. "
        "Process CPU of a normal run minus this one is the cost of MediaPipe.",
    )
    parser.add_argument(
        "--no-ui",
        action="store_true",
        help="control run: camera + tracking pipeline only, no Qt at all (separates the UI's cost)",
    )
    args = parser.parse_args()
    if args.no_ui:
        return run_without_ui(args)
    if not args.window:
        os.environ["QT_QPA_PLATFORM"] = "offscreen"

    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication

    from handpiano.app.application import Application
    from handpiano.app.config import AppConfig
    from handpiano.app.pipeline import PipelineState
    from handpiano.metrics.performance import CpuAccounting, CpuComponent, CpuSampler
    from handpiano.tracking.finger_tracker import TrackStatus
    from handpiano.ui.debug_view import p50_p95
    from handpiano.ui.main_window import MainWindow

    if args.null_detector:
        import handpiano.app.application as application_module

        class NullDetector:
            def __init__(self, _config) -> None:
                pass

            def detect(self, rgb, timestamp_ms):
                return []

            def close(self) -> None:
                pass

        application_module.HandTracker = NullDetector  # type: ignore[assignment,misc]

    qt = QApplication(sys.argv[:1])
    cpu = CpuAccounting()
    config = AppConfig()
    window = MainWindow(config.camera, cpu=cpu)
    app = Application(config, window, cpu=cpu)
    window.show()
    window.set_lab_mode(True)

    profiler = cProfile.Profile() if args.cprofile else None
    sampler = CpuSampler(cpu)
    hand_frames = {"frames": 0, "0": 0, "1": 0, "2": 0}
    last_seq = [-1]
    started = [False]

    def watch_hands() -> None:
        pipeline = app.pipeline
        if pipeline is None:
            return
        if pipeline.state == PipelineState.RUNNING and not started[0]:
            # Measure from when the pipeline is running, not from startup.
            started[0] = True
            sampler.sample()
            if profiler:
                profiler.enable()
            QTimer.singleShot(int(args.seconds * 1000), finish)
        snap = pipeline.latest_snapshot()
        if snap is not None and snap.seq != last_seq[0]:
            last_seq[0] = snap.seq
            n = sum(1 for h in snap.state.hands.values() if h.status is TrackStatus.TRACKED)
            hand_frames["frames"] += 1
            hand_frames[str(n)] += 1
        if pipeline.state == PipelineState.FAILED:
            print("Pipeline failed:", pipeline.error_message)
            qt.quit()

    def finish() -> None:
        if profiler:
            profiler.disable()
        breakdown = sampler.sample()
        pipeline = app.pipeline
        stats = pipeline.stats() if pipeline else None
        view = window.camera_view
        mode = pipeline.mode if pipeline else None

        detector = "NULL detector (no MediaPipe)" if args.null_detector else "MediaPipe"
        print(f"\n=== HandPiano profile: {args.seconds:g} s, {detector}, {'window' if args.window else 'offscreen UI'} ===")
        if mode:
            print(f"Camera: requested {mode.requested_width}x{mode.requested_height}@{mode.requested_fps}, "
                  f"delivered {mode.actual_width}x{mode.actual_height}, driver reports {mode.reported_fps} fps")
        f = hand_frames["frames"] or 1
        print(f"Processed frames: {hand_frames['frames']}  (0 hands {hand_frames['0'] / f:.0%}, "
              f"1 hand {hand_frames['1'] / f:.0%}, 2 hands {hand_frames['2'] / f:.0%})")
        if stats:
            print(f"Camera fps (measured): {stats.camera_fps and round(stats.camera_fps, 1)}   "
                  f"Tracking fps: {stats.tracking_fps and round(stats.tracking_fps, 1)}   "
                  f"Dropped: {stats.frames_dropped}/{stats.frames_captured}")
            print("\nInternal latency (last 300 frames):")
            for name, s in [
                ("conversion", stats.conversion_ms), ("queue wait", stats.queue_wait_ms),
                ("inference", stats.inference_ms), ("tracker", stats.processing_ms),
                ("capture → result", stats.capture_to_result_ms), ("result → UI", view.result_to_ui_ms.summary()),
                ("UI → paint", view.ui_to_paint_ms.summary()), ("capture → paint", view.capture_to_paint_ms.summary()),
                ("paint", view.paint_ms.summary()),
            ]:
                print(f"  {name:<18} {p50_p95(s)}")
        if breakdown:
            print("\nCPU (% of one core):")
            print(f"  process total      {breakdown.process_pct:6.1f}")
            for comp in (CpuComponent.CAPTURE, CpuComponent.INFERENCE, CpuComponent.TRACKER, CpuComponent.UI):
                print(f"  {comp:<18} {breakdown.components_pct.get(comp, 0.0):6.1f}")
            print(f"  unattributed       {breakdown.unattributed_pct:6.1f}   (MediaPipe workers, Qt, OS)")
        if profiler:
            profiler.dump_stats(args.cprofile)
            print(f"\ncProfile written to {args.cprofile}. Top 15 by internal (wall) time:")
            pstats.Stats(profiler).sort_stats("tottime").print_stats(15)
        app.shutdown()
        qt.quit()

    timer = QTimer()
    timer.timeout.connect(watch_hands)
    timer.start(5)
    app.start()
    qt.exec()
    return 0


def run_without_ui(args: argparse.Namespace) -> int:
    from handpiano.app.config import AppConfig
    from handpiano.app.pipeline import PipelineState, TrackingPipeline
    from handpiano.camera.camera_manager import CameraManager
    from handpiano.metrics.performance import CpuAccounting, CpuComponent, CpuSampler
    from handpiano.tracking.finger_tracker import FingerTracker
    from handpiano.tracking.hand_tracker import HandTracker

    class NullDetector:
        def detect(self, rgb, timestamp_ms):
            return []

        def close(self) -> None:
            pass

    config = AppConfig()
    cpu = CpuAccounting()
    factory = NullDetector if args.null_detector else (lambda: HandTracker(config.detector))
    pipeline = TrackingPipeline(CameraManager(config.camera), factory, FingerTracker(config.fingers), cpu=cpu)
    pipeline.start()
    while pipeline.state == PipelineState.STARTING:
        time.sleep(0.01)
    if pipeline.state != PipelineState.RUNNING:
        print("Pipeline failed:", pipeline.error_message)
        return 1
    sampler = CpuSampler(cpu)
    sampler.sample()
    time.sleep(args.seconds)
    breakdown = sampler.sample()
    stats = pipeline.stats()
    pipeline.stop()
    detector = "NULL detector (no MediaPipe)" if args.null_detector else "MediaPipe"
    print(f"\n=== HandPiano profile: {args.seconds:g} s, {detector}, NO UI ===")
    print(f"Camera fps (measured): {stats.camera_fps and round(stats.camera_fps, 1)}   "
          f"Tracking fps: {stats.tracking_fps and round(stats.tracking_fps, 1)}")
    if breakdown:
        print("CPU (% of one core):")
        print(f"  process total      {breakdown.process_pct:6.1f}")
        for comp in (CpuComponent.CAPTURE, CpuComponent.INFERENCE, CpuComponent.TRACKER):
            print(f"  {comp:<18} {breakdown.components_pct.get(comp, 0.0):6.1f}")
        print(f"  unattributed       {breakdown.unattributed_pct:6.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
