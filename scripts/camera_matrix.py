"""Measures which modes a camera really delivers. No frames are saved.

For each requested resolution/fps it opens the camera, reads frames for a few
seconds and prints what was delivered: resolution, the fps the driver reports
and the fps actually measured.

Usage: python scripts/camera_matrix.py [--camera 0] [--seconds 3]
"""

from __future__ import annotations

import argparse
import sys
import time

from handpiano.camera.camera_config import CameraSettings
from handpiano.camera.camera_manager import CameraError, CameraManager

MODES = [
    (640, 480, 30), (640, 480, 60), (640, 360, 30), (960, 540, 30),
    (1280, 720, 30), (1280, 720, 60), (1920, 1080, 30), (1920, 1080, 60),
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--seconds", type=float, default=3.0)
    args = parser.parse_args()

    print("| Solicitado | Entregado | FPS reportado (driver) | FPS medido |")
    print("| --- | --- | --- | --- |")
    for width, height, fps in MODES:
        camera = CameraManager(CameraSettings(device_index=args.camera, width=width, height=height, fps=fps))
        try:
            mode = camera.open()
        except CameraError as exc:
            print(f"| {width}×{height} @ {fps} | error: {exc.kind.value} | — | — |")
            continue
        try:
            camera.read()  # discard one frame after the warm-up
            frames, start = 0, time.perf_counter()
            while time.perf_counter() - start < args.seconds:
                frames += camera.read() is not None
            measured = frames / (time.perf_counter() - start)
        finally:
            camera.release()
        reported = "N/A" if mode.reported_fps is None else f"{mode.reported_fps:g}"
        print(f"| {width}×{height} @ {fps} | {mode.actual_width}×{mode.actual_height} | {reported} | {measured:.1f} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
