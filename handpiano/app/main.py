"""Entry point: ``python -m handpiano``."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import replace

# Quiet MediaPipe/TFLite native logging. Must be set before mediapipe is imported.
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from PySide6.QtGui import QIcon  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from handpiano.app.application import Application  # noqa: E402
from handpiano.app.config import AppConfig  # noqa: E402
from handpiano.ui.main_window import ICON_PATH, MainWindow  # noqa: E402


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="handpiano", description="Instrumento virtual controlado con las manos.")
    parser.add_argument("--camera", type=int, default=0, help="índice de la cámara (por defecto 0)")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30, help="FPS solicitados; la cámara puede entregar otro valor")
    parser.add_argument("--no-mirror", action="store_true", help="no reflejar la imagen (vista no-espejo)")
    parser.add_argument("--lab", action="store_true", help="abrir con el Tracking Lab visible")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    config = AppConfig()
    config = replace(
        config,
        camera=replace(
            config.camera,
            device_index=args.camera,
            width=args.width,
            height=args.height,
            fps=args.fps,
            mirror=not args.no_mirror,
        ),
    )

    qt_app = QApplication(sys.argv[:1])
    qt_app.setApplicationName("HandPiano")
    qt_app.setWindowIcon(QIcon(str(ICON_PATH)))
    window = MainWindow(config.camera)
    app = Application(config, window)
    qt_app.aboutToQuit.connect(app.shutdown)
    window.show()
    if args.lab:
        window.set_lab_mode(True)
    app.start()
    return qt_app.exec()


if __name__ == "__main__":
    sys.exit(main())
