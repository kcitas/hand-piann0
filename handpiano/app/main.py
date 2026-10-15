"""Entry point: ``python -m handpiano``."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import replace
from pathlib import Path

# Quiet MediaPipe/TFLite native logging. Must be set before mediapipe is imported.
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from PySide6.QtGui import QIcon  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from handpiano.app.application import Application  # noqa: E402
from handpiano.app.config import AppConfig  # noqa: E402
from handpiano.audio.engine import AudioEngine, NullOutput, SoundDeviceOutput  # noqa: E402
from handpiano.instrument.keyboard import KeyboardLayout  # noqa: E402
from handpiano.interaction.trace_landmarks import sidecar_paths  # noqa: E402
from handpiano.metrics.performance import CpuAccounting  # noqa: E402
from handpiano.ui.audio_test_window import AudioTestWindow  # noqa: E402
from handpiano.ui.main_window import ICON_PATH, MainWindow  # noqa: E402


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="handpiano", description="Instrumento virtual controlado con las manos.")
    parser.add_argument("--camera", type=int, default=0, help="índice de la cámara (por defecto 0)")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30, help="FPS solicitados; la cámara puede entregar otro valor")
    parser.add_argument("--no-mirror", action="store_true", help="no reflejar la imagen (vista no-espejo)")
    parser.add_argument("--lab", action="store_true", help="abrir con el Tracking Lab visible")
    parser.add_argument(
        "--check-hands",
        action="store_true",
        help="validación guiada LEFT/RIGHT con instrucciones en pantalla (abre el Tracking Lab)",
    )
    parser.add_argument("--start-note", default="C4", help="primera tecla del piano (por defecto C4)")
    parser.add_argument("--octaves", type=int, default=2, help="octavas del piano (por defecto 2)")
    parser.add_argument(
        "--check-keys",
        action="store_true",
        help="prueba guiada de precisión dedo → tecla (requiere el piano calibrado; abre el Tracking Lab)",
    )
    parser.add_argument("--check-keys-csv", type=Path, metavar="RUTA", help="guarda las muestras de --check-keys en CSV")
    parser.add_argument(
        "--record-trace",
        type=Path,
        metavar="RUTA",
        help="grabación guiada de trazas de pulsación (solo números) para analizar con scripts/analyze_trace.py",
    )
    parser.add_argument(
        "--trace-protocol",
        choices=("v2", "focus", "keys", "play", "dedos", "indices"),
        default="v2",
        help="pasos de --record-trace: v2 (8 dedos, avisos cada 1.5 s), focus (dedos problemáticos, cada 3 s), "
        "keys (dedos sobre el teclado del ordenador: cada pulsación real queda medida), play (35 s tocando "
        "libremente con las manos), dedos (5 toques guiados con cada dedo, ~1 min 40 s) o indices (10 toques "
        "guiados con cada índice, ~50 s)",
    )
    parser.add_argument(
        "--audio-test",
        action="store_true",
        help="prueba de sonido sin cámara: los tres instrumentos con el teclado y el ratón",
    )
    parser.add_argument("--no-audio", action="store_true", help="sin salida de sonido")
    parser.add_argument("--microfono", action="store_true",
                        help="tocar sobre la mesa: el micrófono detecta el instante del golpe y la cámara el dedo")
    parser.add_argument("--piano-mesa", action="store_true",
                        help="usar el piano calibrado con 4 esquinas sobre la mesa en lugar del piano en pantalla")
    parser.add_argument("--debug-notes", action="store_true",
                        help="diagnóstico: una línea en consola por pulsación (detector → nota → motor de audio)")
    parser.add_argument("--debug-keys", action="store_true",
                        help="diagnóstico: una línea en consola por cada tecla que entrega el sistema")
    parser.add_argument("--instrument", choices=("grand", "upright", "electric"), default="grand",
                        help="instrumento inicial (por defecto grand)")
    parser.add_argument("--confianza", type=float, default=None, metavar="0-1",
                        help="confianza mínima de MediaPipe para detectar y seguir las manos (por defecto 0.5); "
                             "más baja = pierde menos las manos, pero puede ver manos donde no las hay")
    parser.add_argument("--solo-indices", action="store_true",
                        help="solo tocan los dos dedos índice; los demás dedos no suenan nunca")
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
        piano=replace(config.piano, start=args.start_note, octaves=args.octaves, screen=not args.piano_mesa),
    )
    if args.confianza is not None:
        if not 0.0 < args.confianza <= 1.0:
            print(f"--confianza debe estar entre 0 y 1 (recibido {args.confianza})", file=sys.stderr)
            return 2
        c = args.confianza
        config = replace(config, detector=replace(config.detector, min_hand_detection_confidence=c,
                                                  min_hand_presence_confidence=c, min_tracking_confidence=c))
        logging.getLogger(__name__).info("Confianza de MediaPipe: %.2f", c)
    if args.solo_indices:
        config = replace(config, presses=replace(config.presses, only_index=True))
        logging.getLogger(__name__).info("Modo solo índices: solo suenan los dos dedos índice")
    try:
        layout = KeyboardLayout.from_octaves(config.piano.start, config.piano.octaves)
    except ValueError as exc:
        print(f"Teclado no válido ({args.start_note}, {args.octaves} octavas): {exc}", file=sys.stderr)
        return 2

    if args.record_trace:
        existing = [p for p in sidecar_paths(args.record_trace).values() if p.exists()]
        if existing:
            print("No se graba: ya existen " + ", ".join(map(str, existing)) + ". Elige otro nombre.", file=sys.stderr)
            return 2

    qt_app = QApplication(sys.argv[:1])
    qt_app.setApplicationName("HandPiano")
    qt_app.setWindowIcon(QIcon(str(ICON_PATH)))
    engine = AudioEngine(output=NullOutput("sonido desactivado (--no-audio)") if args.no_audio else SoundDeviceOutput())
    engine.start()
    engine.set_instrument(args.instrument)
    engine.preload()
    if args.audio_test:
        test_window = AudioTestWindow(engine)
        qt_app.aboutToQuit.connect(engine.close)
        test_window.show()
        return qt_app.exec()
    cpu = CpuAccounting()
    window = MainWindow(config.camera, cpu=cpu, layout=layout)
    mic = None
    if args.microfono:
        from handpiano.audio.tap_input import MicTapInput

        mic = MicTapInput()
        if not mic.start():
            print(f"{mic.error} — se sigue sin micrófono (solo cámara).", file=sys.stderr)
            mic = None
    app = Application(config, window, cpu=cpu, audio=engine, mic=mic)
    app.debug_keys = args.debug_keys
    app.debug_notes = args.debug_notes
    qt_app.aboutToQuit.connect(app.shutdown)
    window.show()
    if args.lab or args.check_hands or args.check_keys or args.record_trace:
        window.set_lab_mode(True)
    app.start()
    if args.check_hands:
        app.request_handedness_check()
    if args.check_keys:
        app.request_key_check(args.check_keys_csv)
    if args.record_trace:
        app.request_trace(args.record_trace, args.trace_protocol)
    return qt_app.exec()


if __name__ == "__main__":
    sys.exit(main())
