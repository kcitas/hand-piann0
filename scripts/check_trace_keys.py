"""macOS check of the keys protocol: key events → Qt → recorder → ``.labels.csv``.

Usage:
    python scripts/check_trace_keys.py                     # automatic: Cocoa events posted to this app
    python scripts/check_trace_keys.py --manual            # YOU press the 8 keys physically (≤ 60 s)
    python scripts/check_trace_keys.py --manual --camera   # the same, with the camera and detector running
    python scripts/check_trace_keys.py --help

Automatic mode opens the real HandPiano window (no camera), starts a ``keys``
trace into a temporary folder, posts A S D F J K L Ñ (down + up) through THIS
app's Cocoa event queue (``NSApp postEvent``: the path a physical key takes once
macOS has delivered it to the app), writes the recording and validates the labels.

``--manual`` checks the one link the automatic mode cannot reach: whether macOS
delivers your physical keystrokes to the recording window. It prints one line
per key the window receives and the app's keyboard state every 3 s. A
persistent panel on the window says which test is running and which keys are
still missing (the recording protocol's own banners keep running underneath).

``--camera`` starts the camera pipeline exactly as a recording does. The test
only begins once the camera is ``running``; if it fails, or is still not
running after ``--camera-timeout`` seconds, or stops during the test, the
check FAILS (exit code 3) and never prints PASS.

Nothing is written outside a temporary folder and no key goes to another app.
Exit codes: 0 PASS · 1 keys/labels FAIL · 2 usage error · 3 camera not running.
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import QApplication, QLabel, QWidget  # noqa: E402

# Physical home-row keys (macOS virtual key codes) and the character each one types (ES layout).
HOME_ROW = (("a", 0x00), ("s", 0x01), ("d", 0x02), ("f", 0x03), ("j", 0x26), ("k", 0x28), ("l", 0x25),
            ("ñ", 0x29))
EXPECTED = {"A": "LEFT_PINKY", "S": "LEFT_RING", "D": "LEFT_MIDDLE", "F": "LEFT_INDEX", "J": "RIGHT_INDEX",
            "K": "RIGHT_MIDDLE", "L": "RIGHT_RING", "Ñ": "RIGHT_PINKY"}
EXIT_PASS, EXIT_FAIL, EXIT_USAGE, EXIT_CAMERA = 0, 1, 2, 3
INDICATOR_NAME = "ManualKeyTestIndicator"


class NSPoint(ctypes.Structure):
    _fields_ = [("x", ctypes.c_double), ("y", ctypes.c_double)]


class Cocoa:
    def __init__(self) -> None:
        self.objc = ctypes.cdll.LoadLibrary(ctypes.util.find_library("objc") or "/usr/lib/libobjc.dylib")
        ctypes.cdll.LoadLibrary(ctypes.util.find_library("AppKit")
                                or "/System/Library/Frameworks/AppKit.framework/AppKit")
        self.objc.objc_getClass.restype = ctypes.c_void_p
        self.objc.objc_getClass.argtypes = [ctypes.c_char_p]
        self.objc.sel_registerName.restype = ctypes.c_void_p
        self.objc.sel_registerName.argtypes = [ctypes.c_char_p]

    def cls(self, name: str) -> int:
        return self.objc.objc_getClass(name.encode())

    def sel(self, name: str) -> int:
        return self.objc.sel_registerName(name.encode())

    def send(self, restype, argtypes, receiver, selector, *args):  # type: ignore[no-untyped-def]
        fn = ctypes.CFUNCTYPE(restype, ctypes.c_void_p, ctypes.c_void_p, *argtypes)(("objc_msgSend", self.objc))
        return fn(receiver, self.sel(selector), *args)

    def post_key(self, window_view: int, char: str, keycode: int, down: bool) -> None:
        nswindow = self.send(ctypes.c_void_p, [], window_view, "window")
        number = self.send(ctypes.c_long, [], nswindow, "windowNumber")
        text = self.send(ctypes.c_void_p, [ctypes.c_char_p], self.cls("NSString"), "stringWithUTF8String:",
                         char.encode())
        event = self.send(
            ctypes.c_void_p,
            [ctypes.c_ulong, NSPoint, ctypes.c_ulong, ctypes.c_double, ctypes.c_long, ctypes.c_void_p,
             ctypes.c_void_p, ctypes.c_void_p, ctypes.c_bool, ctypes.c_ushort],
            self.cls("NSEvent"),
            "keyEventWithType:location:modifierFlags:timestamp:windowNumber:context:characters:"
            "charactersIgnoringModifiers:isARepeat:keyCode:",
            10 if down else 11, NSPoint(0, 0), 0, 0.0, number, None, text, text, False, keycode,
        )
        nsapp = self.send(ctypes.c_void_p, [], self.cls("NSApplication"), "sharedApplication")
        self.send(None, [ctypes.c_void_p, ctypes.c_bool], nsapp, "postEvent:atStart:", event, False)


def pump(app: QApplication, seconds: float) -> None:
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        app.processEvents()
        time.sleep(0.01)


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="check_trace_keys.py",
        description="Comprueba que las teclas A S D F J K L Ñ llegan al registrador de la ventana de grabación "
                    "y generan un .labels.csv válido (en una carpeta temporal).")
    parser.add_argument("--manual", action="store_true",
                        help="pulsas TÚ las 8 teclas físicamente (máx. 60 s) en lugar de enviarlas por Cocoa")
    parser.add_argument("--camera", action="store_true",
                        help="arranca la cámara y el detector como en una grabación; la prueba falla si la cámara "
                             "no llega a 'running'")
    parser.add_argument("--camera-timeout", type=float, default=15.0, metavar="S",
                        help="segundos de espera para que la cámara llegue a 'running' (por defecto 15)")
    parser.add_argument("--manual-timeout", type=float, default=60.0, metavar="S",
                        help="segundos de espera de las 8 teclas en modo manual (por defecto 60)")
    return parser.parse_args(argv)  # unknown arguments → error, exit code 2


def show_indicator(window: QWidget, with_camera: bool) -> QLabel:
    """A persistent panel over the top of the camera view, where the protocol's banners are painted.

    A child widget is always drawn above its parent's own painting, so the protocol banners
    (e.g. "No pulses nada") cannot hide it.
    """
    view = window.camera_view  # type: ignore[attr-defined]
    label = QLabel(view)
    label.setObjectName(INDICATOR_NAME)
    label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
    label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    label.setWordWrap(True)
    label.setStyleSheet("background: #22d3ee; color: #070b14; font-size: 17px; font-weight: 800; "
                        "border-radius: 10px; padding: 10px;")
    label.title = (f"PRUEBA MANUAL DE TECLADO {'CON' if with_camera else 'SIN'} CÁMARA "  # type: ignore[attr-defined]
                   "— no es una grabación")
    label.show()
    return label


def update_indicator(label: QLabel, received: set[str]) -> None:
    keys = "  ".join(f"{k}{' ✓' if k in received else ''}" for k in EXPECTED)
    label.setText(f"{label.title}\n"  # type: ignore[attr-defined]
                  "Haz CLIC en esta ventana y pulsa una a una:  A S D F J K L Ñ\n"
                  f"Recibidas: {keys}\n"
                  "Ignora los carteles del protocolo (p. ej. «No pulses nada»): esta prueba sí pide pulsar.")
    parent = label.parentWidget()
    if parent is not None:
        label.setGeometry(8, 8, max(200, parent.width() - 16), 170)
    label.raise_()


def camera_state(handpiano) -> str:  # type: ignore[no-untyped-def]
    return "sin pipeline" if handpiano.pipeline is None else str(handpiano.pipeline.state)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manual, with_camera = args.manual, args.camera
    if sys.platform != "darwin" and not manual:
        print("Solo macOS (usa la cola de eventos de Cocoa). Prueba --manual.")
        return EXIT_USAGE
    from handpiano.app.application import Application
    from handpiano.app.config import AppConfig
    from handpiano.app.pipeline import PipelineState
    from handpiano.audio.engine import AudioEngine, NullOutput
    from handpiano.interaction.labels import load_labels
    from handpiano.ui.main_window import MainWindow

    app = QApplication.instance() or QApplication(sys.argv[:1])
    assert isinstance(app, QApplication)
    folder = Path(tempfile.mkdtemp(prefix="handpiano_keys_"))
    window = MainWindow(AppConfig().camera)
    window.show()
    handpiano = Application(AppConfig(), window, audio=AudioEngine(sounds_dir=folder / "none",
                                                                    output=NullOutput()))

    def finish(code: int) -> int:
        handpiano.trace = None
        handpiano.shutdown()  # also stops the camera pipeline when --camera started it
        window.close()
        return code

    if with_camera:  # exactly as main() does for a recording: pipeline (camera + MediaPipe) first
        window.set_lab_mode(True)
        handpiano.start()
        deadline = time.monotonic() + args.camera_timeout
        while time.monotonic() < deadline:
            if handpiano.pipeline is not None and handpiano.pipeline.state in (PipelineState.RUNNING,
                                                                                PipelineState.FAILED):
                break
            pump(app, 0.1)
        state = camera_state(handpiano)
        print(f"Cámara: {state}", flush=True)
        if state != PipelineState.RUNNING:
            error = handpiano.pipeline.error_message if handpiano.pipeline is not None else ""
            print(f"ERROR: la cámara no está en 'running' (estado: {state}"
                  f"{'; ' + error if error else ''}). La prueba con cámara NO se ha realizado.", flush=True)
            return finish(EXIT_CAMERA)
    handpiano.request_trace(folder / "check.csv", "keys")
    pump(app, 0.5)
    trace = handpiano.trace
    assert trace is not None
    if manual:
        import logging

        logging.basicConfig(level=logging.WARNING, format="%(message)s")
        handpiano.debug_keys = True
        indicator = show_indicator(window, with_camera)
        print("Haz clic en la ventana de HandPiano y pulsa A S D F J K L Ñ "
              f"(máx. {args.manual_timeout:.0f} s)…", flush=True)
        deadline = time.monotonic() + args.manual_timeout
        next_state = 0.0
        while time.monotonic() < deadline:
            released = {e.key for e in trace.key_events if not e.down}
            update_indicator(indicator, released)
            pump(app, 0.1)
            if with_camera and camera_state(handpiano) != PipelineState.RUNNING:
                print(f"ERROR: la cámara dejó de estar en 'running' durante la prueba "
                      f"(estado: {camera_state(handpiano)}). Resultado NO válido.", flush=True)
                return finish(EXIT_CAMERA)
            if set(EXPECTED) <= {e.key for e in trace.key_events if not e.down}:
                break
            if time.monotonic() >= next_state:
                next_state = time.monotonic() + 3.0
                print(f"[estado] app {app.applicationState().name} · ventana activa {window.isActiveWindow()} "
                      f"· teclas recibidas {trace.key_count}"
                      + (f" · cámara {camera_state(handpiano)}" if with_camera else ""), flush=True)
        else:
            print(f"Tiempo agotado. Estado de la app: {app.applicationState().name}, "
                  f"ventana activa: {window.isActiveWindow()}")
    else:
        cocoa = Cocoa()
        for char, keycode in HOME_ROW:
            cocoa.post_key(int(window.winId()), char, keycode, True)
            pump(app, 0.05)
            cocoa.post_key(int(window.winId()), char, keycode, False)
            pump(app, 0.05)
        pump(app, 0.3)
    if with_camera and camera_state(handpiano) != PipelineState.RUNNING:
        print(f"ERROR: la cámara no está en 'running' al terminar (estado: {camera_state(handpiano)}). "
              "Resultado NO válido.", flush=True)
        return finish(EXIT_CAMERA)
    events = [(e.key, e.down) for e in trace.key_events]
    print(f"Eventos registrados: {len(events)} {events}")
    print(f"Contador en pantalla: {window.trace_keys_label.text()!r}")
    written = trace.write(folder / "check.csv")
    if "labels" not in written:
        print("FALLO: no se escribió .labels.csv")
        return finish(EXIT_FAIL)
    labels = load_labels(written["labels"])  # validates every row
    contacts = {lb.note.removeprefix("key="): lb.target for lb in labels if lb.label == "contact"}
    releases = {lb.note.removeprefix("key=") for lb in labels if lb.label == "release"}
    if manual:  # a person may also press other keys: require the 8 home-row keys, each with a release
        ok = all(contacts.get(k) == v for k, v in EXPECTED.items()) and set(EXPECTED) <= releases
    else:
        ok = contacts == EXPECTED and releases == set(EXPECTED)
    print(f"{written['labels']}: {len(labels)} etiquetas válidas; contactos {contacts}")
    print("PASS" if ok else f"FALLO: esperado {EXPECTED} con liberaciones para todas")
    return finish(EXIT_PASS if ok else EXIT_FAIL)


if __name__ == "__main__":
    sys.exit(main())
