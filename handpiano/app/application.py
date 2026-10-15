"""Wires the tracking pipeline to the window. Runs on the Qt main thread."""

from __future__ import annotations

import logging
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path

from PySide6.QtCore import QCoreApplication, QEvent, QObject, Qt, QTimer
from PySide6.QtGui import QGuiApplication, QKeyEvent, QWindow
from PySide6.QtWidgets import QApplication

from handpiano.app.audio_controller import NOTE_KEY_OFFSETS, AudioController, NoteKeys
from handpiano.app.config import AppConfig
from handpiano.app.handedness_check import HandednessCheck
from handpiano.app.piano_controller import PianoController
from handpiano.app.pipeline import PipelineState, TrackingPipeline, TrackingSnapshot
from handpiano.app.placement import PlacementConfig, PlacementGuide, keyboard_box
from handpiano.app.press_sound import FINGER_NOTE_VELOCITY_FLOOR, PressSound, describe, outcomes_by_tag, summary
from handpiano.audio.engine import AudioEngine
from handpiano.audio.tap_input import MicTapInput
from handpiano.camera.camera_config import CameraSettings
from handpiano.camera.camera_manager import CameraManager, probe_cameras
from handpiano.game.chart import LANES, Chart, SongLibrary
from handpiano.game.rhythm import Judgment
from handpiano.instrument.keyboard import PianoKey
from handpiano.interaction.finger_presses import FINGER_NOTES, PressDetector, PressFrame
from handpiano.interaction.labels import KEY_FINGERS
from handpiano.interaction.press_detector import PressEvent
from handpiano.interaction.tap_fusion import TapFusion
from handpiano.interaction.trace import (
    PROTOCOLS,
    TraceRecorder,
    cue_report,
    finger_note_steps,
    load_trace,
    match_cues,
    replay,
)
from handpiano.metrics.performance import CpuAccounting, CpuComponent, CpuSampler
from handpiano.tracking.finger_tracker import FingerTracker
from handpiano.tracking.hand_tracker import HandTracker
from handpiano.tracking.landmarks import FingerId
from handpiano.ui.format import NA, fmt
from handpiano.ui.main_window import MainWindow

log = logging.getLogger(__name__)

SLOW_CAMERA_FPS = 15.0
RHYTHM_KEYS = {"D": 0, "F": 1, "J": 2, "K": 3}
# Live sensitivity (↑ / ↓): press threshold limits and step, in the units of the detector's signal.
SENSITIVITY_STEP = 0.05
SENSITIVITY_MIN = 0.10
SENSITIVITY_MAX = 1.20
SOLFEGE = {0: "Do", 2: "Re", 4: "Mi", 5: "Fa", 7: "Sol", 9: "La", 11: "Si"}
# Finger-note mode: lanes by finger, like D F J K on a computer keyboard (left middle, left index,
# right index, right middle); ring and pinky play the outer lanes.
FINGER_LANES = {FingerId.LEFT_PINKY: 0, FingerId.LEFT_RING: 0, FingerId.LEFT_MIDDLE: 0, FingerId.LEFT_INDEX: 1,
                FingerId.RIGHT_INDEX: 2, FingerId.RIGHT_MIDDLE: 3, FingerId.RIGHT_RING: 3, FingerId.RIGHT_PINKY: 3}
# Trace labels identify a key by Qt's key code, not by its text: the text of a key RELEASE can be
# empty or changed by a modifier (Option+L = "¬" on macOS), and a release that does not match its
# press would leave the key "held" and drop every later press of it.
_TRACE_KEY_NAMES: dict[int, str] = {
    **{getattr(Qt.Key, f"Key_{c}").value: c for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"},
    Qt.Key.Key_Ntilde.value: "Ñ", Qt.Key.Key_Semicolon.value: ";",
}


def trace_key_name(event: QKeyEvent) -> str:
    """Stable name of the physical key of a press/release ("" if it has none)."""
    name = _TRACE_KEY_NAMES.get(int(event.key()))
    if name:
        return name
    text = event.text().upper()
    return text if text.isprintable() else ""
HIT_VELOCITY = 100


class Application(QObject):
    def __init__(self, config: AppConfig, window: MainWindow, cpu: CpuAccounting | None = None,
                 audio: AudioEngine | None = None, mic: MicTapInput | None = None) -> None:
        super().__init__()
        self.config = config
        self.window = window
        self.pipeline: TrackingPipeline | None = None
        self.press_detector: PressDetector | None = None
        # The "Sensibilidad" banner shown by ↑ / ↓ goes away by itself.
        self._sensitivity_banner = QTimer(self)
        self._sensitivity_banner.setSingleShot(True)
        self._sensitivity_banner.setInterval(2500)
        self._sensitivity_banner.timeout.connect(lambda: self.trace is None and self.window.camera_view.set_banner(None))
        self._last_seq = -1
        self._reported_state: str | None = None
        self.cpu = cpu if cpu is not None else CpuAccounting()
        self._cpu_sampler = CpuSampler(self.cpu)
        self._probe_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="camera-probe")
        self._probe_future: Future[list[int]] | None = None
        self.handedness_check: HandednessCheck | None = None
        self._check_requested = False
        self.piano = PianoController(config.piano, window, window.piano_view.layout_, mirror=config.camera.mirror)
        self.piano.on_exit = self._exit
        self.recent_events: deque[PressEvent] = deque(maxlen=12)
        self.trace: TraceRecorder | None = None
        self._trace_path: Path | None = None
        self._trace_started_at: float | None = None
        self._cues_done = 0
        self._cue_shown_at: float | None = None
        self._keys_down: set[str] = set()
        self._key_taps_started = False  # a key-tap cue has been shown (keys are expected from then on)
        # --debug-keys: one console line per key event the system delivers (see eventFilter).
        self.debug_keys = False
        # --debug-notes: one console line per press (detector → note_on → audio thread → note_off).
        self.debug_notes = False
        self._notes_reported: set[int] = set()
        self._keys_heartbeat_at = 0.0

        # Audio: the engine is created by main() (or silent, for tests). Hand presses reach it
        # from the tracking thread (PressSound); mouse, keyboard and the game from this thread.
        self.audio = audio if audio is not None else AudioEngine()
        self.audio_controller = AudioController(self.audio, window.instrument_panel, self)
        self.press_sound = PressSound(self.audio)
        if config.presses.finger_notes:
            self.press_sound.velocity_floor = FINGER_NOTE_VELOCITY_FLOOR
        # Microphone mode (--microfono): the microphone says WHEN a finger hit the table and the camera
        # says WHICH finger; the camera-only detector keeps running but its events are not played.
        self.mic = mic
        self.fusion = TapFusion() if mic is not None else None
        self._fusion_held: dict[int, tuple[FingerId, ...]] = {}
        self._unmatched_reported = 0
        # Finger-note mode: the bar only says when a hand is not seen or leaves the image.
        self.placement = PlacementGuide(PlacementConfig(check_orientation=not config.presses.finger_notes))
        window.camera_view.set_press_threshold(config.presses.press_on)
        window.set_calibration_used(not config.piano.screen)  # the screen piano has a fixed place
        if config.presses.finger_notes:
            window.camera_view.set_finger_labels({f: SOLFEGE[m % 12] for f, m in FINGER_NOTES.items()})
            window.camera_view.set_piano_visible(False)
            window.set_finger_notes(True)
            window.rhythm_view.canvas.lane_labels = ("medio izq.", "índice izq.", "índice der.", "medio der.")
        if mic is not None:
            self.audio.note_listeners.append(mic.note_played)  # ignore our own piano for a moment
            window.piano_view.pressed_source = lambda: self._fusion_held
        self.note_keys = NoteKeys(base=self.piano.layout.keys[0].midi)
        self._rhythm_keys_down: set[str] = set()
        self.game = window.rhythm_game
        self.game.on_judgment.append(self._on_judgment)
        self.songs = SongLibrary()
        window.rhythm_view.set_slots(self.songs.scan())
        window.rhythm_view.start_requested.connect(self.start_song)
        window.rhythm_view.stop_requested.connect(self.stop_song)
        window.rhythm_view.canvas.lane_pressed.connect(lambda lane: self._lane_press(lane, time.perf_counter()))
        window.piano_view.key_pressed.connect(lambda midi: self.audio.note_on(midi, HIT_VELOCITY))
        window.piano_view.key_released.connect(self.audio.note_off)
        window.finger_keys_view.key_pressed.connect(lambda midi: self.audio.note_on(midi, HIT_VELOCITY))
        window.finger_keys_view.key_released.connect(self.audio.note_off)
        window.mode_changed.connect(self._on_mode)
        app = QCoreApplication.instance()
        if app is not None:
            app.installEventFilter(self)
        if isinstance(app, QGuiApplication):
            app.applicationStateChanged.connect(self._on_application_state)

        window.apply_camera_requested.connect(self.restart_camera)
        window.probe_requested.connect(self.probe_cameras)
        window.calibrate_requested.connect(self.start_calibration)
        window.forget_calibration_requested.connect(self.piano.forget_calibration)
        window.calibration_key.connect(lambda command: self.piano.key_command(command, time.monotonic()))

        self._frame_timer = QTimer(self)
        self._frame_timer.setInterval(config.ui.poll_interval_ms)
        self._frame_timer.timeout.connect(self._poll_frame)

        self._stats_timer = QTimer(self)
        self._stats_timer.setInterval(config.ui.stats_interval_ms)
        self._stats_timer.timeout.connect(self._poll_stats)

    # -- lifecycle -------------------------------------------------------------

    def start(self) -> None:
        self._start_pipeline(self.config.camera)
        self._frame_timer.start()
        self._stats_timer.start()

    def shutdown(self) -> None:
        self.audio_controller.stop()
        self._sensitivity_banner.stop()
        self._frame_timer.stop()
        self._stats_timer.stop()
        self.press_sound.stop_all()
        self.audio.all_notes_off()
        self.audio.close()
        if self.mic is not None:
            self.mic.close()
        app = QCoreApplication.instance()
        if app is not None:
            app.removeEventFilter(self)
        if self.pipeline is not None:
            self.pipeline.stop()
            self.pipeline = None
        self._probe_executor.shutdown(wait=False, cancel_futures=True)

    def request_handedness_check(self) -> None:
        """Starts the guided LEFT/RIGHT check as soon as the camera is running."""
        self._check_requested = True
        self.handedness_check = None
        self.window.camera_view.set_banner("Validación LEFT/RIGHT", ["Esperando a la cámara…"])

    def start_calibration(self) -> None:
        self.handedness_check = None
        self._check_requested = False
        snapshot = self.pipeline.latest_snapshot() if self.pipeline is not None else None
        aspect = None if snapshot is None else snapshot.frame.height / snapshot.frame.width
        self.piano.start_calibration(time.monotonic(), aspect)

    def request_trace(self, path: Path, protocol: str = "v2") -> None:
        """Guided press-trace recording (numbers only) into ``path``, with the steps of ``protocol``.

        While recording, physical key presses are logged as independent contact labels.
        """
        steps = PROTOCOLS[protocol]
        if self.config.presses.finger_notes:  # same cues; the instructions say "bend", hands up
            steps = finger_note_steps(steps)
        self.trace = TraceRecorder(steps, protocol=protocol, config=self.config.presses,
                                   mirrored=self.config.camera.mirror,
                                   detector={k: v for k, v in asdict(self.config.detector).items()
                                             if k != "model_path"})
        self._trace_path = path
        self._trace_started_at = None
        self._keys_down.clear()
        self._key_taps_started = False
        if protocol == "keys":
            self._show_trace_keys(self.trace)
        for midi in self.note_keys.release_all():
            self.audio.note_off(midi)
        # Ask for the keyboard focus (the keys protocol needs it). macOS may still leave it in the
        # terminal that launched the app; the banner then asks for a click.
        self.window.raise_()
        self.window.activateWindow()
        if self.pipeline is not None:
            self.pipeline.frame_listener = self.trace.on_frame

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        """Keyboard (UI thread). While a trace is recorded, physical keys are LABELS only (logged,
        never played); otherwise they play notes (Free Piano), hit lanes D F J K (rhythm game),
        select instruments (1–3) and change the volume (− / +).

        Qt may offer the same key event to several widgets; per-key down/up state deduplicates it.
        """
        if event.type() not in (QEvent.Type.KeyPress, QEvent.Type.KeyRelease):
            return False
        assert isinstance(event, QKeyEvent)
        if self.trace is None:
            consumed = self._play_key(event)
            delivered = "no (no se está grabando)"
        else:
            consumed, delivered = self._trace_key(self.trace, event)
        # The platform delivers a key to the QWindow first; logging only there gives one line per key.
        if self.debug_keys and isinstance(watched, QWindow):
            focus = QApplication.focusWidget()
            log.warning("[teclado] %s código=0x%x texto=%r receptor=%s foco=%s ventana_activa=%s "
                        "→ registrador: %s", "PRESS" if event.type() == QEvent.Type.KeyPress else "RELEASE",
                        event.key(), event.text(), type(watched).__name__,
                        type(focus).__name__ if focus else None, self.window.isActiveWindow(), delivered)
        return consumed

    def _trace_key(self, trace: TraceRecorder, event: QKeyEvent) -> tuple[bool, str]:
        """Logs a physical key as a label. Returns (consume the event, what happened — for --debug-keys)."""
        key = trace_key_name(event)
        if event.isAutoRepeat():
            return key in KEY_FINGERS, "no (auto-repetición)"
        if not key:
            return False, "no (tecla sin nombre)"
        down = event.type() == QEvent.Type.KeyPress
        if down == (key in self._keys_down):
            return key in KEY_FINGERS, "no (duplicado)"
        trace.record_key(key, down)
        (self._keys_down.add if down else self._keys_down.discard)(key)
        self._show_trace_keys(trace)
        return key in KEY_FINGERS, f"SÍ ({key}, {trace.key_count} eventos)"

    def _on_application_state(self, state: Qt.ApplicationState) -> None:
        """While a keys recording runs, say on the console when the app gains or loses the keyboard."""
        trace = self.trace
        if trace is None or trace.protocol != "keys":
            return
        if state == Qt.ApplicationState.ApplicationActive:
            log.warning("[teclado] HandPiano es la app activa: las teclas llegan a la grabación")
        else:
            log.warning("[teclado] HandPiano dejó de ser la app activa (%s): las teclas van a OTRA app "
                        "y no se registran", state.name)

    def _show_trace_keys(self, trace: TraceRecorder) -> None:
        """Permanent status-bar counter while recording: total events and presses per key."""
        downs = trace.key_downs
        detail = " ".join(f"{k}:{n}" for k, n in sorted(downs.items()))
        self.window.trace_keys_label.setText(f"Teclas registradas: {trace.key_count} eventos"
                                             + (f" · pulsaciones {detail}" if detail else ""))
        self.window.trace_keys_label.setVisible(True)

    def adjust_sensitivity(self, more_sensitive: bool) -> float:
        """↑ / ↓: a finger has to move less / more to play. Returns the new press threshold."""
        current = self.config.presses.press_on
        step = SENSITIVITY_STEP * (-1 if more_sensitive else 1)
        new = round(min(SENSITIVITY_MAX, max(SENSITIVITY_MIN, current + step)), 2)
        self.config = replace(self.config, presses=replace(self.config.presses, press_on=new, release_off=new / 2))
        if self.press_detector is not None:
            self.press_detector.set_press_on(new)
        self.window.camera_view.set_press_threshold(new)
        level = round((SENSITIVITY_MAX - new) / SENSITIVITY_STEP) + 1
        self.window.camera_view.set_banner(
            f"Sensibilidad {level}", ["↑ más sensible (basta con doblar menos el dedo) · ↓ menos sensible"], None,
            "#22d3ee")
        self._sensitivity_banner.start()
        log.info("Sensitivity: press_on %.2f", new)
        return new

    def _play_key(self, event: QKeyEvent) -> bool:
        if self.piano.busy:  # calibrating / accuracy test: keys belong to them
            return False
        if event.key() in (Qt.Key.Key_Up, Qt.Key.Key_Down):
            if event.type() == QEvent.Type.KeyPress and not event.isAutoRepeat():
                self.adjust_sensitivity(more_sensitive=event.key() == Qt.Key.Key_Up)
            return True
        text = event.text().upper()
        down = event.type() == QEvent.Type.KeyPress
        if event.isAutoRepeat():  # a held key: no new note or hit, but do not let widgets type it
            return text in (RHYTHM_KEYS if self.window.mode == "rhythm" else NOTE_KEY_OFFSETS)
        if down and self.audio_controller.handle_shortcut(text):
            return True
        if self.window.mode == "rhythm":
            lane = RHYTHM_KEYS.get(text)
            if lane is None:
                return False
            if down and text not in self._rhythm_keys_down:
                self._rhythm_keys_down.add(text)
                self._lane_press(lane, time.perf_counter())
            elif not down:
                self._rhythm_keys_down.discard(text)
            return True
        if down:
            midi = self.note_keys.press(text)
            if midi is not None:
                self.audio.note_on(midi, HIT_VELOCITY)
                return True
            return text in ("Z", "X")
        midi = self.note_keys.release(text)
        if midi is not None:
            self.audio.note_off(midi)
            return True
        return False

    # -- modes, presses and the rhythm game ---------------------------------------------

    def _on_presses(self, presses: PressFrame) -> None:
        """Tracking thread: detected presses → notes (Free Piano)."""
        if self.fusion is not None and self.mic is not None:
            self.fusion.add_onsets(self.mic.drain())
            events = self.fusion.on_frame(presses.t, presses)
            self._fusion_held = self.fusion.held_keys  # one reference swap: the UI reads it as a whole
            self.press_sound.on_frame(events, self._fusion_held)
            return
        self.press_sound.on_frame(presses.events, presses.pressed_keys)

    def _on_mode(self, mode: str) -> None:
        self.press_sound.set_enabled(mode == "piano")
        for midi in self.note_keys.release_all():
            self.audio.note_off(midi)
        if mode != "rhythm" and self.game.running:
            self.stop_song()
        if mode == "rhythm":
            self.window.rhythm_view.set_slots(self.songs.scan())

    def lane_for_key(self, key: PianoKey) -> int:
        """The keyboard is split in four equal-width zones, left → lane 0."""
        width = self.piano.layout.size[0]
        return max(0, min(LANES - 1, int((key.x + key.width / 2) / width * LANES)))

    def _lane_press(self, lane: int, t: float) -> None:
        if not self.game.running:
            return
        self.window.rhythm_view.canvas.flash_lane(lane)
        self.game.press(lane, t)

    def _on_judgment(self, judgment: Judgment) -> None:
        self.window.rhythm_view.canvas.judged()
        if judgment.grade != "MISS":
            midi = judgment.note.midi
            self.audio.note_on(midi, HIT_VELOCITY)
            QTimer.singleShot(int(max(0.15, judgment.note.duration) * 1000), lambda m=midi: self.audio.note_off(m))

    def start_song(self, chart: Chart) -> None:
        self.game.start(chart, time.perf_counter())
        self.window.rhythm_view.set_running(True)

    def stop_song(self) -> None:
        self.game.stop()
        self.window.rhythm_view.set_running(False)

    def _update_game(self, events: list[PressEvent]) -> None:
        game = self.game
        if not game.running:
            return
        for event in events:
            if event.kind == "press" and event.key is not None:
                lane = (FINGER_LANES[event.finger] if self.config.presses.finger_notes
                        else self.lane_for_key(event.key))
                self._lane_press(lane, event.t)
        now = time.perf_counter()
        game.update(now)
        if game.finished(now):
            acc = game.accuracy
            self.window.rhythm_view.canvas.notice = (
                f"Fin · {game.score} puntos · precisión {'—' if acc is None else f'{acc * 100:.1f} %'}"
                f" · combo máx {game.max_combo}")
            self.stop_song()

    def _update_trace(self, pipeline: TrackingPipeline) -> None:
        trace = self.trace
        if trace is None or self._trace_path is None or pipeline.state != PipelineState.RUNNING:
            return
        now = time.monotonic()
        if self._trace_started_at is None:
            self._trace_started_at = now
        if trace.protocol == "keys" and trace.key_count == 0 and now - self._keys_heartbeat_at >= 5.0:
            self._keys_heartbeat_at = now
            log.warning("[teclado] %.0f s grabando y 0 teclas · app %s · ventana activa %s",
                        now - self._trace_started_at, QGuiApplication.applicationState().name,
                        self.window.isActiveWindow())
        elapsed = now - self._trace_started_at
        for step in trace.steps:
            if elapsed < step.duration_s:
                if trace.step != step.name:
                    trace.step = step.name
                    self._cues_done = 0
                cue_times = step.cue_times()
                if self._cues_done < len(cue_times) and elapsed >= cue_times[self._cues_done]:
                    trace.mark_cue(step.expected)
                    self._cues_done += 1
                    self._cue_shown_at = now
                if self._cues_done and step.name.startswith("key_taps"):
                    self._key_taps_started = True
                flash = self._cue_shown_at is not None and now - self._cue_shown_at < 0.35
                warning = self._keys_warning(trace)
                if flash:
                    lines = [step.instruction, f"{self._cues_done}/{len(cue_times)}"]
                    if trace.protocol == "keys":
                        lines.append(warning or f"Teclas registradas: {trace.key_count}")
                    cue_word = "¡DOBLA!" if self.config.presses.finger_notes else "¡TOCA!"
                    self.window.camera_view.set_banner(cue_word, lines, None, "#f87171" if warning else "#34d399")
                elif warning:
                    self.window.camera_view.set_banner(
                        warning,
                        ["Haz CLIC en esta ventana de HandPiano y sigue pulsando",
                         (f"Teclas registradas: {trace.key_count} · sin teclas no hay etiquetas y la grabación "
                          "no servirá para el benchmark")],
                        elapsed / step.duration_s, "#f87171")
                else:
                    lines = [f"{step.duration_s - elapsed:.0f} s · grabando trazas (solo números)"]
                    if cue_times:
                        lines.insert(0, f"Señales: {self._cues_done}/{len(cue_times)}")
                    if trace.protocol == "keys":
                        lines.append(f"Teclas registradas: {trace.key_count}")
                    self.window.camera_view.set_banner(step.instruction, lines, elapsed / step.duration_s)
                return
            elapsed -= step.duration_s
        pipeline.frame_listener = None
        try:
            written = trace.write(self._trace_path)
        except FileExistsError as exc:
            log.error("Trace not written: %s", exc)
            self.window.camera_view.set_banner("Traza NO guardada", [str(exc)], None, "#f87171")
            self.trace = None
            return
        log.info("Press trace written: %s (%d rows, %d key events)", ", ".join(map(str, written.values())),
                 trace.row_count, trace.key_count)
        rows = load_trace(self._trace_path)
        if trace.protocol in ("dedos", "indices"):
            # Guided taps, one cue every 2 s: the app itself says how many were recognised with these
            # settings, so the player can compare postures and camera angles without any analysis.
            lines = cue_report(match_cues(rows, self.config.presses, self.piano.layout, window_s=1.0))
        else:
            results = replay(rows, self.config.presses, self.piano.layout)
            lines = [f"{r.step}: intencionadas {r.intended}, NO intencionadas {r.unintended}" for r in results
                     if r.step != "prepare"]
        if "labels" in written:
            contacts = sum(1 for e in trace.key_events if e.down)
            lines.insert(0, f"Teclas registradas: {trace.key_count} eventos ({contacts} pulsaciones) → "
                            f"{written['labels']}")
        elif trace.protocol == "keys":
            lines.insert(0, "⚠ NO se registró ninguna tecla: NO hay etiquetas (¿la ventana no tenía el foco?)")
        print(f"Traza guardada en {self._trace_path} ({trace.row_count} filas)", *lines, sep="\n", flush=True)
        ok = "labels" in written or trace.protocol != "keys"
        self.window.camera_view.set_banner("Grabación terminada" if ok else "Grabación terminada SIN etiquetas",
                                           [str(self._trace_path), *lines], None, "#34d399" if ok else "#f87171")
        self.trace = None

    def _keys_warning(self, trace: TraceRecorder) -> str | None:
        """Why physical keys may not be reaching the recording (keys protocol), or None."""
        if trace.protocol != "keys":
            return None
        if not self.window.isActiveWindow():
            return "⚠ LA VENTANA NO TIENE EL FOCO DEL TECLADO"
        if self._key_taps_started and trace.key_count == 0:
            return "⚠ NO LLEGA NINGUNA TECLA"
        return None

    def request_key_check(self, csv_path=None) -> None:
        self.piano.request_accuracy_check(csv_path)

    @staticmethod
    def _exit(code: int) -> None:
        app = QCoreApplication.instance()
        if app is not None:
            app.exit(code)

    def restart_camera(self, settings: CameraSettings) -> None:
        self.config = replace(self.config, camera=settings)
        if self.pipeline is not None:
            self.pipeline.stop()
        self._start_pipeline(settings)

    def _start_pipeline(self, settings: CameraSettings) -> None:
        detector_config = self.config.detector
        fingers = self.config.fingers
        fingers = replace(fingers, identity=replace(fingers.identity, mirrored_view=settings.mirror))
        self.pipeline = TrackingPipeline(
            camera=CameraManager(settings),
            detector_factory=lambda: HandTracker(detector_config, mirrored_input=settings.mirror),
            finger_tracker=FingerTracker(fingers),
            mirror=settings.mirror,
            cpu=self.cpu,
            key_mapper=self.piano.mapper,
            press_detector=(detector := PressDetector(self.piano.layout, self.config.presses)),
        )
        self.press_detector = detector
        self.recent_events.clear()
        self.press_sound.stop_all()
        self.pipeline.press_listener = self._on_presses
        if self.trace is not None:
            self.pipeline.frame_listener = self.trace.on_frame
        self.piano.mirror = settings.mirror
        self._last_seq = -1
        self._reported_state = None
        self.window.camera_view.set_message("Iniciando cámara y detector de manos…")
        self.window.camera_view.reset_metrics()
        self.window.debug_view.reset()
        self.pipeline.start()

    def probe_cameras(self) -> None:
        if self._probe_future is not None and not self._probe_future.done():
            return
        # The active camera is released first: some drivers refuse a second open.
        if self.pipeline is not None:
            self.pipeline.stop()
            self.pipeline = None
        self.window.camera_view.set_message("Buscando cámaras…")
        self._probe_future = self._probe_executor.submit(probe_cameras)

    # -- polling ---------------------------------------------------------------

    def _poll_frame(self) -> None:
        cpu_start = time.thread_time()
        try:
            self._poll_frame_inner()
        finally:
            self.cpu.add(CpuComponent.UI, time.thread_time() - cpu_start)

    def _poll_frame_inner(self) -> None:
        self._check_probe()
        pipeline = self.pipeline
        if pipeline is None:
            return
        if pipeline.state != self._reported_state:
            self._on_state_change(pipeline)
        events = pipeline.drain_events()
        for event in events:
            self.recent_events.append(event)
            log.debug("Press event: %s %s %s (%s)", event.kind, event.finger.name,
                      None if event.key is None else event.key.name, event.reason)
        if self.window.mode == "rhythm":
            self._update_game(events)
        snapshot = pipeline.latest_snapshot()
        new_snapshot = snapshot is not None and snapshot.seq != self._last_seq
        if new_snapshot:
            assert snapshot is not None
            self._last_seq = snapshot.seq
            self.window.camera_view.set_snapshot(snapshot)
            self._update_placement(snapshot)
            if self.handedness_check is not None:
                self.handedness_check.update(time.monotonic(), snapshot.state)
        self.piano.update(time.monotonic(), snapshot, new_snapshot)
        if snapshot is not None and new_snapshot and self.config.presses.finger_notes:
            self.window.finger_keys_view.set_presses(snapshot.presses)
        self._update_trace(pipeline)
        if not self.piano.busy:
            self._update_handedness_check(pipeline)

    def _update_placement(self, snapshot: TrackingSnapshot) -> None:
        """Live guide: tells the player how to place the hands over the keyboard (bottom of the video)."""
        if self.piano.session is not None:  # calibrating: the calibration banner gives the instructions
            self.window.camera_view.set_placement(None)
            return
        calibration = self.piano.mapper.calibration
        box = None if calibration is None else keyboard_box(calibration.camera_points)
        frame = snapshot.frame
        aspect = frame.height / frame.width if self.config.presses.finger_notes and frame.width else None
        placement = self.placement.update(time.monotonic(), snapshot.state, box, posture_aspect=aspect)
        self.window.camera_view.set_placement(placement.text, placement.ok)

    def _update_handedness_check(self, pipeline: TrackingPipeline) -> None:
        now = time.monotonic()
        if self._check_requested and pipeline.state == PipelineState.RUNNING:
            self._check_requested = False
            self.handedness_check = HandednessCheck(mirrored=self.config.camera.mirror)
            self.handedness_check.start(now)
        check = self.handedness_check
        if check is None:
            return
        view = self.window.camera_view
        spec, remaining, elapsed = check.current(now)
        if spec is None:
            if check.started_at is not None:
                passed = check.passed()
                title = "Validación LEFT/RIGHT: OK" if passed else "Validación LEFT/RIGHT: NO SUPERADA"
                lines = check.summary_lines()
                view.set_banner(title, lines, None, "#34d399" if passed else "#f87171")
                log.info("Handedness check finished (%s): %s", "OK" if passed else "FAILED", " | ".join(lines))
                check.started_at = None  # report once; the banner stays
            return
        detail = [f"{remaining:.0f} s"]
        if spec.step in check.results:
            r = check.results[spec.step]
            rate = "N/A" if r.rate is None else f"{r.rate:.0%}"
            detail.append(f"Coincidencia hasta ahora: {rate} ({r.evaluated} frames)")
        view.set_banner(spec.instruction, detail, elapsed / spec.duration_s)

    def _on_state_change(self, pipeline: TrackingPipeline) -> None:
        self._reported_state = pipeline.state
        if pipeline.state == PipelineState.FAILED:
            self.window.camera_view.set_message(pipeline.error_message)
            self.window.status_label.setText("Sin seguimiento")
        elif pipeline.state == PipelineState.RUNNING:
            self.window.debug_view.update_camera(pipeline.mode)
            self.piano.on_camera_running(pipeline.mode)

    def _check_probe(self) -> None:
        future = self._probe_future
        if future is None or not future.done():
            return
        self._probe_future = None
        try:
            found = future.result()
        except Exception:
            log.exception("Camera probe failed")
            found = []
        if found:
            self.window.set_cameras(found)
        current = self.config.camera
        if found and current.device_index not in found:
            current = replace(current, device_index=found[0])
        self._start_pipeline(current)

    def _poll_stats(self) -> None:
        cpu_start = time.thread_time()
        try:
            self._poll_stats_inner()
        finally:
            self.cpu.add(CpuComponent.UI, time.thread_time() - cpu_start)

    def _report_notes(self) -> None:
        """Press → sound diagnostics: Tracking Lab counters and, with --debug-notes, one line per press
        once it is complete (released) or 2 s old."""
        records = list(self.press_sound.records)
        outcomes = outcomes_by_tag(self.audio.mixer.outcomes)
        last = describe(records[-1], outcomes) if records else None
        self.window.debug_view.update_sound(summary(records, outcomes), last)
        if not self.debug_notes:
            return
        if self.fusion is not None and self.fusion.stats.unmatched > self._unmatched_reported:
            new = self.fusion.stats.unmatched - self._unmatched_reported
            self._unmatched_reported = self.fusion.stats.unmatched
            log.warning("[golpe] %d golpe(s) oído(s) sin ningún dedo que bajara ante la cámara "
                        "(golpes totales %d, dedos tocados %d)", new, self.fusion.stats.onsets,
                        self.fusion.stats.played)
        now = time.perf_counter()
        for record in records:
            if record.id in self._notes_reported:
                continue
            if record.released_t is not None or now - record.t > 2.0:
                self._notes_reported.add(record.id)
                log.warning(describe(record, outcomes))

    def _poll_stats_inner(self) -> None:
        self._report_notes()
        cpu = self._cpu_sampler.sample()
        pipeline = self.pipeline
        if pipeline is None or pipeline.state != PipelineState.RUNNING:
            return
        stats = pipeline.stats()
        view = self.window.camera_view
        snapshot = pipeline.latest_snapshot()

        mode = pipeline.mode
        resolution = f"{mode.actual_width}×{mode.actual_height}" if mode else "N/A"
        warning = ""
        if stats.camera_fps is not None and stats.camera_fps < SLOW_CAMERA_FPS:
            warning = "  ·  ⚠ cámara lenta: mejora la iluminación o baja la resolución"
        hands = str(snapshot.state.hands_tracked) if snapshot else NA
        latency = view.capture_to_paint_ms.summary()
        self.window.status_label.setText(
            f"Cámara {resolution} · {fmt(stats.camera_fps, 'fps')}   ·   "
            f"Tracking {fmt(stats.tracking_fps, 'fps')}   ·   "
            f"Captura→pintado p50 {fmt(latency.p50, 'ms', 0)}   ·   Manos {hands}   ·   "
            f"Piano: {'calibrado' if self.piano.calibrated else 'calibración necesaria'}{warning}"
        )
        if self.window.lab_checkbox.isChecked():
            ui = {
                "result_to_ui": view.result_to_ui_ms.summary(),
                "ui_to_paint": view.ui_to_paint_ms.summary(),
                "capture_to_paint": latency,
                "paint": view.paint_ms.summary(),
            }
            self.window.debug_view.update_stats(stats, ui, cpu)
            self.window.debug_view.update_snapshot(snapshot)
            self.window.debug_view.update_piano(
                self.piano.layout, self.piano.status, None if snapshot is None else snapshot.mapping
            )
            self.window.debug_view.update_presses(None if snapshot is None else snapshot.presses, list(self.recent_events))
