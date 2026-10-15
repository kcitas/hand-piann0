"""Keyboard calibration and the guided key-accuracy test, on the Qt main thread.

Owns the KeyMapper (shared with the tracking thread, which only reads its
calibration reference), the corner-capture session and the calibration store.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Callable
from pathlib import Path

from handpiano.app.config import PianoConfig
from handpiano.app.pipeline import TrackingSnapshot
from handpiano.camera.camera_config import CameraMode
from handpiano.instrument.keyboard import KeyboardLayout
from handpiano.mapping.calibration import CalibrationError, CalibrationStore, CameraKey, PianoCalibration
from handpiano.mapping.calibration_session import CalibrationSession
from handpiano.mapping.key_accuracy import AccuracyReport, AccuracySample, KeyAccuracyCheck, ProbePoint
from handpiano.mapping.key_mapper import KeyMapper
from handpiano.tracking.finger_tracker import TrackStatus
from handpiano.tracking.landmarks import FingerId
from handpiano.ui.main_window import MainWindow

log = logging.getLogger(__name__)

NEEDS_CALIBRATION = "Calibración necesaria · pulsa «Calibrar piano»"
SUCCESS_BANNER_S = 3.0


class PianoController:
    def __init__(self, config: PianoConfig, window: MainWindow, layout: KeyboardLayout, mirror: bool) -> None:
        self.config = config
        self.window = window
        self.mirror = mirror
        self.layout = layout
        self.mapper = KeyMapper(self.layout)
        self.store = CalibrationStore(config.calibration_path)
        self.status = "NO — calibración necesaria"
        self.session: CalibrationSession | None = None
        self._camera_key: CameraKey | None = None
        self._previous: PianoCalibration | None = None
        self._session_tip: tuple[float, float] | None = None
        self._session_finger: FingerId | None = None
        self._two_hands = False
        self.accuracy: KeyAccuracyCheck | None = None
        self._accuracy_requested = False
        # --check-keys: calibration controls are disabled for the whole run.
        self.measurement_only = False
        self.last_report: AccuracyReport | None = None
        self._acc_finger: FingerId | None = None
        self._acc_tip: tuple[float, float] | None = None
        self._acc_two_hands = False
        # Called with an exit code when --check-keys must end the app (no valid calibration).
        self.on_exit: Callable[[int], None] | None = None
        self._finish_at: float | None = None
        self._exit_code = 0
        self._accuracy_csv: Path | None = None
        self._banner_until: float | None = None

    @property
    def busy(self) -> bool:
        """Calibrating or running the accuracy test (they own the banner)."""
        return self.session is not None or self.accuracy is not None or self._banner_until is not None

    @property
    def calibrated(self) -> bool:
        return self.mapper.calibration is not None

    # -- camera lifecycle --------------------------------------------------------

    def on_camera_running(self, mode: CameraMode | None) -> None:
        if mode is None:
            return
        key = CameraKey(mode.device_index, mode.actual_width, mode.actual_height, self.mirror)
        if key == self._camera_key and self.mapper.calibration is not None:
            return
        self._camera_key = key
        if self.config.screen:
            self._set_calibration(self.screen_calibration(key))
            self.status = "OK — piano en pantalla (posición fija en la imagen, sin calibrar)"
            log.info("Piano calibration: %s", self.status)
            return
        calibration, reason = self.store.load(key, self.layout.size)
        self._set_calibration(calibration)
        self.status = f"OK — {reason}" if calibration else f"NO — {reason}"
        log.info("Piano calibration: %s", self.status)

    def screen_calibration(self, key: CameraKey) -> PianoCalibration:
        """The keyboard at a fixed, normalized place of the image: same for any window size or resolution."""
        left, top, right, bottom = self.config.screen_area
        corners = [(left, top), (right, top), (right, bottom), (left, bottom)]
        return PianoCalibration.from_corners(corners, self.layout.size, key, created_at="pantalla")

    def _screen_mode_notice(self) -> bool:
        if not self.config.screen:
            return False
        self.window.camera_view.set_banner(
            "Piano en pantalla", ["El piano está fijo en la imagen y no se calibra.",
                                  "Para calibrar sobre la mesa, arranca con --piano-mesa"], None, "#22d3ee")
        self._banner_until = None
        return True

    def _set_calibration(self, calibration: PianoCalibration | None) -> None:
        self.mapper.calibration = calibration
        self.window.camera_view.set_piano_overlay(self.layout, calibration)
        self.window.piano_view.set_state(None, calibration is not None, None if calibration else NEEDS_CALIBRATION)

    # -- calibration -------------------------------------------------------------

    def start_calibration(self, now: float, frame_aspect: float | None) -> None:
        if self.measurement_only:
            log.warning("Calibration request ignored: --check-keys only measures with the saved calibration")
            return
        if self._screen_mode_notice():
            return
        if self._camera_key is None or frame_aspect is None:
            self.window.camera_view.set_banner("Calibrar piano", ["Esperando a la cámara…"])
            return
        self._previous = self.mapper.calibration
        self._set_calibration(None)
        self.session = CalibrationSession(config=self.config.calibration, aspect=frame_aspect)
        self._banner_until = None
        self.window.piano_view.set_state(None, False, "Calibrando… sigue las instrucciones del vídeo")
        self._update_session(now, None)  # show the first instruction right away
        log.info("Calibration started")

    def forget_calibration(self) -> None:
        if self.measurement_only:
            log.warning("Forget-calibration request ignored during --check-keys")
            return
        if self._screen_mode_notice():  # the stored (table) calibration is never deleted from screen mode
            return
        self.session = None
        self.window.camera_view.set_calibration_overlay(None)
        self.store.clear()
        self._set_calibration(None)
        self.status = "NO — calibración olvidada"
        self.window.camera_view.set_banner(None)

    def key_command(self, command: str, now: float) -> None:
        check = self.accuracy
        if check is not None:
            if command == "capture":
                sample = check.capture_now(now, self._acc_finger, self._acc_tip)
                if sample is not None:
                    self._log_sample(sample)
            elif command == "cancel":
                self._finish_accuracy(now, interrupted=True)
            return
        session = self.session
        if session is None:
            return
        if command == "cancel":
            self.session = None
            self.window.camera_view.set_calibration_overlay(None)
            self._set_calibration(self._previous)
            self.window.camera_view.set_banner(None)
            log.info("Calibration cancelled")
        elif command == "undo":
            session.undo()
        elif command == "capture":
            if session.capture_now(self._session_tip) is not None:
                self._after_capture(now)

    def _pick_tip(self, snapshot: TrackingSnapshot) -> tuple[FingerId | None, tuple[float, float] | None, bool]:
        """(finger, tip, two_hands). Exactly one visible hand is required: with two, it is
        ambiguous which index is pointing (a resting hand would be captured), so none is used."""
        tracked = [
            f for f in (FingerId.RIGHT_INDEX, FingerId.LEFT_INDEX)
            if snapshot.state.fingers[f].status is TrackStatus.TRACKED and snapshot.state.fingers[f].tip is not None
        ]
        if len(tracked) != 1:
            return None, None, len(tracked) == 2
        tip = snapshot.state.fingers[tracked[0]].tip
        assert tip is not None
        return tracked[0], (float(tip[0]), float(tip[1])), False

    def _after_capture(self, now: float) -> None:
        session = self.session
        assert session is not None
        log.info("Corner %d captured at %s", len(session.points), session.points[-1])
        if not session.done:
            return
        assert self._camera_key is not None
        try:
            calibration = PianoCalibration.from_corners(session.points, self.layout.size, self._camera_key)
        except CalibrationError as exc:
            self.session = None
            self.window.camera_view.set_calibration_overlay(None)
            self._set_calibration(self._previous)
            self.window.camera_view.set_banner(
                "Calibración no válida", [exc.user_message, "Pulsa «Calibrar piano» para repetir"], None, "#f87171"
            )
            self._banner_until = now + 6.0
            log.warning("Calibration rejected: %s", exc.user_message)
            return
        self.session = None
        self.window.camera_view.set_calibration_overlay(None)
        self._set_calibration(calibration)
        try:
            self.store.save(calibration)
            self.status = f"OK — nueva ({calibration.created_at})"
        except OSError as exc:
            log.warning("Could not save calibration: %s", exc)
            self.status = "OK — nueva (no se pudo guardar)"
        self.window.camera_view.set_banner("Piano calibrado", ["Mueve los dedos sobre las teclas"], None, "#34d399")
        self._banner_until = now + SUCCESS_BANNER_S
        log.info("Calibration saved: corners %s", calibration.camera_points)

    # -- accuracy test (--check-keys) -------------------------------------------

    def request_accuracy_check(self, csv_path: Path | None) -> None:
        """Measurement mode: uses the saved calibration, never creates one."""
        self._accuracy_requested = True
        self._accuracy_csv = csv_path
        self.measurement_only = True
        self.window.calibrate_button.setEnabled(False)
        self.window.forget_button.setEnabled(False)

    def cancel_accuracy(self) -> None:
        self.accuracy = None
        self.window.piano_view.set_target(None)
        self.window.camera_view.set_target(None)
        self.window.camera_view.set_target_point(None)

    def _visible_points(self, check: KeyAccuracyCheck) -> tuple[tuple[ProbePoint, ...], list[ProbePoint]]:
        """Points whose projection falls inside the camera image (others cannot be pointed at)."""
        calibration = self.mapper.calibration
        assert calibration is not None
        visible: list[ProbePoint] = []
        skipped: list[ProbePoint] = []
        for point in check.points:
            x, y = calibration.to_camera([point.piano_xy])[0]
            (visible if 0.02 <= x <= 0.98 and 0.02 <= y <= 0.98 else skipped).append(point)
        return tuple(visible), skipped

    def _start_accuracy(self, now: float, aspect: float) -> None:
        self._accuracy_requested = False
        if not self.calibrated:
            lines = ["No valid calibration found.", "Run the normal calibration first.", f"Motivo: {self.status}"]
            self.window.camera_view.set_banner("Prueba de precisión: sin calibración válida", lines, None, "#f87171")
            for line in lines:
                log.error("Key accuracy: %s", line)
            print("\n".join(lines), file=sys.stderr, flush=True)
            self._finish(exit_code=1, delay_s=6.0, now=now)
            return
        check = KeyAccuracyCheck(self.mapper, aspect=aspect)
        visible, skipped = self._visible_points(check)
        check.points = visible
        for point in skipped:
            log.warning("Key accuracy: point %d (%s) skipped: outside the camera image", point.sample_id, point.label)
        check.start(now)
        self.accuracy = check
        log.info("Key accuracy test started: %d points (%d skipped), calibration %s",
                 len(visible), len(skipped), self.status)

    def _update_accuracy(self, now: float, snapshot: TrackingSnapshot | None) -> None:
        if self._finish_at is not None and now >= self._finish_at:
            self._finish_at = None
            if self.on_exit is not None:
                self.on_exit(self._exit_code)
            return
        if self._accuracy_requested:
            if self._camera_key is None or snapshot is None:
                self.window.camera_view.set_banner("Prueba de precisión", ["Esperando a la cámara y a la calibración…"])
                return
            self._start_accuracy(now, snapshot.frame.height / snapshot.frame.width)
        check = self.accuracy
        if check is None:
            return
        if snapshot is not None:
            finger, tip, two_hands = self._pick_tip(snapshot)
            self._acc_finger, self._acc_tip, self._acc_two_hands = finger, tip, two_hands
            sample = check.update(now, finger, tip, two_hands)
            if sample is not None:
                self._log_sample(sample)
        if check.done:
            self._finish_accuracy(now, interrupted=False)
            return
        self._show_accuracy(now, check)

    def _show_accuracy(self, now: float, check: KeyAccuracyCheck) -> None:
        if check.preparing(now):
            assert check._target_started_at is not None
            left = check._target_started_at - now
            self.window.camera_view.set_banner(
                "Prueba de precisión: prepárate",
                [f"{len(check.points)} puntos · usa UNA sola mano y apunta con el índice", f"{left:.0f} s"],
                1 - left / check.config.prepare_s,
            )
            return
        point = check.current
        assert point is not None
        calibration = self.mapper.calibration
        if calibration is not None:
            x, y = calibration.to_camera([point.piano_xy])[0]
            self.window.camera_view.set_target_point((float(x), float(y)))
        midi = None if point.expected is None else point.expected.midi
        self.window.camera_view.set_target(midi)
        self.window.piano_view.set_target(midi)
        if self._acc_two_hands:
            status = "Se ven dos manos: usa solo UNA"
        elif self._acc_tip is None:
            status = "No se ve ningún índice"
        elif check.waiting_to_move:
            status = "Lleva el índice al NUEVO punto marcado"
        else:
            status = "Mantén el índice quieto sobre el punto marcado…"
        where = "en el punto marcado, FUERA del piano" if point.expected is None else f"sobre {point.label} (punto marcado)"
        self.window.camera_view.set_banner(
            f"Prueba {len(check.samples) + 1}/{len(check.points)} · Objetivo: {point.label}",
            [f"Coloca el índice {where}", status,
             f"{check.time_left(now):.0f} s · Espacio: capturar ya · Esc: terminar"],
            check.stability(now),
        )

    def _log_sample(self, sample: AccuracySample) -> None:
        log.info(
            "Key accuracy sample %d: expected=%s detected=%s category=%s finger=%s%s",
            sample.sample_id, sample.expected_key or "outside", sample.detected_key or "none",
            sample.error_category, "none" if sample.finger is None else sample.finger.name,
            f" miss_reason={sample.miss_reason}" if sample.miss_reason else "",
        )

    def _finish_accuracy(self, now: float, interrupted: bool) -> None:
        check = self.accuracy
        assert check is not None
        report = check.report()
        lines = report.lines()
        title = "Prueba de precisión " + ("interrumpida (resultados parciales)" if interrupted else "terminada")
        for line in lines:
            log.info("Key accuracy: %s", line)
        print("\n".join([title, *lines]), flush=True)
        if self._accuracy_csv is not None:
            check.write_csv(self._accuracy_csv)
            log.info("Key accuracy samples written to %s", self._accuracy_csv)
        self.last_report = report
        self.cancel_accuracy()
        exact = "N/A" if report.exact_accuracy is None else f"{report.exact_accuracy * 100:.1f}%"
        outside = "N/A" if report.outside_accuracy is None else f"{report.outside_accuracy * 100:.1f}%"
        self.window.camera_view.set_banner(
            title,
            [f"Teclas: {report.exact}/{report.inside} exactas ({exact}) · adyacentes {report.adjacent} · "
             f"lejanas {report.far} · sin detección {report.miss_inside}",
             f"Fuera: {report.outside_correct}/{report.outside} correctas ({outside}) · "
             f"falsos positivos {report.outside_false_positive} · sin detección {report.miss_outside}",
             "Resultados completos en la terminal" + (" y en el CSV" if self._accuracy_csv else "")],
            None,
            "#34d399",
        )
        self.window.calibrate_button.setEnabled(True)
        self.window.forget_button.setEnabled(True)
        self.measurement_only = False

    def _finish(self, exit_code: int, delay_s: float, now: float) -> None:
        self._exit_code = exit_code
        self._finish_at = now + delay_s

    # -- per poll ----------------------------------------------------------------

    def update(self, now: float, snapshot: TrackingSnapshot | None, new_snapshot: bool) -> None:
        if self._banner_until is not None and now >= self._banner_until:
            self._banner_until = None
            self.window.camera_view.set_banner(None)
        if snapshot is not None and new_snapshot and self.session is None:
            self.window.piano_view.set_state(
                snapshot.mapping, self.calibrated, None if self.calibrated else NEEDS_CALIBRATION, snapshot.presses
            )
        self._update_session(now, snapshot if new_snapshot else None)
        self._update_accuracy(now, snapshot if new_snapshot else None)

    def _update_session(self, now: float, snapshot: TrackingSnapshot | None) -> None:
        session = self.session
        if session is None:
            return
        if snapshot is not None:
            self._session_finger, self._session_tip, self._two_hands = self._pick_tip(snapshot)
            if session.update(now, self._session_tip) is not None:
                self._after_capture(now)
                if self.session is None:
                    return
        finger = "N/A" if self._session_finger is None else self._session_finger.short
        lines = ["Espacio: capturar ya · Retroceso: deshacer · Esc: cancelar"]
        if self._two_hands:
            lines.insert(0, "Se ven dos manos: usa solo UNA mano para marcar las esquinas")
        elif self._session_tip is None:
            lines.insert(0, "No se ve ningún dedo índice")
        elif session.rejection:
            lines.insert(0, session.rejection)
        elif session.waiting_to_move_away:
            lines.insert(0, f"Esquina {len(session.points)} capturada. Lleva el dedo a la siguiente esquina")
        else:
            lines.insert(0, f"Dedo usado: {finger} · mantenlo quieto")
        self.window.camera_view.set_banner(
            f"{session.step}/4 · Coloca el índice en la esquina {session.corner_name} del piano",
            lines,
            session.dwell_progress(now),
        )
        self.window.camera_view.set_calibration_overlay(session.points, self._session_tip, session.dwell_progress(now))
