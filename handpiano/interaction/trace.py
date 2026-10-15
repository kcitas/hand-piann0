"""Guided recording of press-detection traces, offline replay and diagnostics.

``python -m handpiano --record-trace trace.csv`` shows timed instructions
(stay still, tap with one named finger, slide without pressing, chord) and
writes one row per finger per processed frame: the three candidate press
signals, keys, detector state and events. Numbers only, never images.

Each step says which fingers SHOULD press, so a replay can count presses by the
instructed finger versus presses by any other finger (the "neighbouring keys"
problem) and presses during steps where nothing should be pressed.

Protocol v2 adds timed cues: during a tap step the screen flashes "¡TOCA!" every
``cue_every_s`` seconds and the frame where each cue appears is marked in the
``cue`` column. Presses can then be matched to cues (hit / missed tap / extra
press) per finger, instead of only counting presses per step. A cue is when a tap
was ASKED for, not when it happened.

Format v4 (every new recording) also writes the 21 landmarks of each hand, a
metadata file and, if the physical keyboard was used, independent contact
labels: see ``trace_landmarks.py`` and ``labels.py``. The ``keys`` protocol has
the fingers on a computer keyboard so each tap is a measured key down/up.
"""

from __future__ import annotations

import csv
import threading
import time
from collections import Counter
from collections.abc import Iterator
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from handpiano.app.pipeline import TrackingSnapshot
from handpiano.instrument.keyboard import KeyboardLayout, PianoKey
from handpiano.interaction.finger_presses import FINGER_NOTES, PressEngine
from handpiano.interaction.labels import KeyEvent, labels_from_keys, write_labels
from handpiano.interaction.press_detector import PressConfig, PressEvent, PressInput
from handpiano.interaction.trace_landmarks import (
    HandSample,
    sample_from_state,
    sidecar_paths,
    write_landmarks,
    write_meta,
)
from handpiano.tracking.landmarks import PLAYING_FINGERS, FingerId

PREPARE = "prepare"


@dataclass(frozen=True, slots=True)
class TraceStep:
    name: str
    duration_s: float
    instruction: str
    # Fingers that are asked to press in this step (empty = nobody should press).
    expected: tuple[FingerId, ...]
    # Protocol v2: a "¡TOCA!" cue every cue_every_s seconds, starting cue_lead_s into the step.
    cue_every_s: float | None = None
    cue_lead_s: float = 2.0

    def cue_times(self) -> list[float]:
        """Seconds after the step start at which a cue is shown."""
        if self.cue_every_s is None:
            return []
        times, t = [], self.cue_lead_s
        while t <= self.duration_s - 1.0:
            times.append(t)
            t += self.cue_every_s
        return times


# Protocol v1 (first recording, 2026-10-11): free tapping, no cues. Kept so old traces stay analyzable.
V1_STEPS: tuple[TraceStep, ...] = (
    TraceStep(PREPARE, 6.0, "Prepárate: ambas manos sobre el piano, relajadas", ()),
    TraceStep("rest", 8.0, "Manos sobre el piano, QUIETAS. No toques nada", ()),
    TraceStep("left_index_taps", 10.0, "Toca 5 veces una tecla con el ÍNDICE IZQUIERDO (los demás dedos quietos)",
              (FingerId.LEFT_INDEX,)),
    TraceStep("right_index_taps", 10.0, "Toca 5 veces una tecla con el ÍNDICE DERECHO", (FingerId.RIGHT_INDEX,)),
    TraceStep("right_middle_taps", 10.0, "Toca 5 veces una tecla con el MEDIO DERECHO", (FingerId.RIGHT_MIDDLE,)),
    TraceStep("slide", 8.0, "Desliza la mano derecha de un lado a otro SIN tocar", ()),
    TraceStep("chord", 10.0, "Toca 3 veces con ambos índices A LA VEZ", (FingerId.LEFT_INDEX, FingerId.RIGHT_INDEX)),
)

_FINGER_ES = {"INDEX": "ÍNDICE", "MIDDLE": "MEDIO", "RING": "ANULAR", "PINKY": "MEÑIQUE"}


def _tap_step(finger: FingerId, cues: int = 8, every: float = 1.5, prefix: str = "taps") -> TraceStep:
    hand = "IZQUIERDO" if finger.name.startswith("LEFT") else "DERECHO"
    name = f"{prefix}_{finger.name}"
    return TraceStep(
        name, 2.0 + cues * every,
        f"Toca con el {_FINGER_ES[finger.finger.name]} {hand} cada vez que aparezca ¡TOCA! (los demás dedos quietos)",
        (finger,), cue_every_s=every,
    )


# Protocol v2: one block per playing finger with timed cues, plus steps that must NOT press.
V2_STEPS: tuple[TraceStep, ...] = (
    TraceStep(PREPARE, 6.0, "Prepárate: ambas manos sobre el piano, relajadas", ()),
    TraceStep("rest", 10.0, "Manos sobre el piano, QUIETAS. No toques nada", ()),
    TraceStep("posture", 10.0, "Cierra y abre las manos LENTAMENTE, sin tocar (cambio de postura)", ()),
    *(_tap_step(f) for f in PLAYING_FINGERS),
    TraceStep("slide", 10.0, "Desliza las manos de un lado a otro SIN tocar", ()),
    TraceStep("chord", 2.0 + 6 * 2.0, "Toca con AMBOS ÍNDICES a la vez cada vez que aparezca ¡TOCA!",
              (FingerId.LEFT_INDEX, FingerId.RIGHT_INDEX), cue_every_s=2.0),
)

def _slow_tap_step(finger: FingerId, cues: int = 6, every: float = 3.0) -> TraceStep:
    hand = "IZQUIERDO" if finger.name.startswith("LEFT") else "DERECHO"
    return TraceStep(
        f"slow_taps_{finger.name}", 3.0 + cues * every,
        f"AMBAS manos sobre el piano. Toca con el {_FINGER_ES[finger.finger.name]} {hand} en cada ¡TOCA! "
        "(un toque claro, vuelve arriba; los demás dedos quietos)",
        (finger,), cue_every_s=every, cue_lead_s=3.0,
    )


# Focus protocol (after traza_v3): the fingers that failed, a known-good control, cues 3 s apart so
# that a press can only belong to one cue (with 1.5 s cues, reaction + detection latency filled the
# whole interval and the hit rate depended on the matching window).
FOCUS_FINGERS = (FingerId.LEFT_INDEX, FingerId.LEFT_RING, FingerId.RIGHT_MIDDLE, FingerId.RIGHT_RING)
FOCUS_STEPS: tuple[TraceStep, ...] = (
    TraceStep(PREPARE, 6.0, "Prepárate: AMBAS manos sobre el piano y visibles todo el tiempo", ()),
    TraceStep("rest", 8.0, "Ambas manos QUIETAS sobre el piano. No toques nada", ()),
    TraceStep("posture", 8.0, "Cierra y abre las manos LENTAMENTE, sin tocar", ()),
    *(_slow_tap_step(f) for f in FOCUS_FINGERS),
)

_KEY_OF = {"LEFT_PINKY": "A", "LEFT_RING": "S", "LEFT_MIDDLE": "D", "LEFT_INDEX": "F",
           "RIGHT_INDEX": "J", "RIGHT_MIDDLE": "K", "RIGHT_RING": "L", "RIGHT_PINKY": "Ñ"}


def _key_tap_step(finger: FingerId, cues: int = 5, every: float = 3.0) -> TraceStep:
    hand = "IZQUIERDO" if finger.name.startswith("LEFT") else "DERECHO"
    return TraceStep(
        f"key_taps_{finger.name}", 3.0 + cues * every,
        f"Pulsa la tecla {_KEY_OF[finger.name]} con el {_FINGER_ES[finger.finger.name]} {hand} en cada ¡TOCA! "
        "(una pulsación completa; los demás dedos apoyados sin pulsar)",
        (finger,), cue_every_s=every, cue_lead_s=3.0,
    )


# Keyboard protocol (format v4): the fingers rest on the home row of a PHYSICAL computer
# keyboard placed where the hands are seen, so every real press is a measured key
# down/up (independent of the camera and of the cue). "lift" adds movements WITHOUT
# contact (hard negatives); "free" adds natural, unprompted playing.
KEYS_STEPS: tuple[TraceStep, ...] = (
    TraceStep(PREPARE, 8.0, "Dedos sobre la fila central del teclado del ordenador: A S D F (izquierda) y "
              "J K L Ñ (derecha), sin pulsar. Ambas manos visibles", ()),
    TraceStep("rest", 10.0, "Dedos apoyados en las teclas, QUIETOS. No pulses nada", ()),
    TraceStep("lift", 10.0, "Levanta y baja los dedos SIN llegar a pulsar ninguna tecla", ()),
    *(_key_tap_step(f) for f in PLAYING_FINGERS),
    TraceStep("free", 30.0, "Pulsa libremente las teclas A S D F J K L Ñ, cada una con SU dedo, a tu ritmo", ()),
)

# Free-play protocol: 30 s of playing the virtual piano with the hands, no keyboard, no cues. It records
# what the camera and the detector saw while really playing, to find out why touches are lost.
PLAY_STEPS: tuple[TraceStep, ...] = (
    TraceStep(PREPARE, 5.0, "Prepárate: AMBAS manos sobre el piano dibujado. Espera a que la barra de abajo esté VERDE",
              ()),
    TraceStep("play_free", 30.0, "Toca el piano con las manos con normalidad: ambas manos, todas las teclas",
              PLAYING_FINGERS),
)

# Finger-count protocol (~1 min 40 s): every playing finger taps 5 times, one cue every 2 s, on the
# screen piano. The number of taps per finger is known, so one recording can be replayed offline with
# several detector settings to count lost taps and extra notes for each (the thresholds are chosen
# from that measurement, not by intuition).
FINGER_STEPS: tuple[TraceStep, ...] = (
    TraceStep(PREPARE, 6.0, "Prepárate: AMBAS manos sobre el piano dibujado. Espera a que la barra de abajo esté VERDE",
              ()),
    *(_tap_step(f, cues=5, every=2.0) for f in PLAYING_FINGERS),
)

# Index-only protocol (~50 s): 10 taps with each index finger, one cue every 2 s, to measure the
# --solo-indices mode (only the two index fingers play) the same way as "dedos".
INDEX_STEPS: tuple[TraceStep, ...] = (
    TraceStep(PREPARE, 6.0, "Prepárate: AMBAS manos sobre el piano dibujado. Espera a que la barra de abajo esté VERDE",
              ()),
    *(_tap_step(f, cues=10, every=2.0, prefix="index") for f in (FingerId.LEFT_INDEX, FingerId.RIGHT_INDEX)),
)

_NOTE_ES = {0: "Do", 2: "Re", 4: "Mi", 5: "Fa", 7: "Sol", 9: "La", 11: "Si"}
FINGER_NOTE_PREPARE = ("Prepárate: LEVANTA las dos manos, palmas hacia la cámara y dedos hacia ARRIBA. "
                       "Espera a que la barra de abajo esté VERDE")


def finger_note_steps(steps: tuple[TraceStep, ...]) -> tuple[TraceStep, ...]:
    """The same steps and cues, with the instructions of finger-note mode (hands up, bend the finger).

    Step names, durations and cues are unchanged, so recordings are analysed exactly the same way.
    """
    out = []
    for step in steps:
        if step.name == PREPARE:
            step = replace(step, instruction=FINGER_NOTE_PREPARE)
        elif len(step.expected) == 1 and step.cue_every_s:
            finger = step.expected[0]
            hand = "IZQUIERDO" if finger.name.startswith("LEFT") else "DERECHO"
            note = _NOTE_ES[FINGER_NOTES[finger] % 12]
            step = replace(step, instruction=(
                f"DOBLA el {_FINGER_ES[finger.finger.name]} {hand} ({note}) hacia la palma en cada ¡DOBLA! "
                "y vuelve a estirarlo (los demás dedos quietos)"))
        out.append(step)
    return tuple(out)


PROTOCOLS: dict[str, tuple[TraceStep, ...]] = {"v2": V2_STEPS, "focus": FOCUS_STEPS, "keys": KEYS_STEPS,
                                               "play": PLAY_STEPS, "dedos": FINGER_STEPS, "indices": INDEX_STEPS}
DEFAULT_STEPS = V2_STEPS
ALL_STEPS: dict[str, TraceStep] = {s.name: s for s in (*V1_STEPS, *V2_STEPS, *FOCUS_STEPS, *KEYS_STEPS,
                                                         *PLAY_STEPS, *INDEX_STEPS)}

# Columns up to "cue" are those of v2/v3 traces; "seq" … "baseline" were added in format v4, and
# "bend" (finger-note mode signal) later in v4: older v4 files simply lack it.
COLUMNS = (
    "t", "step", "finger", "tracked", "discontinuity", "hand_speed", "tip_y", "flex", "tip_z",
    "piano_x", "piano_y", "raw_key", "stable_key", "phase", "depth", "velocity", "events", "cue",
    "seq", "hand_status", "common_mode", "baseline", "bend",
)


def _f(v: float | None) -> str:
    return "" if v is None else f"{v:.5f}"


class TraceRecorder:
    """Collects rows on the tracking thread; the UI thread sets the current step and logs keys."""

    def __init__(self, steps: tuple[TraceStep, ...] = DEFAULT_STEPS, protocol: str = "v2",
                 config: PressConfig | None = None, mirrored: bool | None = None,
                 detector: dict[str, Any] | None = None) -> None:
        self.steps = steps
        self.detector = detector
        self.protocol = protocol
        self.config = config
        self.mirrored = mirrored
        self.step = PREPARE
        self._rows: list[tuple[str, ...]] = []
        self._hands: list[HandSample] = []
        self._keys: list[KeyEvent] = []
        self._frame_size: tuple[int, int] | None = None
        self._lock = threading.Lock()
        self._pending_cue = ""
        self._started_wall = time.strftime("%Y-%m-%dT%H:%M:%S%z")

    def mark_cue(self, fingers: tuple[FingerId, ...]) -> None:
        """Called by the UI when a cue appears; written on the next recorded frame."""
        with self._lock:
            self._pending_cue = "+".join(f.name for f in fingers)

    def on_frame(self, snapshot: TrackingSnapshot) -> None:
        presses = snapshot.presses
        if presses is None:
            return
        t = snapshot.frame.captured_at_ns / 1e9
        step = self.step
        by_finger: dict[FingerId, list[str]] = {}
        for e in presses.events:
            by_finger.setdefault(e.finger, []).append(f"{e.kind}:{e.reason}:{'' if e.key is None else e.key.name}")
        with self._lock:
            cue, self._pending_cue = self._pending_cue, ""
        state = snapshot.state  # None only in tests that fake a snapshot: no landmark rows then
        hands = [] if state is None else [sample_from_state(t, snapshot.seq, state.hands[h])
                                          for h in sorted(state.hands, key=lambda h: h.name)]
        rows = []
        for finger in PLAYING_FINGERS:
            sig = presses.signals[finger]
            fp = presses.fingers[finger]
            fk = None if snapshot.mapping is None else snapshot.mapping.fingers[finger]
            feats = sig.features or {}
            piano = None if fk is None else fk.piano_xy
            rows.append((
                f"{t:.4f}", step, finger.name, str(sig.tracked), str(sig.discontinuity), _f(sig.hand_speed),
                _f(feats.get("tip_y")), _f(feats.get("flex")), _f(feats.get("tip_z")),
                _f(None if piano is None else piano[0]), _f(None if piano is None else piano[1]),
                "" if sig.raw_key is None else sig.raw_key.name, "" if sig.stable_key is None else sig.stable_key.name,
                fp.phase.value, _f(fp.depth), _f(fp.velocity), "|".join(by_finger.get(finger, [])), cue,
                str(snapshot.seq), "" if state is None else state.hands[finger.hand].status.name, _f(sig.common_mode),
                _f(fp.baseline), _f(feats.get("bend")),
            ))
        with self._lock:
            self._rows.extend(rows)
            self._hands.extend(hands)
            self._frame_size = (snapshot.frame.width, snapshot.frame.height)

    def record_key(self, key: str, down: bool, t: float | None = None) -> None:
        """A physical key went down/up (UI thread). ``t`` defaults to now on the trace's clock."""
        with self._lock:
            self._keys.append(KeyEvent(time.perf_counter() if t is None else t, key.upper(), down))

    @property
    def key_count(self) -> int:
        with self._lock:
            return len(self._keys)

    @property
    def key_events(self) -> list[KeyEvent]:
        with self._lock:
            return list(self._keys)

    @property
    def key_downs(self) -> dict[str, int]:
        """Presses (key down) per key, for the on-screen counter."""
        with self._lock:
            out: dict[str, int] = {}
            for e in self._keys:
                if e.down:
                    out[e.key] = out.get(e.key, 0) + 1
            return out

    @property
    def row_count(self) -> int:
        with self._lock:
            return len(self._rows)

    def _meta(self, rows: list[tuple[str, ...]]) -> dict[str, Any]:
        size = self._frame_size
        return {
            "protocol": self.protocol,
            "started": self._started_wall,
            "clock": "time.perf_counter() seconds at frame capture (monotonic, arbitrary origin)",
            "frame_width": None if size is None else size[0],
            "frame_height": None if size is None else size[1],
            "aspect": None if size is None else size[1] / size[0],
            "mirrored": self.mirrored,
            "fingers": [f.name for f in PLAYING_FINGERS],
            "steps": [{"name": s.name, "duration_s": s.duration_s, "expected": [f.name for f in s.expected],
                       "cue_every_s": s.cue_every_s, "cue_lead_s": s.cue_lead_s} for s in self.steps],
            "press_config": None if self.config is None else asdict(self.config),
            "detector_config": self.detector,
            "rows": len(rows),
            "key_events": len(self._keys),
            "conventions": "see handpiano/interaction/trace_landmarks.py and docs/INVESTIGACION_PULSACIONES.md",
        }

    def write(self, path: Path) -> dict[str, Path]:
        """Writes the v4 file set next to ``path``. Refuses (FileExistsError) if ANY of them exists."""
        files = sidecar_paths(path)
        existing = [p for p in files.values() if p.exists()]
        if existing:
            raise FileExistsError(f"refusing to overwrite: {', '.join(map(str, existing))}")
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            rows, hands, keys = list(self._rows), list(self._hands), list(self._keys)
            meta = self._meta(rows)
        with path.open("x", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(COLUMNS)
            writer.writerows(rows)
        write_landmarks(files["landmarks"], hands)
        write_meta(files["meta"], meta)
        written = {"trace": path, "landmarks": files["landmarks"], "meta": files["meta"]}
        if keys:
            end_t = hands[-1].t if hands else None
            write_labels(files["labels"], labels_from_keys(keys, end_t))
            written["labels"] = files["labels"]
        return written


# -- offline replay ---------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class StepResult:
    step: str
    expected: tuple[str, ...]
    presses: dict[str, int]  # finger → presses in this step

    @property
    def intended(self) -> int:
        return sum(n for f, n in self.presses.items() if f in self.expected)

    @property
    def unintended(self) -> int:
        return sum(n for f, n in self.presses.items() if f not in self.expected)


def load_trace(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _inputs_from_rows(frame_rows: list[dict[str, str]], config: PressConfig,
                      keys: dict[str, PianoKey]) -> dict[FingerId, PressInput]:
    inputs: dict[FingerId, PressInput] = {}
    note_keys = {f: None if m is None else keys.get(m) for f, m in _note_key_names(keys).items()}
    for r in frame_rows:
        value = r.get(config.feature) or ""
        finger = FingerId[r["finger"]]
        tracked = r["tracked"] == "True"
        inputs[finger] = PressInput(
            t=float(r["t"]),
            value=float(value) if value else None,
            key=(note_keys[finger] if tracked else None) if config.finger_notes
            else keys.get(r.get("stable_key") or ""),
            tracked=tracked,
            discontinuity=r["discontinuity"] == "True",
            hand_speed=float(r["hand_speed"]) if r["hand_speed"] else 0.0,
        )
    return inputs


def _note_key_names(keys: dict[str, PianoKey]) -> dict[FingerId, str | None]:
    by_midi = {k.midi: name for name, k in keys.items()}
    return {f: by_midi.get(m) for f, m in FINGER_NOTES.items()}


def replay_frames(rows: list[dict[str, str]], config: PressConfig, layout: KeyboardLayout
                  ) -> Iterator[tuple[list[dict[str, str]], list[PressEvent], PressEngine]]:
    """Feeds a trace, frame by frame (rows sharing a timestamp), to the production PressEngine.

    The whole frame is needed at once because the common mode is computed per hand.
    Yields (rows of the frame, events of the frame, engine after the frame).
    """
    keys = {k.name: k for k in layout.keys}
    engine = PressEngine(config)
    frame: list[dict[str, str]] = []
    for r in rows:
        if frame and r["t"] != frame[0]["t"]:
            events, _ = engine.step(_inputs_from_rows(frame, config, keys))
            yield frame, events, engine
            frame = []
        frame.append(r)
    if frame:
        events, _ = engine.step(_inputs_from_rows(frame, config, keys))
        yield frame, events, engine


def replay(rows: list[dict[str, str]], config: PressConfig, layout: KeyboardLayout,
           steps: tuple[TraceStep, ...] = DEFAULT_STEPS) -> list[StepResult]:
    """Runs the production press logic over recorded signals with any config."""
    counts: dict[str, dict[str, int]] = {}
    for frame, events, _ in replay_frames(rows, config, layout):
        step = frame[0]["step"]
        for e in events:
            if e.kind == "press":
                step_counts = counts.setdefault(step, {})
                step_counts[e.finger.name] = step_counts.get(e.finger.name, 0) + 1
    expected = {name: tuple(f.name for f in s.expected) for name, s in ALL_STEPS.items()}
    expected.update({s.name: tuple(f.name for f in s.expected) for s in steps})
    seen = list(dict.fromkeys(r["step"] for r in rows))  # the order the steps were recorded in
    return [StepResult(name, expected.get(name, ()), counts.get(name, {})) for name in seen]


# -- diagnostics ------------------------------------------------------------------------------

TRANSITION_S = 1.5  # presses this soon after a step starts may still belong to the previous instruction


def _float(v: str | None) -> float:
    return float(v) if v else float("nan")


def local_excursions(rows: list[dict[str, str]], feature: str, window_s: float = 1.5) -> dict[str, np.ndarray]:
    """Per finger: signal minus its centred moving median (NaN where there is no data).

    Removes slow posture so that what is left is the fast, tap-like part.
    Rows of one finger are kept in trace order (one value per recorded frame).
    """
    by: dict[str, list[dict[str, str]]] = {}
    for r in rows:
        by.setdefault(r["finger"], []).append(r)
    out: dict[str, np.ndarray] = {}
    for finger, rs in by.items():
        t = np.array([float(r["t"]) for r in rs])
        v = np.array([_float(r[feature]) for r in rs])
        exc = np.full(len(v), np.nan)
        for i in range(len(v)):
            if np.isnan(v[i]):
                continue
            m = (np.abs(t - t[i]) <= window_s / 2) & ~np.isnan(v)
            if m.sum() >= 10:
                exc[i] = v[i] - np.median(v[m])
        out[finger] = exc
    return out


def remove_common_mode(excursions: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Subtracts, frame by frame, the median excursion of the hand's four fingers.

    A tap moves one finger; closing, tilting or rotating the hand moves all of
    them together. Requires every finger of a hand to have one value per frame
    (true for traces written by TraceRecorder).
    """
    out: dict[str, np.ndarray] = {}
    for hand in ("LEFT", "RIGHT"):
        fingers = [f for f in excursions if f.startswith(hand + "_")]
        if not fingers:
            continue
        stack = np.vstack([excursions[f] for f in fingers])
        with np.errstate(all="ignore"):
            valid = ~np.isnan(stack).all(axis=0)
            common = np.full(stack.shape[1], np.nan)
            common[valid] = np.nanmedian(stack[:, valid], axis=0)
        for f in fingers:
            out[f] = excursions[f] - common
    return out


@dataclass(frozen=True, slots=True)
class FingerStepSummary:
    step: str
    finger: str
    expected: bool
    frames: int
    valid: int  # frames with a usable signal
    signal_max: dict[str, float | None]  # None = no data (never 0)
    signal_range: dict[str, float | None]  # p95 − p5
    excursion_p95: dict[str, float | None]  # p95 |local excursion|
    excursion_p95_cm: dict[str, float | None]  # same, after removing the hand's common mode
    presses: int
    transition_presses: int  # within TRANSITION_S of the step start
    rejections: dict[str, int]
    longest_pressed_s: float

    @property
    def missing(self) -> int:
        return self.frames - self.valid


def _p95_abs(x: np.ndarray) -> float | None:
    x = x[~np.isnan(x)]
    return None if len(x) < 48 else float(np.percentile(np.abs(x), 95))


def summarize(rows: list[dict[str, str]], config: PressConfig, layout: KeyboardLayout) -> list[FingerStepSummary]:
    """Per step × finger diagnostics. Presses come from replaying ``config`` over the trace."""
    from handpiano.interaction.press_features import FEATURES

    step_start: dict[str, float] = {}
    for r in rows:
        step_start.setdefault(r["step"], float(r["t"]))
    exc = {f: local_excursions(rows, f) for f in FEATURES}
    exc_cm = {f: remove_common_mode(exc[f]) for f in FEATURES}

    presses: Counter[tuple[str, str]] = Counter()
    transition: Counter[tuple[str, str]] = Counter()
    rejections: dict[tuple[str, str], Counter[str]] = {}
    longest: dict[tuple[str, str], float] = {}
    pressed_since: dict[str, tuple[float, str]] = {}
    index_in_finger: Counter[str] = Counter()
    positions: dict[tuple[str, str], list[int]] = {}
    for frame, events, engine in replay_frames(rows, config, layout):
        step, t = frame[0]["step"], float(frame[0]["t"])
        for e in events:
            key = (step, e.finger.name)
            if e.kind == "press":
                presses[key] += 1
                if t - step_start[step] < TRANSITION_S:
                    transition[key] += 1
            elif e.kind == "rejected":
                rejections.setdefault(key, Counter())[e.reason] += 1
        for r in frame:
            f = r["finger"]
            positions.setdefault((step, f), []).append(index_in_finger[f])
            index_in_finger[f] += 1
            down = engine.machines[FingerId[f]].phase.value in ("PRESSED", "RELEASE_CANDIDATE")
            if down and f not in pressed_since:
                pressed_since[f] = (t, step)
            elif not down and f in pressed_since:
                start, start_step = pressed_since.pop(f)
                longest[(start_step, f)] = max(longest.get((start_step, f), 0.0), t - start)

    out = []
    order = list(dict.fromkeys(r["step"] for r in rows))
    for step in order:
        expected = {f.name for f in ALL_STEPS[step].expected} if step in ALL_STEPS else set()
        for finger in (f.name for f in PLAYING_FINGERS):
            idx = np.array(positions.get((step, finger), []), dtype=int)
            if len(idx) == 0:
                continue
            maxima: dict[str, float | None] = {}
            ranges: dict[str, float | None] = {}
            p95: dict[str, float | None] = {}
            p95_cm: dict[str, float | None] = {}
            raw_by_feature = {}
            for feat in FEATURES:
                vals = np.array([_float(r[feat]) for r in rows if r["finger"] == finger and r["step"] == step])
                raw_by_feature[feat] = vals
                v = vals[~np.isnan(vals)]
                maxima[feat] = None if len(v) == 0 else float(v.max())
                ranges[feat] = None if len(v) < 48 else float(np.percentile(v, 95) - np.percentile(v, 5))
                p95[feat] = _p95_abs(exc[feat][finger][idx])
                p95_cm[feat] = _p95_abs(exc_cm[feat][finger][idx]) if finger in exc_cm[feat] else None
            valid = int(np.sum(~np.isnan(raw_by_feature[config.feature])))
            out.append(FingerStepSummary(
                step=step, finger=finger, expected=finger in expected, frames=len(idx), valid=valid,
                signal_max=maxima, signal_range=ranges, excursion_p95=p95, excursion_p95_cm=p95_cm,
                presses=presses[(step, finger)], transition_presses=transition[(step, finger)],
                rejections=dict(rejections.get((step, finger), {})), longest_pressed_s=longest.get((step, finger), 0.0),
            ))
    return out


@dataclass(frozen=True, slots=True)
class CueResult:
    finger: str
    cues: int
    hits: int  # cues answered by a press of that finger
    missed: int  # cues with no such press
    extra: int  # presses of that finger not assigned to any of its cues (anywhere in the trace)
    offsets: tuple[float, ...]  # press time − cue time of every hit, seconds

    @property
    def hit_rate(self) -> float | None:
        return None if self.cues == 0 else self.hits / self.cues


def _replay_presses(rows: list[dict[str, str]], config: PressConfig, layout: KeyboardLayout) -> dict[str, list[float]]:
    press_times: dict[str, list[float]] = {}
    for _, events, _ in replay_frames(rows, config, layout):
        for e in events:
            if e.kind == "press":
                press_times.setdefault(e.finger.name, []).append(e.t)
    return press_times


def match_cues(rows: list[dict[str, str]], config: PressConfig, layout: KeyboardLayout,
               window_s: float | None = None) -> list[CueResult]:
    """Matches replayed presses to v2 cues, one-to-one.

    Each press is assigned to the NEAREST cue of the same finger, if it is within
    ``window_s`` of it (default: half the cue interval of that cue's step, so
    windows never overlap and anticipating the beat still counts); each cue keeps
    at most one press (the closest). A press is therefore counted at most once,
    either as a hit or as extra. Empty for traces without cues.
    """
    cues: dict[str, list[tuple[float, float]]] = {}  # finger → [(cue time, window)]
    seen_cue_t: set[str] = set()
    for r in rows:
        cue = r.get("cue") or ""
        if cue and r["t"] not in seen_cue_t:
            seen_cue_t.add(r["t"])
            step = ALL_STEPS.get(r["step"])
            interval = step.cue_every_s if step is not None and step.cue_every_s else 1.5
            w = window_s if window_s is not None else interval / 2
            for name in cue.split("+"):
                cues.setdefault(name, []).append((float(r["t"]), w))
    if not cues:
        return []
    press_times = _replay_presses(rows, config, layout)
    results = []
    for finger in (f.name for f in PLAYING_FINGERS):
        finger_cues = cues.get(finger, [])
        presses = press_times.get(finger, [])
        best: dict[int, tuple[float, int]] = {}  # cue index → (|offset|, press index)
        for pi, p in enumerate(presses):
            candidates = [(abs(p - c), ci) for ci, (c, w) in enumerate(finger_cues) if abs(p - c) <= w]
            if not candidates:
                continue
            dist, ci = min(candidates)
            if ci not in best or dist < best[ci][0]:
                best[ci] = (dist, pi)
        offsets = tuple(sorted(presses[pi] - finger_cues[ci][0] for ci, (_, pi) in best.items()))
        hits = len(best)
        results.append(CueResult(finger, len(finger_cues), hits, len(finger_cues) - hits, len(presses) - hits, offsets))
    return results


_FINGER_SHORT = {"INDEX": "índice", "MIDDLE": "medio", "RING": "anular", "PINKY": "meñique"}


def cue_report(results: list[CueResult]) -> list[str]:
    """What the app tells the player after a guided recording: taps recognised, per finger."""
    asked = [r for r in results if r.cues]
    if not asked:
        return []
    cues = sum(r.cues for r in asked)
    hits = sum(r.hits for r in asked)
    extra = sum(r.extra for r in results)
    per_finger = " · ".join(
        f"{_FINGER_SHORT[r.finger.split('_')[1]]} {'izq' if r.finger.startswith('LEFT') else 'der'} {r.hits}/{r.cues}"
        for r in asked)
    return [f"Toques reconocidos: {hits}/{cues} ({hits / cues:.0%}) · notas de más: {extra}", per_finger]


@dataclass(frozen=True, slots=True)
class PressedSpan:
    finger: str
    step: str
    start: float
    duration: float
    min_depth: float | None
    median_depth: float | None
    idle_depth_before: float | None  # median depth while IDLE in the second before the press
    idle_above_press_on: float | None  # fraction of those idle frames already above press_on
    gap_before: bool  # finger untracked at some point in the 3 s before the press
    ended_by: str  # "release", "tracking_lost", "no_event" (outside/uncalibrated press), "trace_end"
    # Other fingers of the same hand whose raw depth also rose above press_on within ±0.25 s of the start.
    others_rising: int = 0
    # Whether the step asked this finger to press.
    instructed: bool = False

    @property
    def cause(self) -> str:
        """Explanation supported by the recorded numbers (not a default label):

        * stale_baseline[_after_gap]: the finger was already above press_on while idle before;
        * hand_comovement: ≥ 2 other fingers of the hand rose together with it (co-occurrence, which
          points to a whole-hand movement but does not prove it);
        * instructed_hold: the finger the step asked for, pressed and not released (a held or
          unreleased tap: the trace cannot tell which);
        * uninstructed_single: a finger that was NOT asked to press went down alone and stayed
          (e.g. a neighbour dragged by the tapping finger).
        """
        if self.idle_above_press_on is not None and self.idle_above_press_on >= 0.8:
            return "stale_baseline_after_gap" if self.gap_before else "stale_baseline"
        if self.others_rising >= 2:
            return "hand_comovement"
        return "instructed_hold" if self.instructed else "uninstructed_single"


def pressed_spans(rows: list[dict[str, str]], config: PressConfig, layout: KeyboardLayout) -> list[PressedSpan]:
    """Every continuous PRESSED/RELEASE_CANDIDATE span of the replayed machines, with context."""
    history: dict[str, list[tuple[float, str, float | None, bool]]] = {f.name: [] for f in PLAYING_FINGERS}
    raw_hist: dict[str, list[tuple[float, float | None]]] = {f.name: [] for f in PLAYING_FINGERS}
    open_spans: dict[str, dict] = {}
    out: list[PressedSpan] = []

    def rose(finger: str, t0: float, t1: float) -> bool:
        before = [d for t, d in raw_hist[finger] if t0 - 0.5 <= t < t0 and d is not None]
        during = [d for t, d in raw_hist[finger] if t0 <= t <= t1 and d is not None]
        return bool(before and during) and max(during) - float(np.median(before)) > config.press_on

    def close(finger: str, t: float, ended_by: str) -> None:
        o = open_spans.pop(finger)
        depths = [d for d in o["depths"] if d is not None]
        others = [f.name for f in PLAYING_FINGERS if f.name != finger and f.hand is FingerId[finger].hand]
        start = o["start"]
        rising = sum(1 for g in others if rose(g, start - 0.25, start + 0.25))
        step = ALL_STEPS.get(o["step"])
        instructed = step is not None and FingerId[finger] in step.expected
        out.append(PressedSpan(
            finger, o["step"], start, t - start,
            min(depths) if depths else None, float(np.median(depths)) if depths else None,
            o["idle_before"], o["idle_above"], o["gap"], ended_by, rising, instructed,
        ))

    for frame, events, engine in replay_frames(rows, config, layout):
        t = float(frame[0]["t"])
        released = {e.finger.name: e.reason for e in events if e.kind == "release"}
        for r in frame:
            f = r["finger"]
            machine = engine.machines[FingerId[f]]
            value = r.get(config.feature) or ""
            raw_hist[f].append((t, machine.depth_of(float(value)) if value else None))
            down = machine.phase.value in ("PRESSED", "RELEASE_CANDIDATE")
            depth = machine.view(None).depth
            if down and f not in open_spans:
                before = [h for h in history[f] if t - 1.0 <= h[0] < t and h[1] == "IDLE" and h[2] is not None]
                idle_depths = [h[2] for h in before if h[2] is not None]
                open_spans[f] = {
                    "step": r["step"], "start": t, "depths": [],
                    "idle_before": float(np.median(idle_depths)) if idle_depths else None,
                    "idle_above": (sum(1 for d in idle_depths if d > config.press_on) / len(idle_depths))
                    if idle_depths else None,
                    "gap": any(not h[3] for h in history[f] if t - 3.0 <= h[0] < t),
                }
            if f in open_spans:
                if down:
                    open_spans[f]["depths"].append(depth)
                else:
                    close(f, t, released.get(f, "no_event"))
            history[f].append((t, machine.phase.value, depth, r["tracked"] == "True"))
            if len(history[f]) > 200:
                del history[f][:100]
    end = float(rows[-1]["t"]) if rows else 0.0
    for f in list(open_spans):
        close(f, end, "trace_end")
    return out


# -- per-cue diagnosis -----------------------------------------------------------------------

MISS_REASONS = (
    "not_tracked", "already_pressed", "relearning", "rejected", "common_mode_removed",
    "moved_in_other_signal", "too_slow", "no_visible_movement",
)


@dataclass(frozen=True, slots=True)
class CueDiagnosis:
    finger: str
    step: str
    t: float  # cue time, seconds after the first trace row
    hit: bool
    reason: str | None  # one of MISS_REASONS for misses (rejected: "rejected:<reason>"), None for hits
    tracked_fraction: float
    raw_depth_max: float | None  # production signal minus baseline, before common mode
    effective_depth_max: float | None  # what the decision used
    excursions: dict[str, float | None]  # per signal: max in window − median just before


def cue_diagnostics(rows: list[dict[str, str]], config: PressConfig, layout: KeyboardLayout) -> list[CueDiagnosis]:
    """Why each v2 cue was (not) answered, decided from what the replayed detector saw.

    Reasons are checked in this order and the first that applies is reported:
    not_tracked (< 50 % tracked in the window) · already_pressed · relearning ·
    rejected:<reason> · common_mode_removed (raw depth > press_on but effective ≤
    press_on) · moved_in_other_signal (production signal moved < press_on but
    another signal moved ≥ press_on) · too_slow (deep enough, never faster than
    min_velocity) · no_visible_movement (no signal moved ≥ press_on).
    """
    from handpiano.interaction.press_features import FEATURES

    per: dict[str, list[dict]] = {f.name: [] for f in PLAYING_FINGERS}
    for frame, events, engine in replay_frames(rows, config, layout):
        t = float(frame[0]["t"])
        evs: dict[str, list[str]] = {}
        for e in events:
            evs.setdefault(e.finger.name, []).append(f"{e.kind}:{e.reason}")
        for r in frame:
            f = r["finger"]
            m = engine.machines[FingerId[f]]
            value = r.get(config.feature) or ""
            per[f].append({
                "t": t, "tracked": r["tracked"] == "True", "phase": m.phase.value, "relearn": m._relearning,
                "raw": m.depth_of(float(value)) if value else None, "eff": m.view(None).depth,
                "vel": m.view(None).velocity, "events": evs.get(f, []),
                "feats": {k: _float(r.get(k)) for k in FEATURES},
            })
    matched = _matched_cues(rows, config, layout)
    t0 = float(rows[0]["t"]) if rows else 0.0
    out: list[CueDiagnosis] = []
    for finger, cue_t, w, step in _cue_list(rows):
        frames = [x for x in per[finger] if cue_t - w <= x["t"] <= cue_t + w]
        before = [x for x in per[finger] if cue_t - w - 0.5 <= x["t"] < cue_t - w + 0.05]
        tracked = float(np.mean([x["tracked"] for x in frames])) if frames else 0.0
        raws = [x["raw"] for x in frames if x["raw"] is not None]
        effs = [x["eff"] for x in frames if x["eff"] is not None]
        vels = [x["vel"] for x in frames if x["vel"] is not None]
        exc: dict[str, float | None] = {}
        for k in FEATURES:
            a = np.array([x["feats"][k] for x in frames])
            b = np.array([x["feats"][k] for x in before])
            a, b = a[~np.isnan(a)], b[~np.isnan(b)]
            exc[k] = None if len(a) < 3 or len(b) < 2 else float(a.max() - np.median(b))
        raw_max = max(raws) if raws else None
        eff_max = max(effs) if effs else None
        hit = (finger, cue_t) in matched
        reason = None
        if not hit:
            rejected = [e.split(":")[1] for x in frames for e in x["events"] if e.startswith("rejected")]
            other = [v for k in FEATURES if k != config.feature and (v := exc[k]) is not None]
            if tracked < 0.5:
                reason = "not_tracked"
            elif any(x["phase"] in ("PRESSED", "RELEASE_CANDIDATE") for x in frames):
                reason = "already_pressed"
            elif any(x["relearn"] for x in frames):
                reason = "relearning"
            elif rejected:
                reason = "rejected:" + rejected[0]
            elif raw_max is not None and raw_max > config.press_on and (eff_max is None or eff_max <= config.press_on):
                reason = "common_mode_removed"
            elif eff_max is not None and eff_max > config.press_on and max(vels, default=0.0) <= config.min_velocity:
                reason = "too_slow"
            elif other and max(other) >= config.press_on:
                reason = "moved_in_other_signal"
            else:
                reason = "no_visible_movement"
        out.append(CueDiagnosis(finger, step, cue_t - t0, hit, reason, tracked, raw_max, eff_max, exc))
    return out


def _cue_list(rows: list[dict[str, str]]) -> list[tuple[str, float, float, str]]:
    """(finger, cue time, matching half-window, step) for every v2 cue."""
    out: list[tuple[str, float, float, str]] = []
    seen: set[str] = set()
    for r in rows:
        cue = r.get("cue") or ""
        if cue and r["t"] not in seen:
            seen.add(r["t"])
            step = ALL_STEPS.get(r["step"])
            w = (step.cue_every_s if step is not None and step.cue_every_s else 1.5) / 2
            for name in cue.split("+"):
                out.append((name, float(r["t"]), w, r["step"]))
    return out


def _matched_cues(rows: list[dict[str, str]], config: PressConfig, layout: KeyboardLayout) -> set[tuple[str, float]]:
    """(finger, cue time) of the cues answered by a press, with the same one-to-one rule as match_cues."""
    press_times = _replay_presses(rows, config, layout)
    by_finger: dict[str, list[tuple[float, float]]] = {}
    for finger, c, w, _ in _cue_list(rows):
        by_finger.setdefault(finger, []).append((c, w))
    matched: set[tuple[str, float]] = set()
    for finger, cues in by_finger.items():
        best: dict[int, tuple[float, int]] = {}
        for pi, p in enumerate(press_times.get(finger, [])):
            candidates = [(abs(p - c), ci) for ci, (c, w) in enumerate(cues) if abs(p - c) <= w]
            if candidates:
                dist, ci = min(candidates)
                if ci not in best or dist < best[ci][0]:
                    best[ci] = (dist, pi)
        matched.update((finger, cues[ci][0]) for ci in best)
    return matched


def press_offsets(rows: list[dict[str, str]], config: PressConfig, layout: KeyboardLayout) -> dict[str, list[float]]:
    """Per finger: signed offset (s) of each of its presses to the NEAREST of its cues (no window).

    Shows where presses fall relative to cues, independently of the matching window.
    """
    press_times = _replay_presses(rows, config, layout)
    cues: dict[str, list[float]] = {}
    for finger, c, _, _ in _cue_list(rows):
        cues.setdefault(finger, []).append(c)
    out: dict[str, list[float]] = {}
    for finger, presses in press_times.items():
        if finger in cues:
            out[finger] = [min((p - c for c in cues[finger]), key=abs) for p in presses]
    return out
