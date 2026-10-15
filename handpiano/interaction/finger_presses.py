"""Runs the eight per-finger press machines on each tracking frame.

Inputs per finger come from the tracker (smoothed landmarks, TRACKED status),
the key mapper (piano position) and ``StableKeys`` (key with hysteresis).
Runs on the tracking thread, right after the key mapping.

Common mode (``PressConfig.common_mode = "hand_low"``)
------------------------------------------------------
Closing, tilting or rotating a hand moves all its fingers at once; a tap moves
one or two. Per hand and per frame:

    cm = second-lowest depth among the hand's fingers with data (≥ 3 of them)

A tap or a two-finger chord leaves at least two fingers low, so cm ≈ 0; a
posture change moves all four, so cm > 0. The depth each machine uses to DECIDE
press/release is reduced by ``common_mode_gain`` × cm, never below the finger's
resting level. The baselines keep learning from the raw signal, so a posture
that is held is absorbed and cm returns to ≈ 0 (subtracting from the signal
itself would hide the new posture from the baseline forever and cancel later
taps). Fingers do not have to move equally: a posture change is suppressed as
long as no finger moves more than ``common_mode_gain`` (1.5) times the hand's
common movement.

Limits: a chord of three or four fingers of the same hand is suppressed like a
posture change; a posture change in which one finger moves more than 1.5× the
others can still produce a press; with fewer than three fingers of a hand in
view no common mode is applied.

Alternative ``"idle_min"`` (evaluated on traza_v2/v3, not the default yet): for
each finger, cm = gain × the MINIMUM depth among the OTHER fingers of its hand
that are IDLE (≥ 2 of them). Fingers holding a press are excluded (their depth is
a note, not posture; on traza_v3 including them cancelled every LEFT_RING tap),
and a 2–3-finger chord keeps at least one idle finger low, so it is not
suppressed; closing the whole hand still moves every idle finger.

``PressEngine`` is the single implementation used both live (``PressDetector``)
and when replaying recorded traces, so replays measure the production logic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

from handpiano.interaction.press_detector import (
    FingerPress,
    PressPhase,
    FingerPressMachine,
    PressConfig,
    PressEvent,
    PressInput,
)
from handpiano.interaction.press_features import hand_scale, press_features
from handpiano.interaction.stable_keys import StableKeys
from handpiano.instrument.keyboard import KeyboardLayout, PianoKey
from handpiano.mapping.key_mapper import MappingState
from handpiano.tracking.finger_tracker import TrackingState, TrackStatus
from handpiano.tracking.landmarks import PLAYING_FINGERS, Finger, FingerId, Handedness, Landmark


# Finger-note mode: one octave, C4 (left pinky) → C5 (right pinky), low to high from left to right.
FINGER_NOTES: dict[FingerId, int] = {
    FingerId.LEFT_PINKY: 60, FingerId.LEFT_RING: 62, FingerId.LEFT_MIDDLE: 64, FingerId.LEFT_INDEX: 65,
    FingerId.RIGHT_INDEX: 67, FingerId.RIGHT_MIDDLE: 69, FingerId.RIGHT_RING: 71, FingerId.RIGHT_PINKY: 72,
}


def finger_note_keys(layout: KeyboardLayout) -> dict[FingerId, PianoKey | None]:
    """The key each finger plays in finger-note mode (None if the keyboard does not contain it)."""
    by_midi = {k.midi: k for k in layout.keys}
    return {f: by_midi.get(m) for f, m in FINGER_NOTES.items()}


class PressEngine:
    """The eight machines plus the per-hand common-mode subtraction."""

    def __init__(self, config: PressConfig) -> None:
        self.config = config
        self.machines: dict[FingerId, FingerPressMachine] = {f: FingerPressMachine(f, config) for f in PLAYING_FINGERS}

    def common_modes(self, inputs: dict[FingerId, PressInput]) -> dict[FingerId, float | None]:
        """Per finger: the amount subtracted from its signal this frame (None = not applied)."""
        out: dict[FingerId, float | None] = {f: None for f in inputs}
        if self.config.common_mode == "off":
            return out
        depths: dict[FingerId, float] = {}
        for f, inp in inputs.items():
            if inp.tracked and inp.value is not None and not inp.discontinuity:
                d = self.machines[f].depth_of(inp.value)
                if d is not None:
                    depths[f] = d
        if self.config.common_mode == "idle_min":
            for f in depths:
                idle_others = [d for g, d in depths.items()
                               if g.hand is f.hand and g is not f and self.machines[g].phase is PressPhase.IDLE]
                if len(idle_others) >= 2:
                    out[f] = max(0.0, min(idle_others)) * self.config.common_mode_gain
            return out
        for hand in {f.hand for f in inputs}:
            hand_depths = sorted(d for f, d in depths.items() if f.hand is hand)
            if len(hand_depths) < 3:
                continue
            cm = max(0.0, hand_depths[1]) * self.config.common_mode_gain  # second lowest
            for f in depths:
                if f.hand is hand:
                    out[f] = cm
        return out

    def step(self, inputs: dict[FingerId, PressInput]) -> tuple[list[PressEvent], dict[FingerId, float | None]]:
        cm = self.common_modes(inputs)
        events: list[PressEvent] = []
        for f, inp in inputs.items():
            if self.config.only_index and f.finger is not Finger.INDEX:
                continue
            c = cm[f]
            events.extend(self.machines[f].update(replace(inp, common_mode=c) if c else inp))
        return events, cm


@dataclass(frozen=True, slots=True)
class FingerSignals:
    """Everything the detector saw for one finger this frame (for diagnostics and traces)."""

    tracked: bool
    features: dict[str, float] | None
    raw_key: PianoKey | None
    stable_key: PianoKey | None
    hand_speed: float | None
    discontinuity: bool
    common_mode: float | None = None  # subtracted from this finger's signal this frame (None = not applied)


@dataclass(frozen=True, slots=True)
class PressFrame:
    fingers: dict[FingerId, FingerPress]
    signals: dict[FingerId, FingerSignals]
    events: tuple[PressEvent, ...]
    t: float = 0.0  # capture time of the frame (perf_counter seconds), the clock of the events

    @property
    def pressed_keys(self) -> dict[int, tuple[FingerId, ...]]:
        """MIDI → fingers currently holding that key pressed (one entry per key)."""
        out: dict[int, list[FingerId]] = {}
        for finger in PLAYING_FINGERS:
            fp = self.fingers[finger]
            if fp.phase.value in ("PRESSED", "RELEASE_CANDIDATE") and fp.key is not None:
                out.setdefault(fp.key.midi, []).append(finger)
        return {m: tuple(f) for m, f in out.items()}


@dataclass(slots=True)
class PressDetector:
    layout: KeyboardLayout
    config: PressConfig = field(default_factory=PressConfig)
    key_margin: float = 0.08
    # A wrist jump larger than this many hand sizes per frame is a discontinuity
    # (re-detection, identity swap), not movement.
    max_wrist_jump: float = 1.0
    _engine: PressEngine | None = None
    _stable: StableKeys | None = None
    _wrist: dict[Handedness, tuple[float, float, float]] = field(default_factory=dict)
    _note_keys: dict[FingerId, PianoKey | None] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._engine = PressEngine(self.config)
        self._stable = StableKeys(self.layout, self.key_margin)
        self._note_keys = finger_note_keys(self.layout)

    def _hand_motion(self, hand: Handedness, landmarks, t: float, aspect: float) -> tuple[float | None, bool]:
        """(wrist speed in hand sizes/s, discontinuity) for this hand."""
        scale = hand_scale(landmarks, aspect)
        w = landmarks[Landmark.WRIST]
        prev = self._wrist.get(hand)
        self._wrist[hand] = (float(w[0]), float(w[1]) * aspect, t)
        if prev is None or scale < 1e-6 or t <= prev[2]:
            return None, False
        jump = math.hypot(float(w[0]) - prev[0], float(w[1]) * aspect - prev[1]) / scale
        return jump / (t - prev[2]), jump > self.max_wrist_jump

    def update(self, t: float, state: TrackingState, mapping: MappingState | None, aspect: float) -> PressFrame:
        assert self._stable is not None
        calibrated = mapping is not None and mapping.calibrated
        hand_info: dict[Handedness, tuple[float | None, bool]] = {}
        for hand, hs in state.hands.items():
            if hs.status is TrackStatus.TRACKED and hs.landmarks is not None:
                speed, jump = self._hand_motion(hand, hs.landmarks, t, aspect)
                # A frame where temporal identity overrode MediaPipe's label is treated as
                # a discontinuity: which hand this is was uncertain.
                hand_info[hand] = (speed, jump or hs.label_overridden)
            else:
                self._wrist.pop(hand, None)

        assert self._engine is not None
        inputs: dict[FingerId, PressInput] = {}
        partial: dict[FingerId, FingerSignals] = {}
        for finger in PLAYING_FINGERS:
            hs = state.hands[finger.hand]
            tracked = state.fingers[finger].status is TrackStatus.TRACKED and hs.landmarks is not None
            features = press_features(hs.landmarks, finger.finger, aspect) if tracked and hs.landmarks is not None else None
            fk = None if mapping is None else mapping.fingers[finger]
            raw_key = None if fk is None else fk.key
            stable_key = self._stable.update(finger, None if fk is None else fk.piano_xy) if tracked else None
            if not tracked:
                self._stable.reset(finger)
            speed, discontinuity = hand_info.get(finger.hand, (None, False))
            note_mode = self.config.finger_notes
            inp = PressInput(
                t=t,
                value=None if features is None else features[self.config.feature],
                key=(self._note_keys[finger] if tracked else None) if note_mode else stable_key,
                tracked=tracked,
                discontinuity=discontinuity,
                hand_speed=0.0 if speed is None else speed,
                calibrated=True if note_mode else calibrated,
            )
            inputs[finger] = inp
            partial[finger] = FingerSignals(tracked, features, raw_key, stable_key, speed, discontinuity)
        events, cm = self._engine.step(inputs)
        views = {f: self._engine.machines[f].view(inputs[f].key) for f in PLAYING_FINGERS}
        signals = {f: replace(partial[f], common_mode=cm[f]) for f in PLAYING_FINGERS}
        return PressFrame(views, signals, tuple(events), t)
