"""Per-finger press detection (no sound: it only produces press/release events).

Each of the eight fingers has its own state machine, history and latched key.
Nothing one finger does changes another finger's state.

Signal
------
``d`` = press signal (see ``press_features``) minus that finger's resting
baseline, in hand-size units. The baseline follows the finger slowly while it
is idle and still, so posture drift is not a press. ``v`` = dd/dt.

States
------
    IDLE ──(d > press_on and v > min_velocity)──▶ PRESS_CANDIDATE
    PRESS_CANDIDATE ──(d > press_on for min_press_frames)──▶ PRESSED      → press event
    PRESS_CANDIDATE ──(d ≤ press_on)──▶ IDLE                              (rejected: too_short)
    PRESSED ──(d < release_off)──▶ RELEASE_CANDIDATE
    RELEASE_CANDIDATE ──(d < release_off for min_release_frames)──▶ IDLE  → release event
    RELEASE_CANDIDATE ──(d ≥ release_off)──▶ PRESSED

* ``release_off < press_on`` (hysteresis): noise around one threshold cannot
  produce press/release/press chatter.
* A finger has to be released before it can press again, so holding still never
  repeats a note; after a release it waits ``refractory_s`` (per finger only).
* The key is latched when the press MOTION starts (the last idle frame), because
  the tap itself moves the fingertip in the image and can cross a key boundary.
  Pressing outside the keyboard is rejected (reason ``outside``).
* Tracking loss (HELD/LOST), a time gap or a hand-identity discontinuity resets
  the finger: a pressed finger is released (reason ``tracking_lost``) and the
  velocity history is discarded, so stale or discontinuous data never fires.
* While the whole hand moves fast, new presses are suppressed (``hand_moving``):
  rotating a hand changes hand-relative signals too.

Baseline (resting reference), revised after traza_v2.csv
--------------------------------------------------------
* While IDLE and still (|v| < baseline_max_speed) the baseline learns the
  current posture whatever ``d`` is, faster (``baseline_recover_tau_s``) when the
  finger is far from it. The original rule (learn only while d < press_on) never
  recovered from a posture change larger than press_on and left fingers PRESSED
  for 10–20 s. It is still frozen during PRESS_CANDIDATE / PRESSED /
  RELEASE_CANDIDATE, so an active press is never learnt as rest.
* On tracking loss or discontinuity the baseline is forgotten
  (``relearn_after_loss``) and presses stay disarmed until ``relearn_still_frames``
  still frames have re-established it: a hand that returns in another posture
  never reuses the old reference.
* ``max_press_s`` (off by default) force-releases very long presses. It also
  cuts legitimately sustained notes, so it is only a configurable safety net.

``PressConfig.legacy()`` (baseline_any_depth=False, relearn_after_loss=False,
common_mode="off") reproduces the original detector exactly, for A/B replays.

All thresholds are in hand-size units (wrist → middle knuckle) and are
provisional until validated on recorded traces.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from handpiano.instrument.keyboard import PianoKey
from handpiano.tracking.landmarks import FingerId


class PressPhase(Enum):
    IDLE = "IDLE"
    PRESS_CANDIDATE = "PRESS_CANDIDATE"
    PRESSED = "PRESSED"
    RELEASE_CANDIDATE = "RELEASE_CANDIDATE"


@dataclass(frozen=True, slots=True)
class PressConfig:
    feature: str = "tip_y"
    press_on: float = 0.12
    release_off: float = 0.06
    min_velocity: float = 0.8  # hand sizes per second, at the candidate frame
    # Above this downward speed the press motion has started: the latched key stops updating.
    onset_velocity: float = 0.3
    min_press_frames: int = 2
    min_release_frames: int = 2
    refractory_s: float = 0.08
    # Baseline: EMA time constant while idle and still.
    baseline_tau_s: float = 1.0
    baseline_max_speed: float = 0.3
    # Frames of valid data needed after a reset before presses are allowed.
    warmup_frames: int = 3
    max_gap_s: float = 0.15
    max_hand_speed: float = 3.0  # hand sizes per second (wrist speed)
    # --- reference recovery and common mode (see module docstring) ---
    baseline_any_depth: bool = True
    baseline_recover_tau_s: float = 0.3
    relearn_after_loss: bool = True
    relearn_still_frames: int = 4
    # Per hand common-mode subtraction: "hand_low" (default), "idle_min" (alternative) or "off" (legacy).
    common_mode: str = "hand_low"
    common_mode_gain: float = 1.5
    max_press_s: float | None = None
    # Only the two index fingers play (--solo-indices); the other machines are never updated and stay IDLE.
    only_index: bool = False
    # Finger-note mode (--dedo-nota): every finger plays its own fixed note (FINGER_NOTES), wherever it is.
    finger_notes: bool = False

    @classmethod
    def finger_note_mode(cls, **overrides: object) -> PressConfig:
        """Hands facing the camera, fingers up; a note = bending a finger towards the palm.

        Signal ``flex`` (apparent finger length, knuckle → tip, in palm lengths, negated): bending
        towards a camera that faces the palm foreshortens the finger from ~1.4 to well under 1 palm
        length, while the 2D joint angle barely changes until the tip passes below the knuckle.
        press_on 0.35 / release_off 0.175 palm lengths, measured on notas_3 (guided ``dedos`` protocol,
        40 cues, 2026-10-15): 0.5 → 18/40 recognised (8 extra presses); 0.30–0.40 → 32–33/40 (18–23
        extra), a plateau, so 0.35 is not a lucky value. On recordings with a wrong posture (notas_1,
        notas_2) 0.35 adds at most 3 extra presses. Confirmed on a second, independent recording
        (notas_4, posture right 92–96 % of the time): 33/40. refractory_s 0.25 (same finger, ≥ 0.25 s
        between notes) removes 2–4 double triggers in both recordings without losing hits; the fastest
        repeated note in the bundled songs is 0.375 s. One player, one camera.
        """
        values: dict[str, object] = {"feature": "flex", "press_on": 0.35, "release_off": 0.175, "refractory_s": 0.25,
                                     "finger_notes": True}
        values.update(overrides)
        return cls(**values)  # type: ignore[arg-type]

    @classmethod
    def legacy(cls, **overrides: object) -> PressConfig:
        """The detector as it was before the traza_v2.csv fixes (for A/B comparisons)."""
        values: dict[str, object] = {
            "baseline_any_depth": False, "relearn_after_loss": False, "common_mode": "off",
        }
        values.update(overrides)
        return cls(**values)  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class PressInput:
    """One frame of data for one finger."""

    t: float
    value: float | None  # press signal; None = not usable this frame
    key: PianoKey | None  # (stable) key under the fingertip; None outside / uncalibrated
    tracked: bool  # fresh landmarks this frame (TRACKED, not HELD/LOST)
    discontinuity: bool = False  # hand identity or history broke this frame
    hand_speed: float = 0.0
    calibrated: bool = True
    # Hand common mode for this frame (see finger_presses). It only lowers the depth used
    # to decide press/release; the baseline keeps learning from the raw signal.
    common_mode: float = 0.0


@dataclass(frozen=True, slots=True)
class PressEvent:
    kind: str  # "press", "release", "rejected"
    finger: FingerId
    t: float
    key: PianoKey | None
    reason: str  # "ok", "outside", "uncalibrated", "too_short", "refractory", "hand_moving", "tracking_lost"
    depth: float | None
    velocity: float | None


@dataclass(frozen=True, slots=True)
class FingerPress:
    """Read-only view of one finger's detector state, for UI and diagnostics."""

    phase: PressPhase
    key: PianoKey | None  # latched key while pressed, else the current key
    depth: float | None
    velocity: float | None
    baseline: float | None
    last_reason: str | None


@dataclass(slots=True)
class FingerPressMachine:
    finger: FingerId
    config: PressConfig = field(default_factory=PressConfig)
    phase: PressPhase = PressPhase.IDLE
    baseline: float | None = None
    _prev_t: float | None = None
    _prev_value: float | None = None
    _velocity: float | None = None
    _depth: float | None = None
    _valid_frames: int = 0
    _count: int = 0
    _latched: PianoKey | None = None
    _idle_key: PianoKey | None = None
    _pressed_key: PianoKey | None = None
    _released_at: float | None = None
    _last_reason: str | None = None
    # A blocked press gesture is reported once, until the finger comes back up.
    _blocked: bool = False
    # After a reset with relearn: presses stay disarmed until enough still frames validate the baseline.
    _relearning: bool = False
    _still_frames: int = 0
    _pressed_at: float | None = None

    def depth_of(self, value: float | None) -> float | None:
        """Depth this value would have against the current baseline (None without one)."""
        if value is None or self.baseline is None:
            return None
        return value - self.baseline

    def view(self, key: PianoKey | None) -> FingerPress:
        shown = self._pressed_key if self.phase in (PressPhase.PRESSED, PressPhase.RELEASE_CANDIDATE) else key
        return FingerPress(self.phase, shown, self._depth, self._velocity, self.baseline, self._last_reason)

    def _event(self, kind: str, inp: PressInput, key: PianoKey | None, reason: str) -> PressEvent:
        self._last_reason = reason
        return PressEvent(kind, self.finger, inp.t, key, reason, self._depth, self._velocity)

    def _reset(self, inp: PressInput, events: list[PressEvent]) -> None:
        if self.phase in (PressPhase.PRESSED, PressPhase.RELEASE_CANDIDATE):
            events.append(self._event("release", inp, self._pressed_key, "tracking_lost"))
            self._released_at = inp.t
        self.phase = PressPhase.IDLE
        self._prev_t = self._prev_value = self._velocity = self._depth = None
        self._valid_frames = 0
        self._count = 0
        self._pressed_key = self._latched = self._idle_key = None
        self._pressed_at = None
        self._still_frames = 0
        if self.config.relearn_after_loss:
            self.baseline = None  # never reuse a reference from before the loss
            self._relearning = True

    def update(self, inp: PressInput) -> list[PressEvent]:
        cfg = self.config
        events: list[PressEvent] = []
        gap = self._prev_t is not None and inp.t - self._prev_t > cfg.max_gap_s
        if not inp.tracked or inp.value is None or inp.discontinuity or gap:
            self._reset(inp, events)
            if inp.tracked and inp.value is not None:
                self._prev_t, self._prev_value, self._valid_frames = inp.t, inp.value, 1
            return events

        value = inp.value
        frame_dt = 0.0 if self._prev_t is None else inp.t - self._prev_t
        if self._prev_value is not None and frame_dt > 0:
            self._velocity = (value - self._prev_value) / frame_dt
        self._prev_t, self._prev_value = inp.t, value
        self._valid_frames += 1
        if self.baseline is None:
            self.baseline = value
        raw_d = value - self.baseline
        # Depth used for decisions: the hand's common movement removed, never below rest.
        d = raw_d - min(inp.common_mode, max(raw_d, 0.0))
        self._depth = d
        v = self._velocity

        still = v is not None and abs(v) < cfg.baseline_max_speed and frame_dt > 0
        self._still_frames = self._still_frames + 1 if still else 0

        if self.phase is PressPhase.IDLE:
            # Baseline tracking while idle and still (posture drift is not a press).
            if still and (cfg.baseline_any_depth or raw_d < cfg.press_on):
                far = cfg.baseline_any_depth and abs(raw_d) > cfg.press_on
                tau = cfg.baseline_recover_tau_s if far else cfg.baseline_tau_s
                alpha = min(1.0, frame_dt / tau)
                self.baseline += alpha * (value - self.baseline)
            if d < cfg.release_off:
                self._blocked = False
            if self._relearning and self._still_frames >= cfg.relearn_still_frames:
                self._relearning = False
            armed = self._valid_frames > cfg.warmup_frames and v is not None and not self._relearning
            if armed and v is not None and d > cfg.press_on and v > cfg.min_velocity:
                refractory = self._released_at is not None and inp.t - self._released_at < cfg.refractory_s
                if refractory or inp.hand_speed > cfg.max_hand_speed:
                    if not self._blocked:
                        events.append(self._event("rejected", inp, inp.key, "refractory" if refractory else "hand_moving"))
                    self._blocked = True
                elif self._blocked:
                    pass  # same gesture that was already rejected: wait for the finger to come back up
                else:
                    self.phase = PressPhase.PRESS_CANDIDATE
                    self._count = 1
                    # The key under the finger before the press motion moved it.
                    self._latched = self._idle_key if self._idle_key is not None else inp.key
                    self._check_press(inp, events)
            elif v is None or v < cfg.onset_velocity:
                self._idle_key = inp.key  # finger not (yet) moving down: this is the key it rests on
            return events

        if self.phase is PressPhase.PRESS_CANDIDATE:
            if d > cfg.press_on:
                self._count += 1
                self._check_press(inp, events)
            else:
                self.phase = PressPhase.IDLE
                events.append(self._event("rejected", inp, self._latched, "too_short"))
                self._idle_key = inp.key
            return events

        if self.phase is PressPhase.PRESSED:
            if cfg.max_press_s is not None and self._pressed_at is not None and inp.t - self._pressed_at > cfg.max_press_s:
                if self._pressed_key is not None and inp.calibrated:
                    events.append(self._event("release", inp, self._pressed_key, "max_duration"))
                self.phase = PressPhase.IDLE
                self._released_at = inp.t
                self._pressed_key = self._latched = None
                self.baseline = value  # whatever it is now is the new rest
                return events
            if d < cfg.release_off:
                self.phase = PressPhase.RELEASE_CANDIDATE
                self._count = 1
                self._check_release(inp, events)
            return events

        # RELEASE_CANDIDATE
        if d < cfg.release_off:
            self._count += 1
            self._check_release(inp, events)
        else:
            self.phase = PressPhase.PRESSED
        return events

    def _check_press(self, inp: PressInput, events: list[PressEvent]) -> None:
        if self._count < self.config.min_press_frames:
            return
        self.phase = PressPhase.PRESSED
        self._pressed_at = inp.t
        self._pressed_key = self._latched
        if not inp.calibrated:
            events.append(self._event("rejected", inp, None, "uncalibrated"))
        elif self._latched is None:
            events.append(self._event("rejected", inp, None, "outside"))
        else:
            events.append(self._event("press", inp, self._latched, "ok"))

    def _check_release(self, inp: PressInput, events: list[PressEvent]) -> None:
        if self._count < self.config.min_release_frames:
            return
        if self._pressed_key is not None and inp.calibrated:
            events.append(self._event("release", inp, self._pressed_key, "ok"))
        self.phase = PressPhase.IDLE
        self._released_at = inp.t
        self._pressed_key = self._latched = None
        self._idle_key = inp.key
