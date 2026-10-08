"""Guided LEFT/RIGHT validation: on-screen instructions, scored from real tracking.

Vision cannot know which hand is physically the left one, so the person is
asked to raise a specific hand and the tracker's answer is compared with the
instruction. The result is only as valid as the person followed the steps; it
is a manual validation assisted by the app, not an automatic proof.

Scoring (frames during the first ``settle_s`` of each step are ignored while
the person changes posture):

* LEFT_ONLY / RIGHT_ONLY: among frames with at least one tracked hand, the
  fraction where exactly the requested hand, and only it, is tracked.
* BOTH: among frames with two tracked hands, the fraction where the left hand
  is on the side where the user's left should appear (image left in the
  mirrored view).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from enum import Enum

from handpiano.tracking.finger_tracker import TrackingState, TrackStatus
from handpiano.tracking.landmarks import Handedness, Landmark


class CheckStep(Enum):
    PREPARE = "prepare"
    LEFT_ONLY = "left_only"
    RIGHT_ONLY = "right_only"
    BOTH = "both"
    DONE = "done"


@dataclass(frozen=True, slots=True)
class StepSpec:
    step: CheckStep
    duration_s: float
    instruction: str


DEFAULT_STEPS: tuple[StepSpec, ...] = (
    StepSpec(CheckStep.PREPARE, 5.0, "Prepárate: siéntate frente a la cámara con las manos bajadas"),
    StepSpec(CheckStep.LEFT_ONLY, 10.0, "Levanta SOLO tu mano IZQUIERDA (palma a la cámara)"),
    StepSpec(CheckStep.RIGHT_ONLY, 10.0, "Baja la izquierda. Levanta SOLO tu mano DERECHA"),
    StepSpec(CheckStep.BOTH, 10.0, "Levanta AMBAS manos, separadas y sin cruzarlas"),
)

SCORED_STEPS = (CheckStep.LEFT_ONLY, CheckStep.RIGHT_ONLY, CheckStep.BOTH)


@dataclass(slots=True)
class StepResult:
    evaluated: int = 0
    correct: int = 0
    # Frames without any tracked hand (single-hand steps) or with fewer than two (BOTH).
    missing: int = 0
    # What was tracked in every scored frame, e.g. {"LEFT": 150, "LEFT+RIGHT": 30, "-": 5}.
    seen: Counter[str] = field(default_factory=Counter)

    @property
    def rate(self) -> float | None:
        return None if self.evaluated == 0 else self.correct / self.evaluated


@dataclass(frozen=True, slots=True)
class CheckConfig:
    steps: tuple[StepSpec, ...] = DEFAULT_STEPS
    settle_s: float = 1.5
    # Minimum evaluated frames for a step's result to count.
    min_frames: int = 30
    pass_rate: float = 0.9


@dataclass(slots=True)
class HandednessCheck:
    config: CheckConfig = field(default_factory=CheckConfig)
    mirrored: bool = True
    started_at: float | None = None
    results: dict[CheckStep, StepResult] = field(default_factory=lambda: {s: StepResult() for s in SCORED_STEPS})

    def start(self, t: float) -> None:
        self.started_at = t
        self.results = {s: StepResult() for s in SCORED_STEPS}

    def current(self, t: float) -> tuple[StepSpec | None, float, float]:
        """(step, seconds remaining in it, seconds elapsed in it). step is None when finished."""
        if self.started_at is None:
            return None, 0.0, 0.0
        elapsed = t - self.started_at
        for spec in self.config.steps:
            if elapsed < spec.duration_s:
                return spec, spec.duration_s - elapsed, elapsed
            elapsed -= spec.duration_s
        return None, 0.0, 0.0

    @property
    def total_s(self) -> float:
        return sum(s.duration_s for s in self.config.steps)

    def is_done(self, t: float) -> bool:
        return self.started_at is not None and self.current(t)[0] is None

    def update(self, t: float, state: TrackingState) -> None:
        spec, _, in_step = self.current(t)
        if spec is None or spec.step not in self.results or in_step < self.config.settle_s:
            return
        result = self.results[spec.step]
        tracked = {h for h, hs in state.hands.items() if hs.status is TrackStatus.TRACKED}
        result.seen["+".join(sorted(h.name for h in tracked)) or "-"] += 1
        if spec.step is CheckStep.BOTH:
            if len(tracked) < 2:
                result.missing += 1
                return
            left = state.hands[Handedness.LEFT].landmarks
            right = state.hands[Handedness.RIGHT].landmarks
            assert left is not None and right is not None
            left_is_image_left = bool(left[Landmark.WRIST, 0] < right[Landmark.WRIST, 0])
            result.evaluated += 1
            result.correct += left_is_image_left == self.mirrored
            return
        if not tracked:
            result.missing += 1
            return
        expected = Handedness.LEFT if spec.step is CheckStep.LEFT_ONLY else Handedness.RIGHT
        result.evaluated += 1
        result.correct += tracked == {expected}

    def verdict(self, step: CheckStep) -> str:
        result = self.results[step]
        if result.evaluated < self.config.min_frames:
            return "datos insuficientes"
        assert result.rate is not None
        return "OK" if result.rate >= self.config.pass_rate else "FALLA"

    def passed(self) -> bool:
        return all(self.verdict(s) == "OK" for s in SCORED_STEPS)

    def summary_lines(self) -> list[str]:
        names = {
            CheckStep.LEFT_ONLY: "Solo izquierda → LEFT",
            CheckStep.RIGHT_ONLY: "Solo derecha → RIGHT",
            CheckStep.BOTH: "Ambas, orden espacial",
        }
        lines = []
        for step in SCORED_STEPS:
            r = self.results[step]
            rate = "N/A" if r.rate is None else f"{r.rate:.0%}"
            seen = ", ".join(f"{k} {v}" for k, v in r.seen.most_common())
            lines.append(f"{names[step]}: {rate} de {r.evaluated} frames ({self.verdict(step)}) · visto: {seen or 'nada'}")
        return lines
