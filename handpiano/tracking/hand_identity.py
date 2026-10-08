"""Keeps a stable LEFT/RIGHT identity for each detected hand across frames.

MediaPipe classifies handedness per frame, and that label occasionally flips
(especially with one hand, a side view or fast motion). For an instrument a
flip means every finger suddenly changes identity, so labels are combined
with temporal continuity:

* two hands with different labels → trust the labels;
* two hands with the same label → disambiguate by horizontal position
  (in the mirrored view the user's left hand is on the image's left; in the
  unmirrored view it is on the right);
* one hand that stays near where a known hand was → keep that identity unless
  the detector insists on the other label for ``label_switch_frames`` frames.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from handpiano.tracking.landmarks import Handedness, HandObservation


@dataclass(frozen=True, slots=True)
class IdentityConfig:
    # Max wrist displacement (normalized units) to consider it the same hand as last frame.
    continuity_max_distance: float = 0.15
    # How long (s) a disappeared hand's last position is remembered for continuity.
    memory_s: float = 0.3
    # Consecutive frames the detector must disagree before a tracked hand is relabeled.
    label_switch_frames: int = 6
    # Whether frames are mirrored (selfie view). Decides which side is the user's left.
    mirrored_view: bool = True


@dataclass(slots=True)
class _Memory:
    wrist: np.ndarray
    seen_at: float
    disagreements: int = 0


class HandIdentityResolver:
    def __init__(self, config: IdentityConfig | None = None) -> None:
        self.config = config or IdentityConfig()
        self._memory: dict[Handedness, _Memory] = {}

    def reset(self) -> None:
        self._memory.clear()

    def resolve(self, observations: list[HandObservation], t: float) -> dict[Handedness, HandObservation]:
        self._forget_stale(t)
        hands = sorted(observations, key=lambda o: o.score, reverse=True)[:2]

        if len(hands) == 2:
            a, b = hands
            if a.handedness is not b.handedness:
                assigned = {a.handedness: a, b.handedness: b}
            else:
                left, right = sorted(hands, key=lambda o: float(o.wrist[0]), reverse=not self.config.mirrored_view)
                assigned = {Handedness.LEFT: left, Handedness.RIGHT: right}
            for memory in self._memory.values():
                memory.disagreements = 0
        elif len(hands) == 1:
            assigned = {self._resolve_single(hands[0]): hands[0]}
        else:
            assigned = {}

        for hand, obs in assigned.items():
            previous = self._memory.get(hand)
            disagreements = previous.disagreements if previous and len(hands) == 1 else 0
            self._memory[hand] = _Memory(wrist=obs.wrist.copy(), seen_at=t, disagreements=disagreements)
        # A hand present this frame cannot also be "the other hand" in memory.
        if len(assigned) == 1:
            (only,) = assigned
            self._memory.pop(_other(only), None)
        return assigned

    def _resolve_single(self, obs: HandObservation) -> Handedness:
        label = obs.handedness
        nearest = self._nearest_memory(obs.wrist)
        if nearest is None:
            return label
        known, memory = nearest
        if known is label:
            memory.disagreements = 0
            return label
        memory.disagreements += 1
        if memory.disagreements >= self.config.label_switch_frames:
            return label
        return known

    def _nearest_memory(self, wrist: np.ndarray) -> tuple[Handedness, _Memory] | None:
        best: tuple[Handedness, _Memory] | None = None
        best_distance = self.config.continuity_max_distance
        for hand, memory in self._memory.items():
            distance = float(np.linalg.norm(memory.wrist - wrist))
            if distance <= best_distance:
                best, best_distance = (hand, memory), distance
        return best

    def _forget_stale(self, t: float) -> None:
        for hand in [h for h, m in self._memory.items() if t - m.seen_at > self.config.memory_s]:
            del self._memory[hand]


def _other(hand: Handedness) -> Handedness:
    return Handedness.RIGHT if hand is Handedness.LEFT else Handedness.LEFT
