"""Threshold-free benchmark of press signals, on any recording with known press events.

For each event (a finger that pressed, at a known time) and each candidate
signal, the EXCURSION of a finger is the maximum of the signal in the event
window minus its median in a short window just before. Three measures follow:

* AUC vs neighbours: is the pressing finger's excursion larger than that of the
  other playing fingers of the same hand at the same event? (Mann–Whitney AUC;
  0.5 = no better than chance.) This is the "neighbouring keys" problem.
* AUC vs background: is the pressing finger's excursion at its events larger
  than the same finger's excursion at times with no event (rest / no-press
  steps)? This is the false-positive-at-rest problem.
* Rank-1: fraction of events where the pressing finger has the largest
  excursion of its hand.

None of these depends on a press threshold or on a cue-matching window, so
signals and preprocessing can be compared before any detector tuning. Signal
variants:

* ``<feature>``          the stored hand-relative feature (tip_y, flex, tip_z);
* ``<feature>:cm``       excursion minus the hand's common mode at that event
                         (second-lowest excursion among the hand's fingers);
* ``<feature>:vel``      peak rate of change (per second) in the window instead of
                         the level excursion;
* ``<feature>:vel:cm``   both (temporal combination: rate of change, minus the hand's
                         common rate of change).

Any feature of ``landmark_features`` can be scored (``variants_for``). Events can
come from cues (``trace_events``: when a tap was ASKED for), from MIDI (PianoVAM) or
from independent labels (``events_from_labels``: when a contact was MEASURED).
Feature extraction, labels and the detector stay separate: this module only
compares signals around given times.

Results only describe how separable the signals are in the recordings used;
small event counts give wide uncertainty.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np

from handpiano.interaction.labels import Label
from handpiano.interaction.press_features import FEATURES
from handpiano.tracking.landmarks import PLAYING_FINGERS

VARIANTS: tuple[str, ...] = tuple(f"{f}{suffix}" for f in FEATURES for suffix in ("", ":cm", ":vel"))
MODIFIERS = frozenset({"cm", "vel"})


def variants_for(features: Iterable[str], suffixes: tuple[str, ...] = ("", ":cm", ":vel", ":vel:cm")) -> tuple[str, ...]:
    return tuple(f"{f}{suffix}" for f in features for suffix in suffixes)


def parse_variant(variant: str) -> tuple[str, frozenset[str]]:
    """'flex:vel:cm' → ('flex', {'vel', 'cm'})."""
    feature, *mods = variant.split(":")
    unknown = set(mods) - MODIFIERS
    if unknown:
        raise ValueError(f"unknown modifier(s) {sorted(unknown)} in {variant!r}")
    return feature, frozenset(mods)


def auc(positives: Iterable[float], negatives: Iterable[float]) -> float | None:
    """P(random positive > random negative) + 0.5·P(tie). None if either side is empty."""
    pos = np.asarray([p for p in positives if np.isfinite(p)], dtype=float)
    neg = np.asarray([n for n in negatives if np.isfinite(n)], dtype=float)
    if len(pos) == 0 or len(neg) == 0:
        return None
    greater = (pos[:, None] > neg[None, :]).sum()
    ties = (pos[:, None] == neg[None, :]).sum()
    return float((greater + 0.5 * ties) / (len(pos) * len(neg)))


@dataclass(frozen=True, slots=True)
class Series:
    """One finger's signal over time (NaN where untracked)."""

    t: np.ndarray
    values: dict[str, np.ndarray]  # feature → values


def excursion(series: Series, feature: str, t_event: float, pre: tuple[float, float],
              post: tuple[float, float], use_velocity: bool = False) -> float:
    """max(signal in [t+post]) − median(signal in [t+pre]); or peak rate of change in [t+post]. NaN if no data."""
    t, v = series.t, series.values[feature]
    post_mask = (t >= t_event + post[0]) & (t <= t_event + post[1])
    if use_velocity:
        idx = np.where(post_mask)[0]
        if len(idx) < 2:
            return float("nan")
        seg_t, seg_v = t[idx], v[idx]
        ok = np.isfinite(seg_v)
        if ok.sum() < 2:
            return float("nan")
        d = np.diff(seg_v[ok]) / np.maximum(np.diff(seg_t[ok]), 1e-6)
        return float(np.max(d)) if len(d) else float("nan")
    pre_mask = (t >= t_event + pre[0]) & (t < t_event + pre[1])
    before = v[pre_mask]
    during = v[post_mask]
    before = before[np.isfinite(before)]
    during = during[np.isfinite(during)]
    if len(before) < 2 or len(during) < 2:
        return float("nan")
    return float(np.max(during) - np.median(before))


@dataclass(frozen=True, slots=True)
class Event:
    finger: str  # pressing finger, e.g. "RIGHT_INDEX"
    t: float


@dataclass(frozen=True, slots=True)
class SignalScore:
    variant: str
    finger: str  # "ALL" for the pooled score
    events: int
    auc_neighbours: float | None
    auc_background: float | None
    rank1: float | None


def _variant_excursion(series: dict[str, Series], finger: str, feature: str, use_velocity: bool, t: float,
                       pre: tuple[float, float], post: tuple[float, float]) -> float:
    if finger not in series or feature not in series[finger].values:
        return float("nan")
    return excursion(series[finger], feature, t, pre, post, use_velocity=use_velocity)


def score_signals(
    series: dict[str, Series],
    events: list[Event],
    background_times: list[float],
    hand_of: Callable[[str], str],
    pre: tuple[float, float] = (-0.8, -0.3),
    post: tuple[float, float] = (-0.3, 1.0),
    variants: tuple[str, ...] = VARIANTS,
) -> list[SignalScore]:
    """Scores every variant, per pressing finger and pooled ("ALL")."""
    fingers = sorted(series)
    out: list[SignalScore] = []
    for variant in variants:
        feature, mods = parse_variant(variant)
        per_finger: dict[str, dict[str, list[float]]] = {}
        rank_hits: dict[str, list[bool]] = {}

        def hand_excursions(t: float, hand: str, feature: str = feature, vel: bool = "vel" in mods,
                            cm: bool = "cm" in mods) -> dict[str, float]:
            exc = {f: _variant_excursion(series, f, feature, vel, t, pre, post) for f in fingers if hand_of(f) == hand}
            if cm:
                finite = sorted(x for x in exc.values() if np.isfinite(x))
                common = max(0.0, finite[1]) if len(finite) >= 3 else 0.0
                exc = {f: (x - common if np.isfinite(x) else x) for f, x in exc.items()}
            return exc

        for ev in events:
            hand = hand_of(ev.finger)
            exc = hand_excursions(ev.t, hand)
            mine = exc.get(ev.finger, float("nan"))
            if not np.isfinite(mine):
                continue
            d = per_finger.setdefault(ev.finger, {"pos": [], "nb": [], "bg": []})
            d["pos"].append(mine)
            others = [x for f, x in exc.items() if f != ev.finger and np.isfinite(x)]
            d["nb"].extend(others)
            if others:
                rank_hits.setdefault(ev.finger, []).append(mine > max(others))
        for t in background_times:
            for hand in {hand_of(f) for f in per_finger}:
                exc = hand_excursions(t, hand)
                for f, x in exc.items():
                    if f in per_finger and np.isfinite(x):
                        per_finger[f]["bg"].append(x)

        pooled: dict[str, list[float]] = {"pos": [], "nb": [], "bg": []}
        pooled_rank: list[bool] = []
        for finger, d in sorted(per_finger.items()):
            r = rank_hits.get(finger, [])
            out.append(SignalScore(variant, finger, len(d["pos"]), auc(d["pos"], d["nb"]), auc(d["pos"], d["bg"]),
                                   float(np.mean(r)) if r else None))
            for k, values in pooled.items():
                values.extend(d[k])
            pooled_rank.extend(r)
        out.append(SignalScore(variant, "ALL", len(pooled["pos"]), auc(pooled["pos"], pooled["nb"]),
                               auc(pooled["pos"], pooled["bg"]), float(np.mean(pooled_rank)) if pooled_rank else None))
    return out


# -- adapters -----------------------------------------------------------------------------------


def series_from_trace(rows: list[dict[str, str]]) -> dict[str, Series]:
    """HandPiano trace rows → per-finger series of the stored features (NaN where untracked)."""
    by: dict[str, list[dict[str, str]]] = {}
    for r in rows:
        by.setdefault(r["finger"], []).append(r)
    out: dict[str, Series] = {}
    for finger, rs in by.items():
        t = np.array([float(r["t"]) for r in rs])
        values = {
            f: np.array([float(r[f]) if r.get(f) and r["tracked"] == "True" else np.nan for r in rs])
            for f in FEATURES
        }
        out[finger] = Series(t, values)
    return out


def trace_events(rows: list[dict[str, str]]) -> tuple[list[Event], list[float]]:
    """Cue events of a v2/focus trace and background times (every 0.5 s of rest/posture/slide)."""
    events: list[Event] = []
    seen: set[str] = set()
    for r in rows:
        if r.get("cue") and r["t"] not in seen:
            seen.add(r["t"])
            for name in r["cue"].split("+"):
                events.append(Event(name, float(r["t"])))
    background: list[float] = []
    last = -1e9
    for r in rows:
        if r["step"] in ("rest", "posture", "slide"):
            t = float(r["t"])
            if t - last >= 0.5:
                background.append(t)
                last = t
    return events, background


def hand_of_finger(finger: str) -> str:
    return finger.split("_")[0]


def events_from_labels(labels: Iterable[Label], start_t: float, end_t: float, step_s: float = 0.5,
                       quiet_s: float = 1.0) -> tuple[list[Event], list[float]]:
    """Independent labels → (contact events, background times).

    Events: ``contact`` labels on a playing finger with ``confirmed`` confidence.
    Background: every ``step_s`` in [start_t, end_t] with no contact of ANY finger
    within ±``quiet_s`` and outside ``ambiguous`` / ``insufficient_tracking`` intervals.
    Cues and detector output are never used.
    """
    labels = list(labels)
    fingers = {f.name for f in PLAYING_FINGERS}
    events = sorted((Event(lb.target, lb.t_start) for lb in labels
                     if lb.label == "contact" and lb.confidence == "confirmed" and lb.target in fingers),
                    key=lambda e: e.t)
    contacts = np.array([lb.t_start for lb in labels if lb.label == "contact"], dtype=float)
    excluded = [(lb.t_start, lb.t_end) for lb in labels
                if lb.label in ("ambiguous", "insufficient_tracking") and lb.t_end is not None]
    background: list[float] = []
    for t in np.arange(start_t + quiet_s, end_t - quiet_s, step_s):
        if len(contacts) and np.min(np.abs(contacts - t)) <= quiet_s:
            continue
        if any(a - quiet_s <= t <= b + quiet_s for a, b in excluded):
            continue
        background.append(float(t))
    return events, background


@dataclass(frozen=True, slots=True)
class DetectionMatch:
    finger: str
    reference: int  # independent events (e.g. key contacts)
    detected: int  # detector presses
    hits: int
    offsets: tuple[float, ...]  # detected − reference (s), per hit

    @property
    def misses(self) -> int:
        return self.reference - self.hits

    @property
    def false_presses(self) -> int:
        return self.detected - self.hits


def match_detections(reference: list[Event], detected: list[Event],
                     window: tuple[float, float] = (-0.2, 0.4)) -> list[DetectionMatch]:
    """One-to-one, per finger, in time order: each reference event takes the earliest unused
    detection with offset (detected − reference) inside ``window``. Per finger and "ALL"."""
    out: list[DetectionMatch] = []
    fingers = sorted({e.finger for e in reference} | {e.finger for e in detected})
    total = [0, 0, 0]
    all_offsets: list[float] = []
    for finger in fingers:
        ref = sorted(e.t for e in reference if e.finger == finger)
        det = sorted(e.t for e in detected if e.finger == finger)
        used = [False] * len(det)
        offsets: list[float] = []
        for r in ref:
            for i, d in enumerate(det):
                if not used[i] and window[0] <= d - r <= window[1]:
                    used[i] = True
                    offsets.append(d - r)
                    break
        out.append(DetectionMatch(finger, len(ref), len(det), len(offsets), tuple(offsets)))
        total[0] += len(ref)
        total[1] += len(det)
        total[2] += len(offsets)
        all_offsets.extend(offsets)
    out.append(DetectionMatch("ALL", total[0], total[1], total[2], tuple(all_offsets)))
    return out


def session_folds(sessions: Iterable[str]) -> list[tuple[list[str], str]]:
    """Leave-one-session-out folds: (training sessions, evaluation session).

    A learned model must be evaluated on recordings it never saw: frames of one
    session are strongly correlated, so a frame-level split leaks.
    """
    ids = list(dict.fromkeys(sessions))
    if len(ids) < 2:
        raise ValueError("need at least two sessions to separate training from evaluation")
    return [([s for s in ids if s != held], held) for held in ids]


def _fmt(v: float | None) -> str:
    return " N/A" if v is None else f"{v:.2f}"


def format_scores(scores: list[SignalScore], title: str) -> list[str]:
    """Report lines: pooled table sorted by AUC vs neighbours, then per-finger AUC vs neighbours."""
    lines = [title]
    pooled = [s for s in scores if s.finger == "ALL"]
    w = max(12, *(len(s.variant) + 1 for s in scores)) if scores else 12
    lines.append(f"  {'variant':<{w}}{'events':>7}{'AUC vs neighbours':>19}{'AUC vs background':>19}{'rank-1':>8}")
    for s in sorted(pooled, key=lambda s: -(s.auc_neighbours or 0)):
        lines.append(f"  {s.variant:<{w}}{s.events:>7}{_fmt(s.auc_neighbours):>19}"
                     f"{_fmt(s.auc_background):>19}{_fmt(s.rank1):>8}")
    fingers = sorted({s.finger for s in scores if s.finger != "ALL"})
    lines.append("\n  Per finger, AUC vs neighbours (n events):")
    lines.append("  " + f"{'variant':<{w}}" + "".join(
        f"{f.replace('LEFT_', 'L-').replace('RIGHT_', 'R-'):>10}" for f in fingers))
    for variant in dict.fromkeys(s.variant for s in scores):
        cells = []
        for f in fingers:
            cell = next((x for x in scores if x.variant == variant and x.finger == f), None)
            cells.append(f"{_fmt(cell.auc_neighbours) if cell else ' N/A'}({cell.events if cell else 0})")
        lines.append("  " + f"{variant:<{w}}" + "".join(f"{c:>10}" for c in cells))
    lines.append("")
    return lines
