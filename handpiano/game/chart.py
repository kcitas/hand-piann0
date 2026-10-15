"""Charts for the 4-lane rhythm game: JSON charts, standard MIDI files and the song library.

``SongLibrary`` fills four slots from the files in ``songs/`` (``.json`` charts or
``.mid`` files whose licence allows it; four public-domain melodies are included);
empty slots say so. ``practice_pattern`` is a scale EXERCISE, labelled as
such, so the game can be tested without any song.

JSON chart (``format: handpiano-chart``, ``version: 1``)::

    {"format": "handpiano-chart", "version": 1,
     "title": "...", "artist": "...", "source": "where it comes from", "license": "...",
     "notes": [{"t": 1.50, "lane": 0, "midi": 60, "duration": 0.40}, ...]}

``t`` seconds from the start, ``lane`` 0–3, ``midi`` the note that sounds when it
is hit, ``duration`` seconds the note sounds (optional, default 0.4).

MIDI files: note-ons of every channel except 10 (drums), tempo changes honoured.
Lanes are assigned by pitch: the distinct pitches are split into four equal
groups, low → lane 0. Notes of one lane closer than ``min_gap_s`` are dropped
(a chord becomes its top note in that lane).
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass, field
from pathlib import Path

LANES = 4
DEFAULT_SONGS_DIR = Path(__file__).resolve().parents[2] / "songs"
SLOTS = 4


class ChartError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ChartNote:
    t: float
    lane: int
    midi: int
    duration: float = 0.4


@dataclass(frozen=True, slots=True)
class Chart:
    title: str
    notes: tuple[ChartNote, ...]
    artist: str = ""
    source: str = ""
    license: str = ""
    is_exercise: bool = False

    @property
    def length_s(self) -> float:
        return max((n.t + n.duration for n in self.notes), default=0.0)


def _validated(notes: list[ChartNote]) -> tuple[ChartNote, ...]:
    for n in notes:
        if not (0 <= n.lane < LANES) or not (0 <= n.midi <= 127) or n.t < 0 or n.duration <= 0:
            raise ChartError(f"nota inválida: {n}")
    if not notes:
        raise ChartError("la partitura no tiene notas")
    return tuple(sorted(notes, key=lambda n: (n.t, n.lane)))


def load_chart_json(path: Path) -> Chart:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ChartError(f"{path.name}: no se puede leer ({exc})") from exc
    if not isinstance(data, dict) or data.get("format") != "handpiano-chart" or data.get("version") != 1:
        raise ChartError(f"{path.name}: no es una partitura handpiano-chart v1")
    try:
        notes = [ChartNote(float(n["t"]), int(n["lane"]), int(n["midi"]), float(n.get("duration", 0.4)))
                 for n in data["notes"]]
    except (KeyError, TypeError, ValueError) as exc:
        raise ChartError(f"{path.name}: nota mal formada ({exc})") from exc
    return Chart(str(data.get("title") or path.stem), _validated(notes), str(data.get("artist", "")),
                 str(data.get("source", "")), str(data.get("license", "")))


# -- standard MIDI files -----------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MidiNote:
    start: float
    end: float
    midi: int
    velocity: int
    channel: int


def _varlen(data: bytes, pos: int) -> tuple[int, int]:
    value = 0
    for _ in range(4):
        if pos >= len(data):
            raise ChartError("MIDI truncado")
        byte = data[pos]
        pos += 1
        value = (value << 7) | (byte & 0x7F)
        if not byte & 0x80:
            return value, pos
    raise ChartError("longitud variable inválida")


def parse_midi(data: bytes) -> list[MidiNote]:
    """Notes of a standard MIDI file (format 0 or 1, ticks per quarter note)."""
    if data[:4] != b"MThd" or len(data) < 14:
        raise ChartError("no es un archivo MIDI estándar")
    _, ntracks, division = struct.unpack(">HHH", data[8:14])
    if division & 0x8000:
        raise ChartError("división SMPTE no soportada")
    pos = 8 + struct.unpack(">I", data[4:8])[0]
    tempos: list[tuple[int, int]] = [(0, 500000)]  # (tick, µs per quarter)
    raw: list[tuple[int, int, int, int, bool]] = []  # (tick, channel, note, velocity, on)
    for _ in range(ntracks):
        if data[pos:pos + 4] != b"MTrk":
            raise ChartError("pista MIDI mal formada")
        length = struct.unpack(">I", data[pos + 4:pos + 8])[0]
        p, end = pos + 8, pos + 8 + length
        if end > len(data):
            raise ChartError("MIDI truncado")
        tick, status = 0, 0
        while p < end:
            delta, p = _varlen(data, p)
            tick += delta
            if data[p] & 0x80:
                status = data[p]
                p += 1
            elif status == 0:
                raise ChartError("running status sin estado previo")
            kind = status & 0xF0
            if status == 0xFF:
                meta = data[p]
                size, p = _varlen(data, p + 1)
                if meta == 0x51 and size == 3:
                    tempos.append((tick, int.from_bytes(data[p:p + 3], "big")))
                p += size
                status = 0
            elif status in (0xF0, 0xF7):
                size, p = _varlen(data, p)
                p += size
                status = 0
            elif kind in (0x80, 0x90, 0xA0, 0xB0, 0xE0):
                a, b = data[p], data[p + 1]
                p += 2
                if kind == 0x90 and b > 0:
                    raw.append((tick, status & 0x0F, a, b, True))
                elif kind == 0x80 or (kind == 0x90 and b == 0):
                    raw.append((tick, status & 0x0F, a, 0, False))
            elif kind in (0xC0, 0xD0):
                p += 1
            else:
                raise ChartError(f"evento MIDI desconocido 0x{status:02x}")
        pos = end
    tempos.sort()

    def seconds(tick: int) -> float:
        t, last_tick, last_tempo = 0.0, 0, 500000
        for at, tempo in tempos:
            if at >= tick:
                break
            t += (at - last_tick) * last_tempo / division / 1e6
            last_tick, last_tempo = at, tempo
        return t + (tick - last_tick) * last_tempo / division / 1e6

    notes: list[MidiNote] = []
    open_notes: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for tick, channel, note, velocity, on in sorted(raw, key=lambda r: (r[0], r[4])):
        if on:
            open_notes.setdefault((channel, note), []).append((tick, velocity))
        elif open_notes.get((channel, note)):
            start, vel = open_notes[(channel, note)].pop(0)
            notes.append(MidiNote(seconds(start), seconds(tick), note, vel, channel))
    return sorted(notes, key=lambda n: (n.start, n.midi))


def chart_from_midi(path: Path, min_gap_s: float = 0.15) -> Chart:
    try:
        notes = [n for n in parse_midi(path.read_bytes()) if n.channel != 9]
    except OSError as exc:
        raise ChartError(f"{path.name}: {exc}") from exc
    except (IndexError, struct.error) as exc:
        raise ChartError(f"{path.name}: MIDI mal formado") from exc
    if not notes:
        raise ChartError(f"{path.name}: sin notas")
    pitches = sorted({n.midi for n in notes})
    lane_of = {p: min(LANES - 1, i * LANES // len(pitches)) for i, p in enumerate(pitches)}
    chart: list[ChartNote] = []
    last: dict[int, ChartNote] = {}
    for n in sorted(notes, key=lambda n: (n.start, -n.midi)):
        lane = lane_of[n.midi]
        prev = last.get(lane)
        if prev is not None and n.start - prev.t < min_gap_s:
            continue
        note = ChartNote(round(n.start, 4), lane, n.midi, max(0.1, min(2.0, n.end - n.start)))
        chart.append(note)
        last[lane] = note
    return Chart(path.stem.replace("_", " "), _validated(chart), source=f"MIDI: {path.name}")


def load_chart(path: Path) -> Chart:
    if path.suffix.lower() == ".json":
        return load_chart_json(path)
    if path.suffix.lower() in (".mid", ".midi"):
        return chart_from_midi(path)
    raise ChartError(f"{path.name}: formato no soportado (usa .json o .mid)")


def practice_pattern(bpm: float = 90.0, start: int = 60) -> Chart:
    """C-major scale up and down, one note per beat, lanes cycling 0-1-2-3. An EXERCISE, not a song."""
    scale = [0, 2, 4, 5, 7, 9, 11, 12, 11, 9, 7, 5, 4, 2, 0]
    beat = 60.0 / bpm
    notes = [ChartNote(round(2.0 + i * beat, 4), i % LANES, start + s, 0.4) for i, s in enumerate(scale)]
    return Chart("Ejercicio: escala de Do (no es una canción)", tuple(notes), source="generado por HandPiano",
                 license="CC0", is_exercise=True)


@dataclass(frozen=True, slots=True)
class SongSlot:
    index: int
    path: Path | None
    chart: Chart | None = None
    error: str | None = None

    @property
    def label(self) -> str:
        if self.chart is not None:
            return self.chart.title
        if self.error is not None:
            return f"⚠ {self.path.name if self.path else '?'}"
        return "Hueco libre"


@dataclass
class SongLibrary:
    folder: Path = DEFAULT_SONGS_DIR
    slots: list[SongSlot] = field(default_factory=list)

    def scan(self) -> list[SongSlot]:
        files = sorted(p for p in self.folder.glob("*") if p.suffix.lower() in (".json", ".mid", ".midi")) \
            if self.folder.is_dir() else []
        slots = []
        for i in range(SLOTS):
            if i < len(files):
                try:
                    slots.append(SongSlot(i, files[i], load_chart(files[i])))
                except ChartError as exc:
                    slots.append(SongSlot(i, files[i], error=str(exc)))
            else:
                slots.append(SongSlot(i, None))
        self.slots = slots
        return slots
