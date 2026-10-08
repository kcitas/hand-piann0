"""Note model and conversions.

MIDI numbers are the canonical identity of a note across the app (keyboard
geometry, audio engine, training). Names are only for display and parsing.
Convention: C4 = MIDI 60 (middle C), A4 = MIDI 69 = 440 Hz.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

PITCH_CLASSES: tuple[str, ...] = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")

_BLACK_PITCH_CLASSES = frozenset({"C#", "D#", "F#", "G#", "A#"})
_NATURAL_SEMITONES = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
_ACCIDENTAL_SEMITONES = {"": 0, "#": 1, "b": -1}
_NOTE_NAME_PATTERN = re.compile(r"^([A-Ga-g])([#b]?)(-?\d+)$")

MIDI_MIN = 0
MIDI_MAX = 127
_A4_MIDI = 69
_A4_FREQUENCY = 440.0


def _assert_valid_midi(midi: int) -> None:
    if not isinstance(midi, int) or isinstance(midi, bool) or not MIDI_MIN <= midi <= MIDI_MAX:
        raise ValueError(f"Invalid MIDI note number: {midi!r}")


def midi_to_pitch_class(midi: int) -> str:
    _assert_valid_midi(midi)
    return PITCH_CLASSES[midi % 12]


def midi_to_octave(midi: int) -> int:
    _assert_valid_midi(midi)
    return midi // 12 - 1


def midi_to_note_name(midi: int) -> str:
    return f"{midi_to_pitch_class(midi)}{midi_to_octave(midi)}"


def is_black_key(midi: int) -> bool:
    return midi_to_pitch_class(midi) in _BLACK_PITCH_CLASSES


def midi_to_frequency(midi: int) -> float:
    _assert_valid_midi(midi)
    return _A4_FREQUENCY * 2 ** ((midi - _A4_MIDI) / 12)


def note_name_to_midi(name: str) -> int:
    """Parse "C4", "c#3", "Bb2", "A-1".

    The octave belongs to the written letter (scientific pitch notation),
    so "Cb4" is B3 (59) and "B#3" is C4 (60).
    """
    match = _NOTE_NAME_PATTERN.match(name.strip())
    if match is None:
        raise ValueError(f"Invalid note name: {name!r}")
    letter, accidental, octave = match.groups()
    semitone = _NATURAL_SEMITONES[letter.upper()] + _ACCIDENTAL_SEMITONES[accidental]
    midi = (int(octave) + 1) * 12 + semitone
    _assert_valid_midi(midi)
    return midi


@dataclass(frozen=True, slots=True)
class Note:
    midi: int
    pitch_class: str
    octave: int
    name: str
    is_black: bool
    frequency: float

    @classmethod
    def from_midi(cls, midi: int) -> Note:
        return cls(
            midi=midi,
            pitch_class=midi_to_pitch_class(midi),
            octave=midi_to_octave(midi),
            name=midi_to_note_name(midi),
            is_black=is_black_key(midi),
            frequency=midi_to_frequency(midi),
        )

    @classmethod
    def from_name(cls, name: str) -> Note:
        return cls.from_midi(note_name_to_midi(name))
