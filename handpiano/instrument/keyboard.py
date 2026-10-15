"""Piano keyboard geometry and hit testing. Pure math: no Qt, no camera.

Piano coordinate space
----------------------
Units are WHITE-KEY WIDTHS. The origin is the back-left corner of the
keyboard (the end where black keys are); x grows towards higher notes, y
grows towards the player (front edge of the white keys).

    x ∈ [0, width)    width  = number of white keys
    y ∈ [0, height)   height = white-key length (default 150/23.5 ≈ 6.38, a real
                               piano's proportion; these are proportions, not centimetres)

Key areas are half-open rectangles ``[x, x + w) × [y, y + h)``:

* the left and top edges belong to the key, the right and bottom edges do not;
* a point exactly on the boundary between two white keys belongs to the right one;
* ``x == width`` or ``y == height`` is outside the keyboard;
* white keys tile the keyboard without gaps: the thin gaps drawn between keys
  are purely visual, there are no dead zones;
* black keys are tested first, so where a black key overlaps two white keys
  the black key wins.

Black keys are centred on the boundary between their two neighbouring white
keys. Real pianos shift some black keys slightly off-centre; that refinement is
deliberately not modelled.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from handpiano.instrument.note import Note, is_black_key, note_name_to_midi


@dataclass(frozen=True, slots=True)
class KeyGeometry:
    """Proportions in white-key widths. Defaults follow a standard acoustic piano."""

    white_length: float = 150.0 / 23.5
    black_width: float = 13.7 / 23.5
    # Black-key length as a fraction of the white-key length.
    black_length_ratio: float = 95.0 / 150.0


@dataclass(frozen=True, slots=True)
class PianoKey:
    note: Note
    x: float
    y: float
    width: float
    height: float

    @property
    def midi(self) -> int:
        return self.note.midi

    @property
    def name(self) -> str:
        return self.note.name

    @property
    def is_black(self) -> bool:
        return self.note.is_black

    @property
    def frequency(self) -> float:
        return self.note.frequency

    @property
    def center(self) -> tuple[float, float]:
        """Geometric centre of the key's full rectangle."""
        return (self.x + self.width / 2, self.y + self.height / 2)

    def contains(self, x: float, y: float) -> bool:
        return self.x <= x < self.x + self.width and self.y <= y < self.y + self.height

    def distance_to_center(self, x: float, y: float) -> float:
        cx, cy = self.center
        return math.hypot(x - cx, y - cy)


class KeyboardLayout:
    def __init__(self, low_midi: int, high_midi: int, geometry: KeyGeometry | None = None) -> None:
        if low_midi > high_midi:
            raise ValueError("low_midi must be <= high_midi")
        if is_black_key(low_midi) or is_black_key(high_midi):
            # A black key at an edge would only have half its neighbours.
            raise ValueError("the keyboard must start and end on white keys")
        self.geometry = geometry or KeyGeometry()
        g = self.geometry

        whites: list[PianoKey] = []
        white_index: dict[int, int] = {}
        for midi in range(low_midi, high_midi + 1):
            if not is_black_key(midi):
                white_index[midi] = len(whites)
                whites.append(PianoKey(Note.from_midi(midi), float(len(whites)), 0.0, 1.0, g.white_length))

        blacks: list[PianoKey] = []
        black_height = g.white_length * g.black_length_ratio
        for midi in range(low_midi, high_midi + 1):
            if is_black_key(midi):
                boundary = white_index[midi - 1] + 1.0  # between the white below and the white above
                blacks.append(PianoKey(Note.from_midi(midi), boundary - g.black_width / 2, 0.0, g.black_width, black_height))

        self.low_midi = low_midi
        self.high_midi = high_midi
        self.white_keys: tuple[PianoKey, ...] = tuple(whites)
        self.black_keys: tuple[PianoKey, ...] = tuple(blacks)
        self.keys: tuple[PianoKey, ...] = tuple(sorted(whites + blacks, key=lambda k: k.midi))
        self._by_midi = {k.midi: k for k in self.keys}
        self.width = float(len(whites))
        self.height = g.white_length

    @classmethod
    def from_octaves(cls, start: str = "C4", octaves: int = 2, geometry: KeyGeometry | None = None) -> KeyboardLayout:
        """``octaves`` × 12 keys starting at ``start`` (e.g. C4..B5 for two octaves from C4)."""
        if octaves < 1:
            raise ValueError("octaves must be >= 1")
        low = note_name_to_midi(start)
        return cls(low, low + 12 * octaves - 1, geometry)

    @property
    def size(self) -> tuple[float, float]:
        return (self.width, self.height)

    def key_by_midi(self, midi: int) -> PianoKey | None:
        return self._by_midi.get(midi)

    def contains(self, x: float, y: float) -> bool:
        return 0.0 <= x < self.width and 0.0 <= y < self.height

    def key_at(self, x: float, y: float) -> PianoKey | None:
        """The key under (x, y) in piano coordinates, or None outside the keyboard (or NaN)."""
        if not self.contains(x, y):  # also rejects NaN: every comparison with NaN is False
            return None
        for key in self.black_keys:  # black keys have priority over the whites they overlap
            if key.contains(x, y):
                return key
        return self.white_keys[min(int(x), len(self.white_keys) - 1)]

    def __repr__(self) -> str:
        return f"KeyboardLayout({self.keys[0].name}–{self.keys[-1].name}, {len(self.keys)} keys)"
