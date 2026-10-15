import math

import pytest

from handpiano.instrument.keyboard import KeyboardLayout, KeyGeometry

TWO_OCTAVES = KeyboardLayout.from_octaves("C4", 2)
G = KeyGeometry()
BLACK_H = G.white_length * G.black_length_ratio


# --- geometry -----------------------------------------------------------------


def test_two_octaves_have_24_keys_14_white_10_black():
    assert len(TWO_OCTAVES.keys) == 24
    assert len(TWO_OCTAVES.white_keys) == 14
    assert len(TWO_OCTAVES.black_keys) == 10
    assert TWO_OCTAVES.width == 14.0
    assert TWO_OCTAVES.height == pytest.approx(150 / 23.5)


def test_keys_are_in_chromatic_order_with_correct_midi():
    assert [k.midi for k in TWO_OCTAVES.keys] == list(range(60, 84))
    assert [k.name for k in TWO_OCTAVES.keys][:13] == [
        "C4", "C#4", "D4", "D#4", "E4", "F4", "F#4", "G4", "G#4", "A4", "A#4", "B4", "C5",
    ]
    assert TWO_OCTAVES.keys[-1].name == "B5"
    assert TWO_OCTAVES.key_by_midi(69).frequency == pytest.approx(440.0)


def test_white_keys_tile_the_keyboard_uniformly():
    for i, key in enumerate(TWO_OCTAVES.white_keys):
        assert not key.is_black
        assert (key.x, key.y, key.width) == (float(i), 0.0, 1.0)
        assert key.height == pytest.approx(G.white_length)


def test_black_keys_sit_on_the_right_boundaries():
    # C# between C(0) and D(1) → centred on x=1; D# on 2; F# on 4; G# on 5; A# on 6; then +7 per octave.
    expected = {61: 1, 63: 2, 66: 4, 68: 5, 70: 6, 73: 8, 75: 9, 78: 11, 80: 12, 82: 13}
    for key in TWO_OCTAVES.black_keys:
        assert key.is_black
        assert key.center[0] == pytest.approx(expected[key.midi])
        assert key.width == pytest.approx(G.black_width)
        assert key.y == 0.0 and key.height == pytest.approx(BLACK_H)


def test_no_black_key_between_e_f_and_b_c():
    centers = {round(k.center[0], 6) for k in TWO_OCTAVES.black_keys}
    assert 3.0 not in centers  # E|F boundary
    assert 7.0 not in centers  # B|C boundary


def test_layout_must_start_and_end_on_white_keys():
    with pytest.raises(ValueError):
        KeyboardLayout(61, 83)
    with pytest.raises(ValueError):
        KeyboardLayout(60, 82)
    with pytest.raises(ValueError):
        KeyboardLayout.from_octaves("D4", 2)  # would end on C#6
    with pytest.raises(ValueError):
        KeyboardLayout.from_octaves("C4", 0)


def test_other_ranges():
    one = KeyboardLayout.from_octaves("F3", 1)  # F3..E4
    assert (one.keys[0].name, one.keys[-1].name, len(one.keys)) == ("F3", "E4", 12)
    assert len(one.white_keys) == 7


# --- hit testing ----------------------------------------------------------------


@pytest.mark.parametrize("key", TWO_OCTAVES.keys, ids=lambda k: k.name)
def test_center_of_every_key_hits_that_key(key):
    # Black keys only cover ±black_width/2 around a boundary, so even a white key's
    # geometric centre (at mid-key) never lies under a black key.
    assert TWO_OCTAVES.key_at(*key.center) is key


def test_black_key_wins_where_it_overlaps_white_keys():
    c_sharp = TWO_OCTAVES.key_by_midi(61)
    # Left part of C# lies over C, right part over D.
    assert TWO_OCTAVES.key_at(c_sharp.x + 0.01, 1.0).name == "C#4"
    assert TWO_OCTAVES.key_at(c_sharp.x + c_sharp.width - 0.01, 1.0).name == "C#4"
    # Same x, below the black key → the white keys.
    assert TWO_OCTAVES.key_at(c_sharp.x + 0.01, BLACK_H + 0.01).name == "C4"
    assert TWO_OCTAVES.key_at(c_sharp.x + c_sharp.width - 0.01, BLACK_H + 0.01).name == "D4"


def test_edges_are_half_open():
    c_sharp = TWO_OCTAVES.key_by_midi(61)
    # Left edge of the black key belongs to it, right edge does not.
    assert TWO_OCTAVES.key_at(c_sharp.x, 0.5).name == "C#4"
    assert TWO_OCTAVES.key_at(c_sharp.x + c_sharp.width, 0.5).name == "D4"
    # Bottom edge of the black key is not part of it.
    assert TWO_OCTAVES.key_at(1.0, BLACK_H).name == "D4"
    # Boundary between two white keys (no black key there) belongs to the right key.
    assert TWO_OCTAVES.key_at(3.0, 1.0).name == "F4"
    assert TWO_OCTAVES.key_at(math.nextafter(3.0, 0), 1.0).name == "E4"
    # Top edge of the keyboard is inside, bottom edge is outside.
    assert TWO_OCTAVES.key_at(0.5, 0.0).name == "C4"
    assert TWO_OCTAVES.key_at(0.5, TWO_OCTAVES.height) is None
    assert TWO_OCTAVES.key_at(0.5, math.nextafter(TWO_OCTAVES.height, 0)).name == "C4"
    # Left edge inside, right edge outside.
    assert TWO_OCTAVES.key_at(0.0, 5.0).name == "C4"
    assert TWO_OCTAVES.key_at(TWO_OCTAVES.width, 5.0) is None
    assert TWO_OCTAVES.key_at(math.nextafter(TWO_OCTAVES.width, 0), 5.0).name == "B5"


def test_gaps_between_black_keys_fall_on_white_keys():
    # Between D# (ends ≈2.29) and F# (starts ≈3.71) there are no black keys: E and F.
    assert TWO_OCTAVES.key_at(2.5, 0.5).name == "E4"
    assert TWO_OCTAVES.key_at(3.5, 0.5).name == "F4"


@pytest.mark.parametrize(
    ("x", "y"),
    [(-0.001, 1.0), (1.0, -0.001), (-5, -5), (14.0, 1.0), (100, 1.0), (1.0, 100), (float("nan"), 1.0), (1.0, float("inf"))],
)
def test_outside_the_keyboard_is_none(x, y):
    assert TWO_OCTAVES.key_at(x, y) is None


def test_every_point_inside_maps_to_exactly_one_key():
    """Dense scan: no dead zones, and black areas only ever return black keys."""
    steps_x, steps_y = 700, 64
    for i in range(steps_x):
        x = (i + 0.5) * TWO_OCTAVES.width / steps_x
        for j in range(steps_y):
            y = (j + 0.5) * TWO_OCTAVES.height / steps_y
            key = TWO_OCTAVES.key_at(x, y)
            assert key is not None
            assert key.contains(x, y)
            if y >= BLACK_H:
                assert not key.is_black


def test_distance_to_center():
    key = TWO_OCTAVES.key_by_midi(60)
    cx, cy = key.center
    assert key.distance_to_center(cx, cy) == 0.0
    assert key.distance_to_center(cx + 0.3, cy + 0.4) == pytest.approx(0.5)
