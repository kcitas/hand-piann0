import pytest

from handpiano.instrument.note import (
    Note,
    is_black_key,
    midi_to_frequency,
    midi_to_note_name,
    midi_to_octave,
    note_name_to_midi,
)


def test_middle_c_and_a4_standard_numbers():
    assert note_name_to_midi("C4") == 60
    assert note_name_to_midi("A4") == 69
    assert midi_to_note_name(60) == "C4"
    assert midi_to_note_name(69) == "A4"


def test_round_trip_every_midi_note():
    for midi in range(128):
        assert note_name_to_midi(midi_to_note_name(midi)) == midi


@pytest.mark.parametrize(
    ("name", "midi"),
    [("C#4", 61), ("Db4", 61), ("bb3", 58), ("Cb4", 59), ("Fb4", 64), ("B#3", 60), ("C-1", 0)],
)
def test_parses_accidentals_case_and_negative_octaves(name, midi):
    assert note_name_to_midi(name) == midi


@pytest.mark.parametrize("bad", ["H4", "C", "", "C##4", "4C"])
def test_rejects_malformed_names(bad):
    with pytest.raises(ValueError):
        note_name_to_midi(bad)


@pytest.mark.parametrize("bad", [128, -1, 60.5, True])
def test_rejects_invalid_midi(bad):
    with pytest.raises(ValueError):
        midi_to_note_name(bad)


def test_rejects_out_of_range_name():
    with pytest.raises(ValueError):
        note_name_to_midi("C#99")


def test_octave_boundary_is_at_c():
    assert midi_to_octave(59) == 3  # B3
    assert midi_to_octave(60) == 4  # C4


def test_black_keys():
    assert all(is_black_key(m) for m in (61, 63, 66, 68, 70))
    assert not any(is_black_key(m) for m in (60, 62, 64, 65, 67, 69, 71))


def test_equal_tempered_frequencies():
    assert midi_to_frequency(69) == pytest.approx(440.0)
    assert midi_to_frequency(81) == pytest.approx(880.0)
    assert midi_to_frequency(60) == pytest.approx(261.6256, abs=1e-3)


def test_note_descriptor():
    note = Note.from_name("Gb4")
    assert note.midi == 66
    assert note.name == "F#4"
    assert note.pitch_class == "F#"
    assert note.octave == 4
    assert note.is_black
    assert note.frequency == pytest.approx(369.994, abs=1e-2)
