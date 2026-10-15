"""Rhythm game 4K: charts, MIDI, song slots, judging, combo, accuracy; scoring independent of audio."""

import json
import struct

import pytest

from handpiano.game.chart import (
    Chart,
    ChartError,
    ChartNote,
    SongLibrary,
    chart_from_midi,
    load_chart_json,
    parse_midi,
    practice_pattern,
)
from handpiano.game.rhythm import RhythmGame


def chart(*notes) -> Chart:
    return Chart("test", tuple(ChartNote(t, lane, 60 + lane) for t, lane in notes))


def started(*notes, now=100.0):
    game = RhythmGame()
    game.start(chart(*notes), now, lead_in=0.0)
    return game


# --- judging -----------------------------------------------------------------------------------


@pytest.mark.parametrize("offset, grade", [(0.0, "PERFECT"), (0.045, "PERFECT"), (-0.08, "GREAT"),
                                           (0.15, "GOOD"), (-0.159, "GOOD")])
def test_windows(offset, grade):
    game = started((1.0, 0))
    j = game.press(0, 101.0 + offset)
    assert j is not None and j.grade == grade and j.offset_s == pytest.approx(offset)


def test_press_outside_any_window_or_wrong_lane_hits_nothing():
    game = started((1.0, 0))
    assert game.press(0, 100.5) is None  # far too early: ignored, note still pending
    assert game.press(1, 101.0) is None  # wrong lane
    assert game.press(0, 101.0).grade == "PERFECT"


def test_missed_notes_reset_the_combo_and_count_in_accuracy():
    game = started((1.0, 0), (2.0, 1), (3.0, 2))
    game.press(0, 101.0)  # PERFECT
    game.press(1, 102.08)  # GREAT
    assert game.combo == 2
    game.update(103.2)  # lane 2 note passed without a press
    assert game.counts == {"PERFECT": 1, "GREAT": 1, "GOOD": 0, "MISS": 1}
    assert game.combo == 0 and game.max_combo == 2
    assert game.accuracy == pytest.approx((1.0 + 0.7 + 0.0) / 3)


def test_one_press_hits_at_most_one_note():
    game = started((1.0, 0), (1.05, 0))  # two notes close together in the same lane
    assert game.press(0, 101.02) is not None
    assert len(game.judgments) == 1
    assert game.press(0, 101.06) is not None  # a SECOND press is needed for the second note
    assert len(game.judgments) == 2


def test_a_held_key_does_not_hit_later_notes():
    """Holding produces no new press events: notes that pass under a held finger are missed."""
    game = started((1.0, 0), (1.5, 0))
    game.press(0, 101.0)  # the only press; the finger stays down
    for now in (101.2, 101.5, 101.8):
        game.update(now)
    assert [j.grade for j in game.judgments] == ["PERFECT", "MISS"]


def test_judgments_notify_listeners_once_and_scoring_needs_no_audio():
    heard = []
    game = RhythmGame(on_judgment=[heard.append])
    game.start(chart((1.0, 0), (2.0, 3)), 100.0, lead_in=0.0)
    game.press(0, 101.0)
    game.press(3, 102.0)
    game.update(110.0)
    assert [j.grade for j in heard] == ["PERFECT", "PERFECT"] and game.score > 0
    assert game.finished(110.0) and not game.finished(101.5)


def test_upcoming_notes_for_drawing():
    game = started((1.0, 0), (5.0, 1))
    assert [n.t for n in game.upcoming(100.0, 1.6)] == [1.0]


# --- charts and songs -------------------------------------------------------------------------


def test_json_chart_round_trip_and_validation(tmp_path):
    path = tmp_path / "song.json"
    path.write_text(json.dumps({"format": "handpiano-chart", "version": 1, "title": "Mi canción",
                                "source": "propia", "license": "CC0",
                                "notes": [{"t": 2.0, "lane": 3, "midi": 72}, {"t": 1.0, "lane": 0, "midi": 60,
                                                                               "duration": 0.5}]}))
    c = load_chart_json(path)
    assert c.title == "Mi canción" and [n.t for n in c.notes] == [1.0, 2.0] and c.notes[1].duration == 0.4
    for bad in ({"format": "x"}, {"format": "handpiano-chart", "version": 1, "notes": []},
                {"format": "handpiano-chart", "version": 1, "notes": [{"t": 1, "lane": 4, "midi": 60}]},
                {"format": "handpiano-chart", "version": 1, "notes": [{"lane": 0}]}):
        path.write_text(json.dumps(bad))
        with pytest.raises(ChartError):
            load_chart_json(path)


def _midi(tracks: list[bytes], division: int = 480) -> bytes:
    out = b"MThd" + struct.pack(">IHHH", 6, 1, len(tracks), division)
    for t in tracks:
        out += b"MTrk" + struct.pack(">I", len(t)) + t
    return out


def test_midi_notes_tempo_and_lanes(tmp_path):
    tempo = b"\x00\xff\x51\x03" + (1_000_000).to_bytes(3, "big") + b"\x00\xff\x2f\x00"  # 60 BPM
    # Quarter = 480 ticks = 1 s. Notes C4, E4, G4, C5 one beat apart (running status for the
    # note-offs), plus a drum note on channel 10 that must be ignored.
    notes = (b"\x00\x90\x3c\x64" b"\x83\x60\x3c\x00" b"\x00\x40\x64" b"\x83\x60\x40\x00"
             b"\x00\x43\x64" b"\x83\x60\x43\x00" b"\x00\x48\x64" b"\x83\x60\x48\x00"
             b"\x00\x99\x24\x64" b"\x10\x89\x24\x00" b"\x00\xff\x2f\x00")
    data = _midi([tempo, notes])
    parsed = parse_midi(data)
    melodic = [n for n in parsed if n.channel != 9]
    assert [(round(n.start, 3), n.midi) for n in melodic] == [(0.0, 60), (1.0, 64), (2.0, 67), (3.0, 72)]
    path = tmp_path / "scale.mid"
    path.write_bytes(data)
    c = chart_from_midi(path)
    assert [n.lane for n in c.notes] == [0, 1, 2, 3] and [n.midi for n in c.notes] == [60, 64, 67, 72]
    assert all(n.duration == pytest.approx(1.0) for n in c.notes)
    path.write_bytes(data[:40])
    with pytest.raises(ChartError):
        chart_from_midi(path)
    path.write_bytes(b"garbage")
    with pytest.raises(ChartError):
        chart_from_midi(path)


def test_song_library_has_four_slots_and_reports_bad_files(tmp_path):
    assert [s.label for s in SongLibrary(tmp_path / "missing").scan()] == ["Hueco libre"] * 4
    (tmp_path / "a.json").write_text(json.dumps({"format": "handpiano-chart", "version": 1, "title": "A",
                                                 "notes": [{"t": 1, "lane": 0, "midi": 60}]}))
    (tmp_path / "b.mid").write_bytes(b"garbage")
    (tmp_path / "notes.txt").write_text("ignored")
    slots = SongLibrary(tmp_path).scan()
    assert slots[0].chart is not None and slots[0].label == "A"
    assert slots[1].chart is None and slots[1].error and slots[1].label.startswith("⚠")
    assert slots[2].path is None and slots[3].path is None


def test_practice_pattern_is_labelled_as_an_exercise():
    c = practice_pattern()
    assert c.is_exercise and "no es una canción" in c.title
    assert {n.lane for n in c.notes} == {0, 1, 2, 3}
