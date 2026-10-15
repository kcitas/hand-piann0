"""Audio engine: instruments, mapping, polyphony, release, volume, switching, bad resources, non-blocking."""

import json
import math
import threading
import time

import numpy as np
import pytest

from handpiano.audio import soundbank
from handpiano.audio.engine import AudioEngine, NullOutput
from handpiano.audio.instruments import INSTRUMENTS, RANGE
from handpiano.audio.mixer import Mixer
from handpiano.audio.soundbank import SoundBankError, load_bank
from handpiano.audio.wavio import WavError, parse_wav, write_wav16
from handpiano.instrument.note import midi_to_frequency
from tests.audio_fakes import RATE, make_all, make_bank, peak_frequency, render


@pytest.fixture(scope="module")
def sounds(tmp_path_factory):
    return make_all(tmp_path_factory.mktemp("sounds"))


def ready_engine(sounds, instrument="grand", **kw) -> AudioEngine:
    engine = AudioEngine(sounds_dir=sounds, **kw)
    engine.set_instrument(instrument)
    assert engine.wait_ready()
    return engine


def cents(f: float, midi: int) -> float:
    return 1200 * math.log2(f / midi_to_frequency(midi))


# --- the three instruments ----------------------------------------------------------------


def test_three_real_instruments_with_distinct_sources_and_licences():
    assert set(INSTRUMENTS) == {"grand", "upright", "electric"}
    urls = [s.url for i in INSTRUMENTS.values() for s in i.samples]
    assert len(urls) == len(set(urls))  # no recording shared between instruments
    for instrument in INSTRUMENTS.values():
        assert instrument.license and instrument.attribution and instrument.source and instrument.recorded
        assert instrument.layer_max_velocity[-1] == 127
        roots = sorted({s.root for s in instrument.samples})
        # Every note of the playable range is within 3 semitones of a recorded sample.
        assert all(min(abs(m - r) for r in roots) <= 3 for m in range(RANGE[0], RANGE[1] + 1))
        for layer in range(instrument.layers):
            assert {s.root for s in instrument.samples if s.layer == layer} == set(roots)
    assert "NC" in INSTRUMENTS["electric"].license  # non-commercial: must stay visible
    assert all("@" in i.source for i in INSTRUMENTS.values())  # pinned commit in the source


@pytest.mark.parametrize("instrument", list(INSTRUMENTS))
def test_each_instrument_plays_the_right_pitch(sounds, instrument):
    engine = ready_engine(sounds, instrument)
    for midi in (60, 61, 66, 72):  # on a sample root and between roots (resampled)
        engine.note_on(midi, 100)
        y = render(engine, 0.25)
        assert abs(cents(peak_frequency(y), midi)) < 5, (instrument, midi)
        engine.all_notes_off()
        render(engine, 0.3)
    assert midi_to_frequency(69) == 440.0 and abs(midi_to_frequency(60) - 261.6256) < 1e-3


def test_region_choice_and_velocity_layers(sounds):
    bank = load_bank(sounds, "electric")
    soft, loud = bank.region(62, 30), bank.region(62, 127)
    assert soft.sample.layer == 0 and loud.sample.layer == 2 and loud.gain > soft.gain
    r = bank.region(63, 100)  # nearest root 62 (D4), shifted up one semitone
    assert r.sample.root == 62 and r.ratio == pytest.approx(2 ** (1 / 12))
    assert bank.region(10, 100).sample.root == min(bank.roots)  # far outside: nearest available


# --- polyphony, release, volume -------------------------------------------------------------


def test_chord_plays_all_notes_at_once(sounds):
    engine = ready_engine(sounds)
    for midi in (60, 64, 67):
        engine.note_on(midi, 100)
    y = render(engine, 0.25)
    assert len(engine.mixer.voices) == 3 and engine.mixer.active_notes == [60, 64, 67]
    spec = np.abs(np.fft.rfft(y.mean(axis=1) * np.hanning(len(y)), 1 << 19))
    freqs = np.fft.rfftfreq(1 << 19, 1 / RATE)
    for midi in (60, 64, 67):
        f = midi_to_frequency(midi)
        band = (freqs > f * 0.98) & (freqs < f * 1.02)
        assert spec[band].max() > 0.2 * spec.max()


def test_release_stops_the_note_and_frees_the_voice(sounds):
    engine = ready_engine(sounds, "upright")
    engine.note_on(60, 100)
    render(engine, 0.1)
    engine.note_off(60)
    release = INSTRUMENTS["upright"].release_s
    y = render(engine, release + 0.05)
    tail = y[-256:]
    assert np.sqrt(np.mean(tail ** 2)) < 1e-3  # below −60 dB
    assert engine.mixer.voices == []


def test_a_held_note_ends_with_its_sample(sounds):
    engine = ready_engine(sounds)  # synthetic samples last 0.4 s
    engine.note_on(60, 100)
    render(engine, 0.6)
    assert engine.mixer.voices == []  # never sounds forever, even without note_off


def test_restrike_releases_the_previous_voice(sounds):
    engine = ready_engine(sounds)
    engine.note_on(60, 100)
    render(engine, 0.05)
    engine.note_on(60, 100)
    render(engine, 0.01)
    voices = engine.mixer.voices
    assert len(voices) == 2 and voices[0].releasing and not voices[1].releasing
    render(engine, 0.06)
    assert len(engine.mixer.voices) == 1


def test_voice_limit_steals_the_oldest(sounds):
    engine = ready_engine(sounds, max_voices=4)
    for midi in range(60, 66):
        engine.note_on(midi, 100)
    render(engine, 0.01)
    assert len(engine.mixer.voices) == 4 and engine.mixer.stats.stolen == 2
    assert engine.mixer.active_notes == [62, 63, 64, 65]


def test_volume_scales_output_and_is_clamped(sounds):
    def rms_at(volume):
        engine = ready_engine(sounds, volume=volume)
        engine.note_on(60, 40)  # quiet: the soft limiter is linear here
        y = render(engine, 0.1)
        return float(np.sqrt(np.mean(y ** 2)))

    full, half = rms_at(0.8), rms_at(0.4)
    assert half / full == pytest.approx(0.5, rel=0.03)
    assert rms_at(0.0) == 0.0
    engine = AudioEngine(sounds_dir=sounds)
    engine.set_volume(3.0)
    assert engine.volume == 1.0
    engine.set_volume(-1)
    assert engine.volume == 0.0


def test_volume_change_is_ramped_without_clicks(sounds):
    engine = ready_engine(sounds, volume=0.8)
    engine.note_on(60, 60)
    render(engine, 0.05)
    engine.set_volume(0.1)
    y = render(engine, 0.02)
    # Largest sample-to-sample jump stays within what the sine itself does.
    assert np.max(np.abs(np.diff(y[:, 0]))) < 0.05


# --- instrument switching ---------------------------------------------------------------------


def test_switching_instrument_needs_no_restart(sounds):
    engine = ready_engine(sounds, "grand")
    engine.note_on(60, 100)
    render(engine, 0.01)
    assert engine.mixer.voices[0].data.shape[1] == 2  # grand: stereo samples
    engine.set_instrument("electric")
    assert engine.wait_ready()
    engine.note_on(64, 100)
    render(engine, 0.01)
    assert [v.data.shape[1] for v in engine.mixer.voices] == [2, 1]  # old note continues, new one is the Rhodes
    assert engine.status().instrument == "electric" and engine.status().ready
    with pytest.raises(KeyError):
        engine.set_instrument("harpsichord")


def test_preloaded_switch_is_immediate(sounds):
    engine = AudioEngine(sounds_dir=sounds)
    engine.preload()
    deadline = time.monotonic() + 10
    while not all(engine.is_loaded(i) for i in INSTRUMENTS) and time.monotonic() < deadline:
        time.sleep(0.01)
    t0 = time.perf_counter()
    engine.set_instrument("upright")
    assert (time.perf_counter() - t0) < 0.01 and engine.status().ready


# --- missing / corrupt resources ------------------------------------------------------------


def test_missing_bank_is_reported_and_the_engine_stays_silent(tmp_path):
    engine = AudioEngine(sounds_dir=tmp_path / "nothing")
    engine.set_instrument("grand")
    assert engine.wait_ready(timeout=5) is False
    status = engine.status()
    assert not status.ready and "download_sounds.py" in status.message
    engine.note_on(60, 100)  # no crash, no sound
    assert np.all(render(engine, 0.05) == 0)


def test_corrupt_or_incomplete_banks_are_rejected(tmp_path):
    folder = make_bank(tmp_path, "electric", seconds=0.3)
    victim = folder / INSTRUMENTS["electric"].samples[5].cache_name
    original = victim.read_bytes()
    victim.write_bytes(b"not a wav at all")
    with pytest.raises(SoundBankError, match="dañada"):
        load_bank(tmp_path, "electric")
    victim.write_bytes(original[: len(original) // 2])  # truncated data chunk
    with pytest.raises(SoundBankError, match="truncated"):
        load_bank(tmp_path, "electric")
    victim.unlink()
    with pytest.raises(SoundBankError):
        load_bank(tmp_path, "electric")
    victim.write_bytes(original)
    load_bank(tmp_path, "electric")  # restored: loads
    manifest = json.loads((folder / "bank.json").read_text())
    (folder / "bank.json").write_text(json.dumps({**manifest, "instrument": "grand"}))
    with pytest.raises(SoundBankError, match="otra versión"):
        load_bank(tmp_path, "electric")
    (folder / "bank.json").write_text("{broken")
    with pytest.raises(SoundBankError, match="ilegible"):
        load_bank(tmp_path, "electric")
    with pytest.raises(SoundBankError, match="desconocido"):
        load_bank(tmp_path, "theremin")


def test_wav_reader_formats_and_errors(tmp_path):
    x = (np.sin(np.linspace(0, 20, 1000)) * 0.5).astype(np.float32)
    write_wav16(tmp_path / "a.wav", x, 44100)
    wav = parse_wav((tmp_path / "a.wav").read_bytes())
    assert wav.channels == 1 and wav.bits == 16 and np.allclose(wav.samples[:, 0], x, atol=1e-4)
    with pytest.raises(FileExistsError):
        write_wav16(tmp_path / "a.wav", x, 44100)
    # 24-bit stereo built by hand.
    pcm24 = b"".join(int(v * 2**23).to_bytes(3, "little", signed=True) for v in (0.5, -0.5, 0.25, -0.25))
    import struct
    header = b"RIFF" + struct.pack("<I", 36 + len(pcm24)) + b"WAVEfmt " + struct.pack(
        "<IHHIIHH", 16, 1, 2, 44100, 44100 * 6, 6, 24) + b"data" + struct.pack("<I", len(pcm24))
    wav = parse_wav(header + pcm24)
    assert wav.bits == 24 and wav.samples.tolist() == [[0.5, -0.5], [0.25, -0.25]]
    for bad, message in ((b"RIFX", "RIFF"), (header[:30], "fmt"), (header[:36], "no data"),
                         (header + pcm24[:5], "truncated")):
        with pytest.raises(WavError, match=message):
            parse_wav(bad)
    assert parse_wav(header + pcm24[:6], allow_truncated=True).truncated


# --- never blocking the UI or the tracking thread -----------------------------------------------


def test_calls_return_immediately_while_a_bank_loads_and_audio_renders(sounds, monkeypatch):
    real = soundbank.load_bank

    def slow_load(*args):
        time.sleep(0.5)
        return real(*args)

    monkeypatch.setattr("handpiano.audio.engine.load_bank", slow_load)
    engine = AudioEngine(sounds_dir=sounds)
    t0 = time.perf_counter()
    engine.set_instrument("grand")  # loading takes 0.5 s in the background
    assert time.perf_counter() - t0 < 0.02
    assert not engine.status().ready
    stop = threading.Event()

    def audio_thread():
        while not stop.is_set():
            engine.mixer.render(256)

    worker = threading.Thread(target=audio_thread)
    worker.start()
    worst = 0.0
    for i in range(500):
        t0 = time.perf_counter()
        engine.note_on(60 + i % 12, 100)
        engine.note_off(60 + i % 12)
        worst = max(worst, time.perf_counter() - t0)
    stop.set()
    worker.join()
    assert worst < 0.005  # enqueue only: no lock shared with the audio thread
    assert engine.wait_ready()


def test_output_failure_keeps_the_app_running(sounds):
    class Broken:
        description = "broken"

        def start(self, render):
            raise RuntimeError("no device")

        def close(self):
            pass

    engine = AudioEngine(sounds_dir=sounds, output=Broken())
    engine.start()
    assert isinstance(engine.output, NullOutput) and "no device" in engine.status().output


def test_mixer_without_bank_renders_silence():
    mixer = Mixer()
    mixer.post("on", 60, 100)
    assert np.all(mixer.render(128) == 0) and mixer.voices == []
