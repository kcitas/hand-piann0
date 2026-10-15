"""Audio diagnostics on this machine: plays a short arpeggio with each instrument and reports.

Usage: python scripts/audio_check.py [--volume 0.4] [--block 256]

Reported (what software can measure):
* note_on call time (must be microseconds: callers never wait);
* command delay: note_on call → the audio thread applies it (≤ one block);
* PortAudio's own output-latency estimate and the last outputBufferDacTime lead;
* callback time and underflows (audible glitches).

NOT reported: the acoustic latency (finger → sound reaching the ear), which
depends on the speakers/Bluetooth and must be measured with a microphone; see
docs/AUDIO.md, "Medir la latencia real".
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from handpiano.audio.engine import AudioEngine, SoundDeviceOutput  # noqa: E402
from handpiano.audio.instruments import INSTRUMENTS  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--volume", type=float, default=0.4)
    parser.add_argument("--block", type=int, default=256)
    args = parser.parse_args()
    output = SoundDeviceOutput(block=args.block)
    engine = AudioEngine(output=output, volume=args.volume)
    engine.start()
    print(f"Salida: {engine.status().output}")
    ok = True
    for instrument_id, instrument in INSTRUMENTS.items():
        engine.set_instrument(instrument_id)
        if not engine.wait_ready():
            print(f"{instrument.name}: NO disponible — {engine.status().message}")
            ok = False
            continue
        calls, delays = [], []
        for midi in (60, 64, 67, 72, 67, 64, 60):
            t0 = time.perf_counter()
            engine.note_on(midi, 90)
            calls.append((time.perf_counter() - t0) * 1e6)
            time.sleep(0.18)
            delays.append(engine.mixer.stats.last_command_delay_ms)
            engine.note_off(midi)
        time.sleep(0.4)
        print(f"{instrument.name:15s} note_on {statistics.median(calls):5.1f} µs (mediana) · "
              f"orden → hilo de audio {statistics.median(delays):4.1f} ms (mediana), "
              f"{max(delays):4.1f} ms (máx)")
    s = output.stats
    lead = "N/A" if s.dac_lead_s is None else f"{s.dac_lead_s * 1000:.1f} ms"
    reported = "N/A" if s.reported_latency_s is None else f"{s.reported_latency_s * 1000:.1f} ms"
    print(f"PortAudio: latencia de salida estimada {reported}; buffer → DAC (último bloque) {lead}")
    print(f"Bloques {s.callbacks}, cortes (underflow) {s.underflows}, callback más lento {s.max_callback_ms:.2f} ms "
          f"(presupuesto {args.block / 44100 * 1000:.1f} ms)")
    print("La latencia ACÚSTICA (dedo → sonido) no se mide aquí: ver docs/AUDIO.md.")
    engine.close()
    return 0 if ok and s.underflows == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
