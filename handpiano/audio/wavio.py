"""Minimal WAV reading/writing with numpy (PCM 16/24/32-bit integer, 32-bit float).

The standard library's ``wave`` module does not read 24-bit or float data into
arrays, and adding libsndfile for that is not justified. Malformed files raise
``WavError`` with the reason, never return garbage.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PCM = 1
FLOAT = 3
EXTENSIBLE = 0xFFFE


class WavError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class WavData:
    samples: np.ndarray  # float32, shape (frames, channels), in [-1, 1]
    sample_rate: int
    bits: int
    truncated: bool  # the data chunk was shorter than its header says (e.g. a partial download)

    @property
    def channels(self) -> int:
        return int(self.samples.shape[1])

    @property
    def duration_s(self) -> float:
        return self.samples.shape[0] / self.sample_rate


def parse_wav(data: bytes, allow_truncated: bool = False) -> WavData:
    if len(data) < 12 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise WavError("not a RIFF/WAVE file")
    fmt: tuple[int, int, int, int] | None = None
    pos = 12
    while pos + 8 <= len(data):
        cid = data[pos:pos + 4]
        size = struct.unpack("<I", data[pos + 4:pos + 8])[0]
        body = pos + 8
        if cid == b"fmt ":
            if size < 16 or body + 16 > len(data):
                raise WavError("fmt chunk too short")
            tag, channels, rate, _, _, bits = struct.unpack("<HHIIHH", data[body:body + 16])
            if tag == EXTENSIBLE and size >= 26:
                tag = struct.unpack("<H", data[body + 24:body + 26])[0]
            fmt = (tag, channels, rate, bits)
        elif cid == b"data":
            if fmt is None:
                raise WavError("data chunk before fmt chunk")
            available = data[body:body + size]
            truncated = len(available) < size
            if truncated and not allow_truncated:
                raise WavError(f"data chunk truncated ({len(available)} of {size} bytes)")
            return _decode(available, *fmt, truncated=truncated)
        pos = body + size + (size & 1)
    raise WavError("no data chunk")


def _decode(raw: bytes, tag: int, channels: int, rate: int, bits: int, truncated: bool) -> WavData:
    if channels < 1 or channels > 8 or rate < 8000 or rate > 192000:
        raise WavError(f"unsupported layout: {channels} channels at {rate} Hz")
    width = bits // 8
    if (tag, bits) not in ((PCM, 16), (PCM, 24), (PCM, 32), (FLOAT, 32)):
        raise WavError(f"unsupported sample format (tag {tag}, {bits} bits)")
    usable = len(raw) // (width * channels) * width * channels
    if usable == 0:
        raise WavError("no audio frames")
    buf = raw[:usable]
    if tag == FLOAT:
        x = np.frombuffer(buf, "<f4").astype(np.float32)
    elif bits == 16:
        x = np.frombuffer(buf, "<i2").astype(np.float32) / 32768.0
    elif bits == 32:
        x = (np.frombuffer(buf, "<i4").astype(np.float64) / 2147483648.0).astype(np.float32)
    else:
        b = np.frombuffer(buf, np.uint8).reshape(-1, 3).astype(np.int32)
        v = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
        x = (np.where(v >= 1 << 23, v - (1 << 24), v) / float(1 << 23)).astype(np.float32)
    if not np.all(np.isfinite(x)):
        raise WavError("non-finite samples")
    return WavData(x.reshape(-1, channels), rate, bits, truncated)


def read_wav(path: Path, allow_truncated: bool = False) -> WavData:
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise WavError(f"{path}: {exc.strerror or exc}") from exc
    try:
        return parse_wav(data, allow_truncated)
    except WavError as exc:
        raise WavError(f"{path.name}: {exc}") from exc


def write_wav16(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    """16-bit PCM; ``samples`` float in [-1, 1], shape (frames,) or (frames, channels). Refuses to overwrite."""
    x = np.asarray(samples, dtype=np.float32)
    if x.ndim == 1:
        x = x[:, None]
    pcm = (np.clip(x, -1.0, 1.0) * 32767.0).round().astype("<i2")
    channels = pcm.shape[1]
    body = pcm.tobytes()
    header = b"RIFF" + struct.pack("<I", 36 + len(body)) + b"WAVE" + b"fmt " + struct.pack(
        "<IHHIIHH", 16, PCM, channels, sample_rate, sample_rate * channels * 2, channels * 2, 16
    ) + b"data" + struct.pack("<I", len(body))
    with path.open("xb") as f:
        f.write(header + body)
