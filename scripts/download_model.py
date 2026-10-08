"""Downloads the MediaPipe hand landmarker model once. After this, HandPiano runs offline.

Usage: python scripts/download_model.py [--output PATH]
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
EXPECTED_SHA256 = "fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1"
TARGET = Path(__file__).resolve().parents[1] / "models" / "hand_landmarker.task"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=TARGET, help=f"destination (default: {TARGET})")
    target: Path = parser.parse_args().output
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file():
        print(f"Model already present: {target}")
    else:
        print(f"Downloading {MODEL_URL}")
        tmp = target.with_suffix(".part")
        urllib.request.urlretrieve(MODEL_URL, tmp)
        tmp.replace(target)
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    print(f"{target} ({target.stat().st_size / 1e6:.1f} MB) sha256={digest}")
    if digest != EXPECTED_SHA256:
        print("WARNING: checksum differs from the version HandPiano was tested with.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
