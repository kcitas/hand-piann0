"""Downloads the MediaPipe hand landmarker model once. After this, HandPiano runs offline.

Usage: python scripts/download_model.py
"""

from __future__ import annotations

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
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    if TARGET.is_file():
        print(f"Model already present: {TARGET}")
    else:
        print(f"Downloading {MODEL_URL}")
        tmp = TARGET.with_suffix(".part")
        urllib.request.urlretrieve(MODEL_URL, tmp)
        tmp.replace(TARGET)
    digest = hashlib.sha256(TARGET.read_bytes()).hexdigest()
    print(f"{TARGET} ({TARGET.stat().st_size / 1e6:.1f} MB) sha256={digest}")
    if digest != EXPECTED_SHA256:
        print("WARNING: checksum differs from the version HandPiano was tested with.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
