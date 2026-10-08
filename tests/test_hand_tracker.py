"""Runs the REAL MediaPipe model on controlled images (skipped if the model is not downloaded).

There is no hand photo in the repo, so this verifies loading, the VIDEO-mode
timestamp contract and that empty scenes yield no hands. Detection of actual
hands is verified manually with the camera (see README).
"""

import numpy as np
import pytest

from handpiano.app.config import DEFAULT_MODEL_PATH
from handpiano.tracking.hand_tracker import DetectorConfig, HandTracker, ModelLoadError

needs_model = pytest.mark.skipif(not DEFAULT_MODEL_PATH.is_file(), reason="model not downloaded")


def test_missing_model_raises_user_facing_error(tmp_path):
    with pytest.raises(ModelLoadError) as err:
        HandTracker(DetectorConfig(model_path=tmp_path / "nope.task"))
    assert "download_model.py" in err.value.user_message


def test_corrupt_model_raises_user_facing_error(tmp_path):
    bad = tmp_path / "bad.task"
    bad.write_bytes(b"not a model")
    with pytest.raises(ModelLoadError) as err:
        HandTracker(DetectorConfig(model_path=bad))
    assert "detector" in err.value.user_message


@needs_model
def test_real_model_finds_no_hands_in_empty_scenes():
    tracker = HandTracker(DetectorConfig(model_path=DEFAULT_MODEL_PATH))
    try:
        black = np.zeros((480, 640, 3), np.uint8)
        noise = np.random.default_rng(0).integers(0, 255, (480, 640, 3), dtype=np.uint8)
        assert tracker.detect(black, 0) == []
        assert tracker.detect(noise, 33) == []
        # Non-increasing timestamps are corrected instead of crashing MediaPipe.
        assert tracker.detect(black, 33) == []
        assert tracker.detect(black, 10) == []
    finally:
        tracker.close()
