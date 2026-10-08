"""MediaPipe Hand Landmarker wrapper (runs fully on this machine)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from handpiano.tracking.landmarks import HandObservation, observations_from_result

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class DetectorConfig:
    model_path: Path
    num_hands: int = 2
    min_hand_detection_confidence: float = 0.5
    min_hand_presence_confidence: float = 0.5
    min_tracking_confidence: float = 0.5


class ModelLoadError(Exception):
    def __init__(self, detail: str, missing: bool = False) -> None:
        super().__init__(detail)
        if missing:
            self.user_message = (
                "Falta el modelo de detección de manos. Ejecuta una vez "
                "'python scripts/download_model.py' (requiere Internet solo esa vez)."
            )
        else:
            self.user_message = (
                "No se pudo iniciar el detector de manos. Vuelve a descargar el modelo con "
                "'python scripts/download_model.py' o reinstala las dependencias."
            )


class HandTracker:
    """Detects up to ``num_hands`` hands per RGB frame.

    Uses MediaPipe's VIDEO running mode: synchronous, and it reuses the previous
    frame's hand region (tracking) instead of running the palm detector on
    every frame, which is both faster and steadier than IMAGE mode.
    """

    def __init__(self, config: DetectorConfig) -> None:
        if not config.model_path.is_file():
            raise ModelLoadError(f"model not found: {config.model_path}", missing=True)
        # Imported lazily: loading MediaPipe is slow and tests that fake the
        # detector should not pay for it.
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        self._mp = mp
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(config.model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=config.num_hands,
            min_hand_detection_confidence=config.min_hand_detection_confidence,
            min_hand_presence_confidence=config.min_hand_presence_confidence,
            min_tracking_confidence=config.min_tracking_confidence,
        )
        try:
            self._landmarker = vision.HandLandmarker.create_from_options(options)
        except Exception as exc:  # MediaPipe raises RuntimeError/ValueError variants
            raise ModelLoadError(f"landmarker init failed: {exc}") from exc
        self._last_timestamp_ms = -1

    def detect(self, rgb: np.ndarray, timestamp_ms: int) -> list[HandObservation]:
        # VIDEO mode requires strictly increasing timestamps.
        timestamp_ms = max(timestamp_ms, self._last_timestamp_ms + 1)
        self._last_timestamp_ms = timestamp_ms
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        result = self._landmarker.detect_for_video(image, timestamp_ms)
        return observations_from_result(result)

    def close(self) -> None:
        self._landmarker.close()
