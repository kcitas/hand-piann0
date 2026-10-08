"""Central configuration. Every tunable parameter of the pipeline lives here."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from handpiano.camera.camera_config import CameraSettings
from handpiano.tracking.finger_tracker import FingerTrackerConfig
from handpiano.tracking.hand_tracker import DetectorConfig

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "hand_landmarker.task"


@dataclass(frozen=True, slots=True)
class UiConfig:
    # How often the UI polls for a new tracking result. Only repaints when one exists.
    poll_interval_ms: int = 8
    # How often text metrics are refreshed (they do not need frame rate).
    stats_interval_ms: int = 250


@dataclass(frozen=True, slots=True)
class AppConfig:
    camera: CameraSettings = field(default_factory=CameraSettings)
    detector: DetectorConfig = field(default_factory=lambda: DetectorConfig(model_path=DEFAULT_MODEL_PATH))
    fingers: FingerTrackerConfig = field(default_factory=FingerTrackerConfig)
    ui: UiConfig = field(default_factory=UiConfig)
