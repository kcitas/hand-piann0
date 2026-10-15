"""Central configuration. Every tunable parameter of the pipeline lives here."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from handpiano.camera.camera_config import CameraSettings
from handpiano.interaction.press_detector import PressConfig
from handpiano.mapping.calibration_session import SessionConfig
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
class PianoConfig:
    # First key and number of octaves: C4 + 2 octaves = C4..B5 (24 keys).
    start: str = "C4"
    octaves: int = 2
    # Where the calibration (four corner coordinates, no images) is saved. None = per-user default.
    calibration_path: Path | None = None
    calibration: SessionConfig = field(default_factory=SessionConfig)
    # Screen piano: the keyboard is drawn at a FIXED place of the camera image (normalized
    # left, top, right, bottom; the top edge is the black-key side) instead of the 4 corners
    # calibrated on the table. Nothing is saved and the stored calibration is never touched.
    # The app enables it by default (main.py); --piano-mesa uses the stored calibration.
    screen: bool = False
    # Measured on tocar_1 (2026-10-13): with the keyboard at y 0.42–0.76 the fingertips sat at y ≈ 0.65 and the
    # wrists ≈ 0.33 lower, so the right wrist left the image (y up to 1.26) and MediaPipe lost that hand 51 % of
    # the time. Placed higher, the wrists stay inside the image (≈ 0.85).
    screen_area: tuple[float, float, float, float] = (0.15, 0.30, 0.85, 0.62)


@dataclass(frozen=True, slots=True)
class AppConfig:
    camera: CameraSettings = field(default_factory=CameraSettings)
    detector: DetectorConfig = field(default_factory=lambda: DetectorConfig(model_path=DEFAULT_MODEL_PATH))
    fingers: FingerTrackerConfig = field(default_factory=FingerTrackerConfig)
    ui: UiConfig = field(default_factory=UiConfig)
    piano: PianoConfig = field(default_factory=PianoConfig)
    presses: PressConfig = field(default_factory=PressConfig)
