"""Requested camera settings and the mode the camera actually delivered."""

from __future__ import annotations

from dataclasses import dataclass

RESOLUTION_PRESETS: tuple[tuple[int, int], ...] = ((640, 480), (1280, 720), (1920, 1080))
FPS_PRESETS: tuple[int, ...] = (30, 60)


@dataclass(frozen=True, slots=True)
class CameraSettings:
    """What we ASK the camera for. The driver may deliver something else."""

    device_index: int = 0
    width: int = 1280
    height: int = 720
    fps: int = 30
    # Selfie view: the user sees themselves as in a mirror, and MediaPipe's
    # handedness labels assume a mirrored input image.
    mirror: bool = True
    # None = leave the driver default. Only applied if the backend accepts it.
    exposure: float | None = None
    autofocus: bool | None = None


@dataclass(frozen=True, slots=True)
class CameraMode:
    """What the camera actually delivered after opening."""

    device_index: int
    backend: str
    requested_width: int
    requested_height: int
    requested_fps: int
    actual_width: int
    actual_height: int
    # FPS the driver claims; None if it does not report it. The measured
    # rate (frames actually received per second) is tracked separately.
    reported_fps: float | None
    exposure_supported: bool
    autofocus_supported: bool
