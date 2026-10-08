"""Opens a physical camera through OpenCV and reports what it really delivers."""

from __future__ import annotations

import logging
import sys
from collections.abc import Callable
from enum import Enum
from typing import Any, Protocol

import cv2
import numpy as np

from handpiano.camera.camera_config import CameraMode, CameraSettings

log = logging.getLogger(__name__)


class VideoCaptureLike(Protocol):
    def isOpened(self) -> bool: ...  # noqa: N802 (OpenCV naming)
    def read(self) -> tuple[bool, np.ndarray | None]: ...
    def set(self, prop: int, value: float) -> bool: ...
    def get(self, prop: int) -> float: ...
    def release(self) -> None: ...


CaptureFactory = Callable[[int, int], VideoCaptureLike]


class CameraErrorKind(Enum):
    NOT_FOUND = "not_found"
    CANNOT_OPEN = "cannot_open"
    NO_FRAMES = "no_frames"


_MESSAGES = {
    CameraErrorKind.NOT_FOUND: "No se encontró ninguna cámara conectada.",
    CameraErrorKind.CANNOT_OPEN: (
        "No se pudo abrir la cámara. En macOS, revisa Ajustes del Sistema → Privacidad y seguridad → "
        "Cámara y concede permiso a la aplicación desde la que ejecutas HandPiano (Terminal, iTerm, "
        "VS Code…); después reinicia esa aplicación. También puede que otra app esté usando la cámara."
    ),
    CameraErrorKind.NO_FRAMES: (
        "La cámara se abrió pero no está entregando imágenes. Prueba con otra cámara o reconéctala."
    ),
}


class CameraError(Exception):
    def __init__(self, kind: CameraErrorKind, detail: str = "") -> None:
        super().__init__(f"{kind.value}: {detail}" if detail else kind.value)
        self.kind = kind
        self.user_message = _MESSAGES[kind]


def default_backend() -> int:
    if sys.platform == "darwin":
        return cv2.CAP_AVFOUNDATION
    if sys.platform == "win32":
        return cv2.CAP_MSMF
    return cv2.CAP_ANY


def _backend_name(backend: int) -> str:
    return {cv2.CAP_AVFOUNDATION: "AVFoundation", cv2.CAP_MSMF: "Media Foundation"}.get(backend, "auto")


def _opencv_factory(index: int, backend: int) -> VideoCaptureLike:
    return cv2.VideoCapture(index, backend)


def probe_cameras(
    max_devices: int = 4,
    factory: CaptureFactory = _opencv_factory,
    backend: int | None = None,
) -> list[int]:
    """Indices of cameras that open and deliver a frame.

    OpenCV cannot list device names portably, so devices are identified by index.
    Probing opens each device briefly (the camera LED may blink).
    """
    backend = default_backend() if backend is None else backend
    found: list[int] = []
    for index in range(max_devices):
        cap = factory(index, backend)
        try:
            if cap.isOpened():
                ok, _ = cap.read()
                if ok:
                    found.append(index)
        finally:
            cap.release()
    return found


class CameraManager:
    """Owns one opened camera. Not thread-safe: use from the capture thread only after open()."""

    WARMUP_READS = 5

    def __init__(
        self,
        settings: CameraSettings,
        factory: CaptureFactory = _opencv_factory,
        backend: int | None = None,
    ) -> None:
        self.settings = settings
        self._factory = factory
        self._backend = default_backend() if backend is None else backend
        self._cap: VideoCaptureLike | None = None
        self.mode: CameraMode | None = None

    def open(self) -> CameraMode:
        s = self.settings
        cap = self._factory(s.device_index, self._backend)
        if not cap.isOpened():
            cap.release()
            raise CameraError(CameraErrorKind.CANNOT_OPEN, f"device {s.device_index}")

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, s.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, s.height)
        cap.set(cv2.CAP_PROP_FPS, s.fps)
        # Keep the driver queue minimal so we always read a recent frame (not all backends honor it).
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        exposure_supported = self._try_set(cap, cv2.CAP_PROP_EXPOSURE, s.exposure)
        autofocus_supported = self._try_set(
            cap, cv2.CAP_PROP_AUTOFOCUS, None if s.autofocus is None else float(s.autofocus)
        )

        frame = None
        for _ in range(self.WARMUP_READS):
            ok, candidate = cap.read()
            if ok and candidate is not None:
                frame = candidate
                break
        if frame is None:
            cap.release()
            raise CameraError(CameraErrorKind.NO_FRAMES, f"device {s.device_index}")

        reported_fps = cap.get(cv2.CAP_PROP_FPS)
        self._cap = cap
        self.mode = CameraMode(
            device_index=s.device_index,
            backend=_backend_name(self._backend),
            requested_width=s.width,
            requested_height=s.height,
            requested_fps=s.fps,
            actual_width=int(frame.shape[1]),
            actual_height=int(frame.shape[0]),
            reported_fps=float(reported_fps) if reported_fps and reported_fps > 0 else None,
            exposure_supported=exposure_supported,
            autofocus_supported=autofocus_supported,
        )
        log.info("Camera opened: %s", self.mode)
        return self.mode

    @staticmethod
    def _try_set(cap: VideoCaptureLike, prop: int, value: Any) -> bool:
        """Applies a property only if requested; supported means the driver accepted it."""
        if value is None:
            return False
        return bool(cap.set(prop, value))

    def read(self) -> np.ndarray | None:
        if self._cap is None:
            raise RuntimeError("camera not opened")
        ok, frame = self._cap.read()
        return frame if ok else None

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
