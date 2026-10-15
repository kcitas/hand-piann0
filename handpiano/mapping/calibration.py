"""Where the virtual keyboard is in the camera image: a 2-D, four-corner calibration.

Coordinate spaces
-----------------
1. Camera: MediaPipe's normalized landmarks, x, y ∈ [0, 1] of the frame width
   and height, in the image the app shows (mirrored by default). Origin top-left.
2. Piano: white-key widths, x ∈ [0, layout.width), y ∈ [0, layout.height); see
   ``handpiano.instrument.keyboard``.
3. UI: widget pixels; each widget converts piano or camera coordinates itself.

The user marks four corners in the camera image, in this order (as seen on screen):

    P1 (top-left)  ───── P2 (top-right)       top edge  = back of the keys (black-key end)
         │                     │
    P4 (bottom-left) ── P3 (bottom-right)     bottom edge = front of the white keys

and a homography maps camera → piano so that P1 → (0, 0), P2 → (W, 0),
P3 → (W, H), P4 → (0, H). It is 2-D: MediaPipe's z is not used, and nothing
here knows real distances.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from handpiano.mapping.homography import DegenerateHomographyError, apply_homography, homography_from_points

log = logging.getLogger(__name__)

CORNER_NAMES = ("superior izquierda", "superior derecha", "inferior derecha", "inferior izquierda")

# Minimum distance between any two corners and minimum quadrilateral area, in
# normalized camera units. Below this a few pixels of tracking noise would
# swing the whole keyboard.
MIN_CORNER_DISTANCE = 0.05
MIN_AREA = 0.01


class CalibrationError(ValueError):
    def __init__(self, user_message: str) -> None:
        super().__init__(user_message)
        self.user_message = user_message


@dataclass(frozen=True, slots=True)
class CameraKey:
    """Identifies the camera view a calibration was made for."""

    device_index: int
    width: int
    height: int
    mirror: bool


def _cross(o: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    return float((a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]))


def validate_corners(points: np.ndarray) -> None:
    """Raises CalibrationError (with a message for the user) if the corners are unusable."""
    pts = np.asarray(points, dtype=np.float64)
    if pts.shape != (4, 2):
        raise CalibrationError(f"Se necesitan exactamente 4 esquinas (hay {len(pts)}).")
    if not np.all(np.isfinite(pts)):
        raise CalibrationError("Alguna esquina no tiene una posición válida.")
    for i in range(4):
        for j in range(i + 1, 4):
            if np.linalg.norm(pts[i] - pts[j]) < MIN_CORNER_DISTANCE:
                raise CalibrationError(
                    f"Las esquinas {i + 1} y {j + 1} están demasiado cerca. Separa más las esquinas del piano."
                )
    # In image coordinates (y down), TL → TR → BR → BL turns clockwise on screen,
    # which gives a positive cross product at every corner of a convex quad.
    turns = [_cross(pts[i], pts[(i + 1) % 4], pts[(i + 2) % 4]) for i in range(4)]
    if all(t < 0 for t in turns):
        raise CalibrationError(
            "Las esquinas se marcaron en sentido contrario. Orden: superior izquierda, superior derecha, "
            "inferior derecha, inferior izquierda."
        )
    if not all(t > 0 for t in turns):
        raise CalibrationError("Las 4 esquinas no forman un cuadrilátero convexo (se cruzan o hay 3 alineadas).")
    x, y = pts[:, 0], pts[:, 1]
    area = 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))
    if area < MIN_AREA:
        raise CalibrationError("El área marcada es demasiado pequeña. Haz el piano más grande en la imagen.")


@dataclass(frozen=True, slots=True, eq=False)
class PianoCalibration:
    camera_points: tuple[tuple[float, float], ...]  # P1..P4, normalized camera coordinates
    piano_size: tuple[float, float]
    camera: CameraKey
    created_at: str
    _to_piano: np.ndarray
    _to_camera: np.ndarray

    @classmethod
    def from_corners(
        cls,
        points,
        piano_size: tuple[float, float],
        camera: CameraKey,
        created_at: str | None = None,
    ) -> PianoCalibration:
        pts = np.asarray(points, dtype=np.float64)
        validate_corners(pts)
        w, h = piano_size
        piano_corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float64)
        try:
            to_piano = homography_from_points(pts, piano_corners)
            to_camera = homography_from_points(piano_corners, pts)
        except DegenerateHomographyError as exc:
            raise CalibrationError("Las esquinas no definen una transformación válida. Vuelve a calibrar.") from exc
        to_piano.setflags(write=False)
        to_camera.setflags(write=False)
        return cls(
            camera_points=tuple((float(x), float(y)) for x, y in pts),
            piano_size=(float(w), float(h)),
            camera=camera,
            created_at=created_at or datetime.now().isoformat(timespec="seconds"),
            _to_piano=to_piano,
            _to_camera=to_camera,
        )

    def to_piano(self, camera_points) -> np.ndarray:
        """(n, 2) normalized camera points → (n, 2) piano coordinates."""
        return apply_homography(self._to_piano, camera_points)

    def to_camera(self, piano_points) -> np.ndarray:
        """(n, 2) piano coordinates → (n, 2) normalized camera points."""
        return apply_homography(self._to_camera, piano_points)

    def with_piano_size(self, piano_size: tuple[float, float]) -> PianoCalibration:
        """Same corners mapped onto a keyboard of another size (e.g. a different number of octaves)."""
        return PianoCalibration.from_corners(self.camera_points, piano_size, self.camera, self.created_at)

    # -- persistence -------------------------------------------------------------

    def to_json(self) -> str:
        return json.dumps(
            {
                "version": 1,
                "camera_points": self.camera_points,
                "camera": {
                    "device_index": self.camera.device_index,
                    "width": self.camera.width,
                    "height": self.camera.height,
                    "mirror": self.camera.mirror,
                },
                "created_at": self.created_at,
            },
            indent=2,
        )

    @classmethod
    def from_json(cls, text: str, piano_size: tuple[float, float]) -> PianoCalibration:
        data = json.loads(text)
        if data.get("version") != 1:
            raise CalibrationError("Formato de calibración desconocido.")
        cam = data["camera"]
        camera = CameraKey(int(cam["device_index"]), int(cam["width"]), int(cam["height"]), bool(cam["mirror"]))
        return cls.from_corners(data["camera_points"], piano_size, camera, data["created_at"])


def default_store_path() -> Path:
    if sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support" / "HandPiano"
    elif sys.platform == "win32":
        base = Path(os.environ.get("APPDATA", Path.home())) / "HandPiano"
    else:
        base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")) / "handpiano"
    return base / "calibration.json"


class CalibrationStore:
    """Saves one calibration on disk (corner coordinates only, never images)."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or default_store_path()

    def save(self, calibration: PianoCalibration) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(calibration.to_json(), encoding="utf-8")
        tmp.replace(self.path)

    def clear(self) -> None:
        self.path.unlink(missing_ok=True)

    def load(self, camera: CameraKey, piano_size: tuple[float, float]) -> tuple[PianoCalibration | None, str]:
        """(calibration, reason). Only returns a calibration made for exactly this camera view."""
        if not self.path.is_file():
            return None, "no hay calibración guardada"
        try:
            calibration = PianoCalibration.from_json(self.path.read_text(encoding="utf-8"), piano_size)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            log.warning("Ignoring unreadable calibration %s: %s", self.path, exc)
            return None, "la calibración guardada está dañada"
        if calibration.camera != camera:
            return None, "la calibración guardada es de otra cámara, resolución o modo espejo"
        return calibration, f"cargada (creada {calibration.created_at})"
