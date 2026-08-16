"""Mapping image pixels onto the projector, and the calibration transform.

Two jobs, kept separate on purpose:

* :func:`fit_transform` works out how to place a camera image inside the DAC's
  square coordinate space without distorting it.
* :class:`Calibration` is the part a human tweaks at the projector -- flips and
  rotations, because a projector mounted upside-down behind a mirror is normal.

Both are pure integer/float maths on numpy arrays; nothing here touches a camera
or a DAC, which is what makes it straightforward to test.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .frame import COORD_CENTER, COORD_MAX, COORD_MIN


@dataclass(slots=True)
class Calibration:
    """Orientation fixes applied after the image is fitted into DAC space.

    Order is fixed and deliberate: rotate, then flip X, then flip Y. Changing the
    order changes the result, so pinning it means a saved config always
    reproduces what was seen at the projector.
    """

    rotate: int = 0          #: 0, 90, 180 or 270 degrees, counter-clockwise
    flip_x: bool = False
    flip_y: bool = False
    scale_pct: float = 70.0  #: percentage of full projector scale to fill
    offset_x: int = 0
    offset_y: int = 0

    def normalised_rotate(self) -> int:
        return int(self.rotate) % 360 // 90 * 90


def fit_transform(width: int, height: int, scale_pct: float = 70.0) -> tuple[float, float, float]:
    """Return ``(scale, cx, cy)`` mapping an image of this size into DAC space.

    A single scale factor is used for both axes -- taken from the *longer* image
    edge -- so a 16:9 camera frame stays 16:9 on the wall instead of being
    stretched to the projector's square field.
    """
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    scale = (scale_pct / 100.0) * (COORD_MAX - COORD_MIN) / float(max(width, height))
    return scale, width / 2.0, height / 2.0


def image_to_dac(
    points: np.ndarray,
    width: int,
    height: int,
    calib: Calibration | None = None,
) -> np.ndarray:
    """Map an ``(N, 2)`` array of image pixels to ``(N, 2)`` DAC coordinates.

    Image space has its origin top-left with y increasing downwards; DAC space
    has its origin bottom-left with y increasing upwards, so y is inverted here
    once and never again.
    """
    calib = calib or Calibration()
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if pts.size == 0:
        return np.zeros((0, 2), dtype=np.int32)

    scale, cx, cy = fit_transform(width, height, calib.scale_pct)
    x = COORD_CENTER + (pts[:, 0] - cx) * scale
    y = COORD_CENTER - (pts[:, 1] - cy) * scale  # y inversion

    x, y = _apply_calibration(x, y, calib)
    x += calib.offset_x
    y += calib.offset_y

    out = np.stack([x, y], axis=1)
    np.clip(out, COORD_MIN, COORD_MAX, out=out)
    return np.rint(out).astype(np.int32)


def _apply_calibration(x: np.ndarray, y: np.ndarray, calib: Calibration) -> tuple[np.ndarray, np.ndarray]:
    """Rotate then flip, about the centre of the DAC field."""
    rx = x - COORD_CENTER
    ry = y - COORD_CENTER

    rot = calib.normalised_rotate()
    if rot == 90:
        rx, ry = -ry, rx
    elif rot == 180:
        rx, ry = -rx, -ry
    elif rot == 270:
        rx, ry = ry, -rx

    if calib.flip_x:
        rx = -rx
    if calib.flip_y:
        ry = -ry

    return rx + COORD_CENTER, ry + COORD_CENTER


def clamp_point(x: int, y: int) -> tuple[int, int]:
    """Clamp a single point into the DAC field."""
    return (
        max(COORD_MIN, min(COORD_MAX, int(x))),
        max(COORD_MIN, min(COORD_MAX, int(y))),
    )


def polyline_length(points: np.ndarray) -> float:
    """Total arc length of an ``(N, 2)`` polyline, in whatever units it carries."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if len(pts) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))


__all__ = [
    "Calibration",
    "clamp_point",
    "fit_transform",
    "image_to_dac",
    "polyline_length",
]
