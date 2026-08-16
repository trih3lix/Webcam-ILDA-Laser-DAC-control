"""Built-in test patterns and a synthetic camera.

The test patterns are the first thing to project at a new venue: known geometry,
known size, low power. A square tells you instantly whether the projection is
mirrored, rotated, clipped, or keystoned, and whether the scan rate is too high
for the scanners (the corners round off).

:class:`SyntheticSource` stands in for a camera so that CI, and anyone without
hardware, can exercise the whole pipeline end to end.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from .frame import COORD_CENTER, COORD_MAX
from .pathopt import LaserPath

PATTERNS = ("square", "circle", "grid", "cross")


def test_pattern(name: str, scale_pct: float = 70.0, color: tuple[int, int, int] = (0, 255, 0)) -> list[LaserPath]:
    """Return the named alignment pattern as paths in DAC coordinates."""
    name = name.lower()
    half = (scale_pct / 100.0) * COORD_MAX / 2.0
    c = COORD_CENTER

    if name == "square":
        pts = np.array(
            [[c - half, c - half], [c + half, c - half], [c + half, c + half], [c - half, c + half]]
        )
        return [LaserPath(pts, closed=True, color=color)]

    if name == "circle":
        t = np.linspace(0, 2 * math.pi, 72, endpoint=False)
        pts = np.stack([c + half * np.cos(t), c + half * np.sin(t)], axis=1)
        return [LaserPath(pts, closed=True, color=color)]

    if name == "cross":
        return [
            LaserPath(np.array([[c - half, c], [c + half, c]]), closed=False, color=color),
            LaserPath(np.array([[c, c - half], [c, c + half]]), closed=False, color=color),
        ]

    if name == "grid":
        paths: list[LaserPath] = []
        steps = 5
        for i in range(steps):
            frac = -1.0 + 2.0 * i / (steps - 1)
            paths.append(
                LaserPath(np.array([[c - half, c + half * frac], [c + half, c + half * frac]]), color=color)
            )
            paths.append(
                LaserPath(np.array([[c + half * frac, c - half], [c + half * frac, c + half]]), color=color)
            )
        return paths

    raise ValueError(f"unknown test pattern {name!r}; choose from {', '.join(PATTERNS)}")


class SyntheticSource:
    """A moving scene generated in software, so no camera is required.

    Produces a rotating polygon and an orbiting circle on a dark field -- enough
    structure that edge detection, contour extraction, path ordering and the
    point budget all do real work.
    """

    def __init__(self, width: int = 640, height: int = 480) -> None:
        self.width = width
        self.height = height
        self._t = 0

    def read(self) -> np.ndarray:
        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        cx, cy = self.width // 2, self.height // 2
        phase = self._t * 0.06
        self._t += 1

        radius = min(cx, cy) * 0.55
        pts = []
        for k in range(5):
            a = phase + k * 2 * math.pi / 5
            pts.append([cx + radius * math.cos(a), cy + radius * math.sin(a)])
        cv2.polylines(img, [np.array(pts, dtype=np.int32)], True, (255, 255, 255), 2)

        orbit = min(cx, cy) * 0.85
        ox = int(cx + orbit * math.cos(-phase * 1.7))
        oy = int(cy + orbit * math.sin(-phase * 1.7))
        cv2.circle(img, (ox, oy), 28, (255, 255, 255), 2)
        cv2.rectangle(img, (20, 20), (110, 90), (255, 255, 255), 2)
        return img

    def release(self) -> None:
        pass

    @property
    def description(self) -> str:
        return "synthetic"


__all__ = ["PATTERNS", "SyntheticSource", "test_pattern"]
