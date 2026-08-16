"""Deciding what colour each contour is drawn in.

Three strategies, because they suit different scenes. ``fixed`` is the honest
default -- a single-colour projector is the common case and a green trace is the
brightest per milliwatt. ``sample`` reads the source image along each contour,
which on an RGB projector makes the laser drawing genuinely look like the scene.
``rainbow`` is for when the subject is motion rather than colour.
"""

from __future__ import annotations

import colorsys

import numpy as np

MODES = ("fixed", "sample", "rainbow")


def parse_color(text: str) -> tuple[int, int, int]:
    """Parse ``"R,G,B"`` (0-255 each) into a tuple."""
    parts = [p.strip() for p in str(text).split(",")]
    if len(parts) != 3:
        raise ValueError(f"expected 'R,G,B', got {text!r}")
    try:
        rgb = tuple(int(p) for p in parts)
    except ValueError as exc:
        raise ValueError(f"colour components must be integers, got {text!r}") from exc
    if any(not 0 <= c <= 255 for c in rgb):
        raise ValueError(f"colour components must be 0-255, got {text!r}")
    return rgb  # type: ignore[return-value]


def sample_contour_color(
    image: np.ndarray,
    contour: np.ndarray,
    band: int = 3,
    boost: float = 1.0,
) -> tuple[int, int, int]:
    """Mean colour of the source image in a small band around a contour.

    Sampling *on* the edge would pick up the transition between subject and
    background, so this reads a few pixels either side and averages. The result
    is saturated a little, because a laser renders a desaturated colour as a
    washed-out beam.
    """
    if image is None or image.size == 0 or len(contour) == 0:
        return (0, 255, 0)

    h, w = image.shape[:2]
    pts = np.asarray(contour, dtype=np.int32).reshape(-1, 2)
    # Sample a sparse subset; a contour with 400 points does not need 400 reads.
    idx = np.linspace(0, len(pts) - 1, min(len(pts), 32)).astype(int)
    samples: list[np.ndarray] = []
    for i in idx:
        x, y = pts[i]
        x0, x1 = max(0, x - band), min(w, x + band + 1)
        y0, y1 = max(0, y - band), min(h, y + band + 1)
        if x1 > x0 and y1 > y0:
            samples.append(image[y0:y1, x0:x1].reshape(-1, 3).mean(axis=0))
    if not samples:
        return (0, 255, 0)

    bgr = np.mean(samples, axis=0)
    r, g, b = float(bgr[2]), float(bgr[1]), float(bgr[0])

    peak = max(r, g, b, 1.0)
    r, g, b = (c / peak * 255.0 * boost for c in (r, g, b))
    return (
        int(max(0, min(255, r))),
        int(max(0, min(255, g))),
        int(max(0, min(255, b))),
    )


def rainbow_color(index: int, total: int) -> tuple[int, int, int]:
    """Evenly spaced hues across the contours in a frame."""
    hue = (index / max(1, total)) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
    return (int(r * 255), int(g * 255), int(b * 255))


def assign_colors(
    contours: list[np.ndarray],
    mode: str = "fixed",
    fixed: tuple[int, int, int] = (0, 255, 0),
    image: np.ndarray | None = None,
) -> list[tuple[int, int, int]]:
    """Return one colour per contour, according to ``mode``."""
    mode = mode.lower()
    if mode not in MODES:
        raise ValueError(f"unknown colour mode {mode!r}; choose from {', '.join(MODES)}")
    if mode == "fixed":
        return [fixed] * len(contours)
    if mode == "rainbow":
        return [rainbow_color(i, len(contours)) for i in range(len(contours))]
    return [sample_contour_color(image, c) for c in contours]


__all__ = ["MODES", "assign_colors", "parse_color", "rainbow_color", "sample_contour_color"]
