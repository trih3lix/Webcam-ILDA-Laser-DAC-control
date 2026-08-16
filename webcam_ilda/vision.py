"""Camera frame in, polylines out.

A laser draws lines, so the vision stage's only job is to find lines worth
drawing. Four modes, because "what should the laser trace?" has genuinely
different answers depending on the scene:

``canny``
    Edges everywhere. Busy, detailed, and the best general answer.
``threshold``
    Adaptive binarisation then external contours -- clean silhouettes, which
    read far better than edge soup when the subject is backlit or high-contrast.
``motion``
    Background subtraction, so only what moves gets drawn. This is the mode that
    makes the demo: stand in front of the camera and the laser outlines *you*,
    and nothing else in the room.
``color``
    An HSV key. Hold up something in the keyed colour and only that is traced.

Everything is simplified with Douglas-Peucker before it leaves here. A raw
contour has one point per boundary pixel, and the laser has a budget of about a
thousand points for the entire scene.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

MODES = ("canny", "threshold", "motion", "color")

WORKING_WIDTH = 640


@dataclass(slots=True)
class VisionConfig:
    """Tuning for the extraction stage. Most of these are live hotkeys."""

    mode: str = "canny"
    canny_lo: int = 80
    canny_hi: int = 160
    auto_threshold: bool = False
    #: Douglas-Peucker tolerance in pixels. The single biggest lever on how many
    #: points the scene costs.
    simplify_px: float = 2.0
    #: Contours shorter than this are noise, not subject matter.
    min_arc_px: float = 30.0
    min_bbox_px: int = 8
    #: Hard cap before the optimiser even sees them.
    max_paths: int = 48
    blur_ksize: int = 5
    adaptive_block: int = 21
    adaptive_c: int = 5
    key_hue: int = 60
    key_hue_tol: int = 15
    key_sat_min: int = 60
    key_val_min: int = 60

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"unknown vision mode {self.mode!r}; choose from {', '.join(MODES)}")
        if self.blur_ksize % 2 == 0:
            self.blur_ksize += 1
        if self.adaptive_block % 2 == 0:
            self.adaptive_block += 1


class VisionPipeline:
    """Holds the per-mode state that has to persist between frames."""

    def __init__(self, config: VisionConfig | None = None) -> None:
        self.config = config or VisionConfig()
        self._bg = None
        self.last_edges: np.ndarray | None = None

    def _background(self):
        if self._bg is None:
            self._bg = cv2.createBackgroundSubtractorMOG2(
                history=300, varThreshold=25, detectShadows=False
            )
        return self._bg

    def reset_background(self) -> None:
        """Forget the learned background -- use after moving the camera."""
        self._bg = None

    def preprocess(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Downscale to the working width and return ``(bgr, blurred_gray)``."""
        h, w = frame.shape[:2]
        if w > WORKING_WIDTH:
            scale = WORKING_WIDTH / float(w)
            frame = cv2.resize(frame, (WORKING_WIDTH, int(round(h * scale))), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        k = self.config.blur_ksize
        gray = cv2.GaussianBlur(gray, (k, k), 0)
        return frame, gray

    def _mask(self, bgr: np.ndarray, gray: np.ndarray) -> tuple[np.ndarray, int]:
        """Produce the binary image contours are found in, and the retrieval mode."""
        cfg = self.config

        if cfg.mode == "canny":
            lo, hi = cfg.canny_lo, cfg.canny_hi
            if cfg.auto_threshold:
                # Otsu-free auto thresholds: a well-known heuristic that tracks
                # the frame's median brightness, so it survives changing light.
                median = float(np.median(gray))
                lo = int(max(0, 0.66 * median))
                hi = int(min(255, 1.33 * median))
            edges = cv2.Canny(gray, lo, hi)
            self.last_edges = edges
            # RETR_LIST because edge images have no meaningful nesting -- Canny
            # returns ridges, so each edge yields an outline on both sides.
            return edges, cv2.RETR_LIST

        if cfg.mode == "threshold":
            binary = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
                cfg.adaptive_block, cfg.adaptive_c,
            )
            self.last_edges = binary
            return binary, cv2.RETR_EXTERNAL

        if cfg.mode == "motion":
            mask = self._background().apply(gray)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            self.last_edges = mask
            return mask, cv2.RETR_EXTERNAL

        # colour key
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        # OpenCV hue is 0-179, so a degree-based key is halved here.
        centre = (cfg.key_hue // 2) % 180
        tol = max(1, cfg.key_hue_tol // 2)
        lo = np.array([max(0, centre - tol), cfg.key_sat_min, cfg.key_val_min], dtype=np.uint8)
        hi = np.array([min(179, centre + tol), 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lo, hi)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        self.last_edges = mask
        return mask, cv2.RETR_EXTERNAL

    def extract(self, frame: np.ndarray) -> tuple[list[np.ndarray], np.ndarray]:
        """Run the pipeline. Returns ``(contours, working_bgr_image)``.

        Contours are ``(N, 2)`` int arrays in working-image pixel coordinates,
        simplified and ordered largest-first.
        """
        cfg = self.config
        bgr, gray = self.preprocess(frame)
        mask, retr = self._mask(bgr, gray)

        found, _ = cv2.findContours(mask, retr, cv2.CHAIN_APPROX_NONE)

        kept: list[np.ndarray] = []
        for c in found:
            if cv2.arcLength(c, False) < cfg.min_arc_px:
                continue
            _, _, bw, bh = cv2.boundingRect(c)
            if bw < cfg.min_bbox_px and bh < cfg.min_bbox_px:
                continue
            closed = _is_closed(c)
            simplified = cv2.approxPolyDP(c, cfg.simplify_px, closed)
            pts = simplified.reshape(-1, 2)
            if len(pts) < 2:
                continue
            kept.append(pts)

        # Longest first: if the budget forces cuts, lose the small stuff.
        kept.sort(key=lambda p: cv2.arcLength(p.reshape(-1, 1, 2).astype(np.int32), False), reverse=True)
        return kept[: cfg.max_paths], bgr


def _is_closed(contour: np.ndarray, tol: float = 3.0) -> bool:
    """A contour is closed if its ends meet."""
    pts = contour.reshape(-1, 2)
    if len(pts) < 3:
        return False
    return bool(np.linalg.norm(pts[0].astype(float) - pts[-1].astype(float)) < tol)


def contour_is_closed(contour: np.ndarray, tol: float = 3.0) -> bool:
    """Public wrapper for :func:`_is_closed`."""
    return _is_closed(contour, tol)


__all__ = ["MODES", "WORKING_WIDTH", "VisionConfig", "VisionPipeline", "contour_is_closed"]
