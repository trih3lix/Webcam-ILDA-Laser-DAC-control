"""The preview window: what the laser is actually being told to do.

This draws the point stream, not the camera image. That distinction is the whole
point -- the camera view tells you what the software sees, but only the point
stream tells you what the galvos are being asked to trace, including the blanked
travel moves that decide whether the projection looks clean or smeared.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from .frame import COORD_MAX, LaserFrame

BG = (12, 12, 14)
TRAVEL_COLOR = (70, 70, 70)
HUD_COLOR = (200, 200, 200)
WARN_COLOR = (60, 120, 255)


class Preview:
    """An OpenCV window rendering DAC space plus a HUD."""

    WINDOW = "webcam-ilda"

    def __init__(self, size: int = 720, show_travel: bool = True, show_camera: bool = True) -> None:
        self.size = size
        self.show_travel = show_travel
        self.show_camera = show_camera
        self._created = False

    def _ensure_window(self) -> None:
        if not self._created:
            cv2.namedWindow(self.WINDOW, cv2.WINDOW_AUTOSIZE)
            self._created = True

    def _to_px(self, x: int, y: int) -> tuple[int, int]:
        """DAC coords to window pixels, flipping y back for screen display."""
        s = self.size / float(COORD_MAX)
        return int(x * s), int(self.size - y * s)

    def render(
        self,
        frame: LaserFrame,
        hud: str = "",
        camera: Optional[np.ndarray] = None,
        warning: str = "",
    ) -> np.ndarray:
        canvas = np.full((self.size, self.size, 3), BG, dtype=np.uint8)

        prev = None
        for p in frame.points:
            cur = self._to_px(p.x, p.y)
            if prev is not None:
                if p.blank:
                    if self.show_travel:
                        cv2.line(canvas, prev, cur, TRAVEL_COLOR, 1, cv2.LINE_AA)
                else:
                    # OpenCV is BGR; LaserPoint is RGB.
                    cv2.line(canvas, prev, cur, (p.b, p.g, p.r), 2, cv2.LINE_AA)
            prev = cur

        if self.show_camera and camera is not None and camera.size:
            self._draw_pip(canvas, camera)

        if hud:
            cv2.rectangle(canvas, (0, self.size - 28), (self.size, self.size), (0, 0, 0), -1)
            cv2.putText(
                canvas, hud, (8, self.size - 9),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, HUD_COLOR, 1, cv2.LINE_AA,
            )
        if warning:
            cv2.putText(
                canvas, warning, (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, WARN_COLOR, 1, cv2.LINE_AA,
            )
        return canvas

    def _draw_pip(self, canvas: np.ndarray, camera: np.ndarray) -> None:
        pip_w = self.size // 4
        h, w = camera.shape[:2]
        pip_h = max(1, int(pip_w * h / float(w)))
        pip = cv2.resize(camera, (pip_w, pip_h), interpolation=cv2.INTER_AREA)
        canvas[4 : 4 + pip_h, self.size - pip_w - 4 : self.size - 4] = pip
        cv2.rectangle(
            canvas,
            (self.size - pip_w - 4, 4),
            (self.size - 4, 4 + pip_h),
            (90, 90, 90),
            1,
        )

    def show(self, image: np.ndarray) -> int:
        """Display a rendered canvas and return the key pressed, or -1."""
        self._ensure_window()
        cv2.imshow(self.WINDOW, image)
        return cv2.waitKey(1) & 0xFF

    def close(self) -> None:
        if self._created:
            cv2.destroyWindow(self.WINDOW)
            self._created = False


HOTKEY_HELP = """
Hotkeys
  ESC        emergency stop -- blank, stop, exit
  SPACE      mute / unmute output (soft blank)
  m          cycle extraction mode
  [ ]        Canny low threshold  -10 / +10
  { }        Canny high threshold -10 / +10
  a          toggle automatic thresholds
  f / g      flip X / flip Y
  r          rotate 90 degrees
  - =        scale down / up
  , .        brightness cap down / up
  t          show / hide blanked travel moves
  c          show / hide camera picture-in-picture
  b          relearn background (motion mode)
  p          freeze the current frame
  s          save tuning to webcam_ilda.yaml
""".strip()


__all__ = ["HOTKEY_HELP", "Preview"]
