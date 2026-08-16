"""A DAC backend with no DAC behind it.

This is what makes the project testable and demonstrable without hardware. It
enforces exactly the same validation as the real backends and models frame
timing, so the pump thread paces itself identically in ``--dry-run`` as it does
against a Helios. Frames are retained in a ring buffer for tests and for the
preview window.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Callable, Deque, Optional

from ..frame import LaserFrame
from .base import FLAGS_DEFAULT, FLAGS_START_IMMEDIATELY, clamp_pps, validate_frame_shape


class SimulatorDac:
    """In-memory stand-in for a Helios DAC.

    :param history: how many recent frames to retain.
    :param clock: injectable time source, so tests need not sleep.
    """

    def __init__(
        self,
        history: int = 8,
        clock: Callable[[], float] = time.monotonic,
        on_frame: Optional[Callable[[LaserFrame, int], None]] = None,
    ) -> None:
        self._clock = clock
        self._on_frame = on_frame
        self._lock = threading.Lock()
        self.frames: Deque[tuple[LaserFrame, int]] = deque(maxlen=history)
        self.frame_count = 0
        self.stopped = False
        self.shutter_open = False
        self._opened = False
        # Wall-clock time at which the currently-playing frame finishes.
        self._busy_until = 0.0

    @property
    def name(self) -> str:
        return "Simulator (no hardware)"

    def open(self) -> int:
        self._opened = True
        self.stopped = False
        return 1

    def ready(self) -> bool:
        if not self._opened:
            return False
        return self._clock() >= self._busy_until

    def write_frame(self, frame: LaserFrame, pps: int, flags: int = FLAGS_DEFAULT) -> None:
        pps = clamp_pps(pps)
        validate_frame_shape(frame, pps)
        with self._lock:
            self.frames.append((frame.copy(), pps))
            self.frame_count += 1
            self.stopped = False
            # A frame of n points at p points-per-second occupies n/p seconds.
            # START_IMMEDIATELY pre-empts whatever is playing, so the clock resets.
            duration = len(frame) / float(pps)
            now = self._clock()
            if flags & FLAGS_START_IMMEDIATELY:
                self._busy_until = now + duration
            else:
                self._busy_until = max(now, self._busy_until) + duration
        if self._on_frame is not None:
            self._on_frame(frame, pps)

    def stop(self) -> None:
        with self._lock:
            self.stopped = True
            self._busy_until = 0.0

    def set_shutter(self, open_: bool) -> None:
        self.shutter_open = bool(open_)

    def close(self) -> None:
        self.stop()
        self.set_shutter(False)
        self._opened = False

    # -- test helpers ---------------------------------------------------------

    @property
    def last_frame(self) -> Optional[LaserFrame]:
        with self._lock:
            return self.frames[-1][0] if self.frames else None

    @property
    def last_pps(self) -> Optional[int]:
        with self._lock:
            return self.frames[-1][1] if self.frames else None
