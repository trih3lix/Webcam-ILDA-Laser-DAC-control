"""The thread that keeps the scanners fed, and the only thing that talks to the DAC.

Separating this from the vision loop is what lets the projector show a steady
image while OpenCV runs at whatever rate it manages. The DAC loops its current
frame, so the pump's real jobs are to swap that frame when a new one is ready,
to notice when new frames have stopped arriving, and to guarantee the output
goes dark on the way out.

Every frame passes through :func:`webcam_ilda.safety.validate` here, regardless
of which stage produced it. Validation lives at the boundary on purpose: it
cannot be forgotten by a new code path, and a test-pattern bug is as capable of
parking the beam as a camera bug.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable, Optional

from .dac.base import FLAGS_DEFAULT, FLAGS_START_IMMEDIATELY, DacBackend, DacError, clamp_pps
from .frame import LaserFrame, blank_frame
from .safety import SafetyConfig, validate

log = logging.getLogger(__name__)


class DacPump:
    """Owns the DAC, and owns it exclusively."""

    def __init__(
        self,
        dac: DacBackend,
        pps: int = 30000,
        safety: Optional[SafetyConfig] = None,
        max_points: int = 4096,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.dac = dac
        self.pps = clamp_pps(pps)
        self.safety = safety or SafetyConfig()
        self.max_points = max_points
        self._clock = clock

        self._lock = threading.Lock()
        self._pending: Optional[LaserFrame] = None
        self._last_submit = clock()
        self._blanked = True
        self._muted = False
        self._stop_evt = threading.Event()
        self._thread: Optional[threading.Thread] = None

        # Observable state for the HUD.
        self.frames_written = 0
        self.points_written = 0
        self.last_error: Optional[str] = None

    # -- public API -----------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop_evt.clear()
        self._thread = threading.Thread(target=self._run, name="dac-pump", daemon=True)
        self._thread.start()

    def submit(self, frame: LaserFrame) -> None:
        """Hand over the next frame. Replaces any frame not yet written."""
        with self._lock:
            self._pending = frame
            self._last_submit = self._clock()

    def set_muted(self, muted: bool) -> None:
        """Soft blank: keep running, emit no light."""
        self._muted = bool(muted)

    @property
    def muted(self) -> bool:
        return self._muted

    @property
    def scan_fps(self) -> float:
        """Actual refresh rate of the projected image."""
        with self._lock:
            n = self._last_points
        return (self.pps / n) if n else 0.0

    def blank_and_stop(self) -> None:
        """Take the output dark. Safe to call repeatedly and from any thread."""
        try:
            # START_IMMEDIATELY pre-empts the looping frame rather than waiting
            # for it to finish -- the one place that flag is the right choice.
            self.dac.write_frame(blank_frame(), self.pps, FLAGS_START_IMMEDIATELY)
        except Exception:
            log.debug("blank frame write failed during shutdown", exc_info=True)
        try:
            self.dac.stop()
        except Exception:
            log.debug("Stop failed during shutdown", exc_info=True)
        try:
            self.dac.set_shutter(False)
        except Exception:
            log.debug("SetShutter failed during shutdown", exc_info=True)
        self._blanked = True

    def stop(self, timeout: float = 2.0) -> None:
        self._stop_evt.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None
        self.blank_and_stop()

    # -- internals ------------------------------------------------------------

    _last_points: int = 0

    def _take(self) -> tuple[Optional[LaserFrame], bool]:
        """Pop the pending frame, or report that the watchdog has expired."""
        with self._lock:
            frame = self._pending
            self._pending = None
            stale = (self._clock() - self._last_submit) > self.safety.watchdog_s
        return frame, stale

    def _run(self) -> None:
        while not self._stop_evt.is_set():
            frame, stale = self._take()

            if stale and not self._blanked:
                # The producer has gone quiet -- an RTSP dropout, an exception in
                # the vision loop, a wedged camera. Fail dark.
                log.warning("no frame for %.1fs; blanking output", self.safety.watchdog_s)
                self._write(blank_frame(), FLAGS_START_IMMEDIATELY)
                self._blanked = True

            if frame is None:
                time.sleep(0.002)
                continue

            if self._muted:
                frame = blank_frame()

            safe = validate(frame, self.safety, self.max_points)
            self._write(safe, FLAGS_DEFAULT)
            self._blanked = not safe.lit_points

    def _write(self, frame: LaserFrame, flags: int) -> None:
        try:
            if not self.dac.ready():
                # Not an error: the DAC is still playing. It will loop the
                # current frame, so dropping this one costs nothing but latency.
                return
            self.dac.write_frame(frame, self.pps, flags)
            self.frames_written += 1
            self.points_written += len(frame)
            self._last_points = len(frame)
            self.last_error = None
        except DacError as exc:
            self.last_error = str(exc)
            log.error("DAC write failed: %s", exc)
            time.sleep(0.05)
        except Exception as exc:  # pragma: no cover - defensive
            self.last_error = str(exc)
            log.exception("unexpected DAC failure; blanking")
            try:
                self.dac.stop()
            except Exception:
                pass
            time.sleep(0.05)


__all__ = ["DacPump"]
