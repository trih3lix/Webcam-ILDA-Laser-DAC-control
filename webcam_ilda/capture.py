"""Frame sources, and the thread that keeps only the newest frame.

Deliberately source-agnostic: a USB webcam, an RTSP stream off a network camera,
an HTTP snapshot endpoint, a video file, a still image, or a synthetic generator
all arrive as the same thing. The dev machine this was written on has no webcam
attached at all -- the first real footage came off an RTSP camera -- which is
exactly why the abstraction earns its keep.

The mailbox pattern in :class:`CaptureThread` is not optional for RTSP. OpenCV
buffers decoded frames, so a consumer slower than the stream falls progressively
further behind until it is projecting the past. Grabbing continuously and
keeping only the latest frame bounds the lag at one frame.
"""

from __future__ import annotations

import logging
import threading
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

log = logging.getLogger(__name__)

RECONNECT_MIN = 0.5
RECONNECT_MAX = 8.0


class CaptureError(RuntimeError):
    """Raised when a source cannot be opened at all."""


@dataclass(slots=True)
class SourceSpec:
    """A parsed ``--source`` argument."""

    kind: str          #: device | stream | snapshot | file | image | synthetic
    value: str
    index: int = 0


def parse_source(spec: str, kind_hint: str = "auto") -> SourceSpec:
    """Work out what the user meant by ``--source``."""
    text = str(spec).strip()
    if text.lower() in ("synthetic", "sim", "test"):
        return SourceSpec("synthetic", text)
    if text.isdigit():
        return SourceSpec("device", text, index=int(text))
    lowered = text.lower()
    if lowered.startswith(("rtsp://", "rtsps://")):
        return SourceSpec("stream", text)
    if lowered.startswith(("http://", "https://")):
        if kind_hint == "snapshot" or lowered.endswith((".jpg", ".jpeg", ".png")):
            return SourceSpec("snapshot", text)
        return SourceSpec("stream", text)
    path = Path(text)
    if path.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"):
        return SourceSpec("image", text)
    return SourceSpec("file", text)


class FrameSource:
    """Base class: something that yields BGR numpy frames."""

    def read(self) -> Optional[np.ndarray]:
        raise NotImplementedError

    def release(self) -> None:
        pass

    @property
    def description(self) -> str:
        return self.__class__.__name__

    @staticmethod
    def create(spec: str, kind_hint: str = "auto", width: int = 640, height: int = 480) -> "FrameSource":
        parsed = parse_source(spec, kind_hint)
        if parsed.kind == "synthetic":
            from .patterns import SyntheticSource

            return SyntheticSource(width, height)
        if parsed.kind == "snapshot":
            return SnapshotSource(parsed.value)
        if parsed.kind == "image":
            return StillImageSource(parsed.value)
        return VideoCaptureSource(parsed)


class VideoCaptureSource(FrameSource):
    """Webcams, RTSP streams, MJPEG streams and video files, via OpenCV."""

    def __init__(self, spec: SourceSpec) -> None:
        self.spec = spec
        self._cap: Optional[cv2.VideoCapture] = None
        self._open()

    def _open(self) -> None:
        if self.spec.kind == "device":
            # DirectShow avoids the multi-second MSMF open delay on Windows.
            backend = cv2.CAP_DSHOW if hasattr(cv2, "CAP_DSHOW") else cv2.CAP_ANY
            cap = cv2.VideoCapture(self.spec.index, backend)
        else:
            cap = cv2.VideoCapture(self.spec.value, cv2.CAP_FFMPEG)
        if not cap or not cap.isOpened():
            raise CaptureError(
                f"could not open source {self.spec.value!r}. "
                "For a webcam try a different index; for RTSP check the URL and credentials."
            )
        try:
            # Ask the driver to keep the smallest possible buffer. Not every
            # backend honours it, hence the mailbox thread as well.
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        self._cap = cap

    def read(self) -> Optional[np.ndarray]:
        if self._cap is None:
            return None
        ok, frame = self._cap.read()
        if not ok:
            if self.spec.kind == "file":
                # Loop video files so a recorded clip makes a continuous demo.
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = self._cap.read()
            if not ok:
                return None
        return frame

    def reconnect(self) -> bool:
        self.release()
        try:
            self._open()
            return True
        except CaptureError:
            return False

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    @property
    def description(self) -> str:
        return f"{self.spec.kind}:{self.spec.value}"


class SnapshotSource(FrameSource):
    """Polls an HTTP JPEG endpoint -- the shape most camera APIs offer."""

    def __init__(self, url: str, timeout: float = 5.0, min_interval: float = 0.2) -> None:
        self.url = url
        self.timeout = timeout
        self.min_interval = min_interval
        self._last = 0.0
        self._cached: Optional[np.ndarray] = None

    def read(self) -> Optional[np.ndarray]:
        now = time.monotonic()
        if self._cached is not None and (now - self._last) < self.min_interval:
            return self._cached
        try:
            with urllib.request.urlopen(self.url, timeout=self.timeout) as resp:
                data = resp.read()
        except Exception as exc:
            log.warning("snapshot fetch failed: %s", exc)
            return self._cached
        buf = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is None:
            log.warning("snapshot response was not a decodable image")
            return self._cached
        self._cached = img
        self._last = now
        return img

    @property
    def description(self) -> str:
        return f"snapshot:{self.url}"


class StillImageSource(FrameSource):
    """A single image, decoded once and replayed. Useful for calibration."""

    def __init__(self, path: str) -> None:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise CaptureError(f"could not read image {path!r}")
        self._img = img
        self.path = path

    def read(self) -> Optional[np.ndarray]:
        return self._img

    @property
    def description(self) -> str:
        return f"image:{self.path}"


class CaptureThread:
    """Reads a source as fast as it will go, keeping only the newest frame."""

    def __init__(self, source: FrameSource) -> None:
        self.source = source
        self._frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.frames_read = 0
        self.last_frame_time = 0.0

    def start(self) -> "CaptureThread":
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="capture", daemon=True)
        self._thread.start()
        return self

    def _run(self) -> None:
        backoff = RECONNECT_MIN
        while not self._stop.is_set():
            try:
                frame = self.source.read()
            except Exception:
                log.exception("capture read failed")
                frame = None

            if frame is None:
                log.warning("source returned no frame; retrying in %.1fs", backoff)
                if not self._stop.wait(backoff):
                    reconnect = getattr(self.source, "reconnect", None)
                    if callable(reconnect):
                        reconnect()
                backoff = min(RECONNECT_MAX, backoff * 2)
                continue

            backoff = RECONNECT_MIN
            with self._lock:
                self._frame = frame
                self.frames_read += 1
                self.last_frame_time = time.monotonic()

    def latest(self) -> Optional[np.ndarray]:
        with self._lock:
            return self._frame

    def stop(self, timeout: float = 2.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None
        self.source.release()


__all__ = [
    "CaptureError",
    "CaptureThread",
    "FrameSource",
    "SnapshotSource",
    "SourceSpec",
    "StillImageSource",
    "VideoCaptureSource",
    "parse_source",
]
