"""Primary DAC backend: ctypes against the official Helios SDK DLL.

Chosen as primary because the SDK is the vendor's own MIT-licensed code, ships
vendored in this repo, needs no third-party Python packages (ctypes is stdlib),
and handles the USB quirks internally -- including the 64-byte packet workaround
that the raw protocol requires.
"""

from __future__ import annotations

import ctypes
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

from ..frame import LaserFrame
from .base import (
    FLAGS_DEFAULT,
    MAX_POINTS,
    DacError,
    clamp_pps,
    validate_frame_shape,
)

log = logging.getLogger(__name__)

#: Where the vendored DLL lives inside the package.
VENDOR_DLL = Path(__file__).resolve().parent.parent / "vendor" / "HeliosLaserDAC.dll"

#: GetStatus polling: how long to wait for the device to accept a frame, and how
#: long to sleep between polls. 500 us keeps latency low without spinning a core.
STATUS_POLL_INTERVAL = 0.0005
STATUS_TIMEOUT = 0.5


class HeliosPointStruct(ctypes.Structure):
    """The SDK's ``HeliosPoint``: two 16-bit coords then RGBI, tightly packed."""

    _pack_ = 1
    _fields_ = [
        ("x", ctypes.c_uint16),
        ("y", ctypes.c_uint16),
        ("r", ctypes.c_uint8),
        ("g", ctypes.c_uint8),
        ("b", ctypes.c_uint8),
        ("i", ctypes.c_uint8),
    ]


def _candidate_paths(explicit: Optional[str]) -> list[Path]:
    """DLL search order: explicit flag, env var, vendored copy, then the loader."""
    out: list[Path] = []
    if explicit:
        out.append(Path(explicit))
    env = os.environ.get("HELIOS_DLL")
    if env:
        out.append(Path(env))
    out.append(VENDOR_DLL)
    out.append(Path("HeliosLaserDAC.dll"))
    return out


def _load_library(explicit: Optional[str]) -> ctypes.CDLL:
    tried: list[str] = []
    for path in _candidate_paths(explicit):
        try:
            return ctypes.CDLL(str(path))
        except OSError as exc:  # not present, or wrong architecture
            tried.append(f"{path}: {exc}")
    hint = ""
    if sys.maxsize <= 2**32:
        hint = (
            "\nThe vendored DLL is 64-bit; you appear to be running 32-bit "
            "Python. Use a 64-bit interpreter."
        )
    raise DacError(
        "could not load HeliosLaserDAC.dll. Tried:\n  " + "\n  ".join(tried) + hint
    )


class HeliosDllDac:
    """Helios DAC driven through the vendor SDK."""

    def __init__(self, device: int = 0, dll_path: Optional[str] = None) -> None:
        self._device = device
        self._dll_path = dll_path
        self._lib: Optional[ctypes.CDLL] = None
        self._name = f"Helios #{device}"
        self._opened = False

    @property
    def name(self) -> str:
        return self._name

    def _bind(self, lib: ctypes.CDLL) -> None:
        """Declare argument/return types so ctypes marshals correctly on all ABIs."""
        lib.OpenDevices.restype = ctypes.c_int
        lib.CloseDevices.restype = ctypes.c_int
        lib.GetStatus.argtypes = [ctypes.c_uint]
        lib.GetStatus.restype = ctypes.c_int
        lib.WriteFrame.argtypes = [
            ctypes.c_uint,
            ctypes.c_int,
            ctypes.c_ubyte,
            ctypes.POINTER(HeliosPointStruct),
            ctypes.c_int,
        ]
        lib.WriteFrame.restype = ctypes.c_int
        lib.Stop.argtypes = [ctypes.c_uint]
        lib.Stop.restype = ctypes.c_int
        lib.SetShutter.argtypes = [ctypes.c_uint, ctypes.c_bool]
        lib.SetShutter.restype = ctypes.c_int
        lib.GetName.argtypes = [ctypes.c_uint, ctypes.c_char_p]
        lib.GetName.restype = ctypes.c_int
        lib.GetFirmwareVersion.argtypes = [ctypes.c_uint]
        lib.GetFirmwareVersion.restype = ctypes.c_int

    def open(self) -> int:
        lib = _load_library(self._dll_path)
        self._bind(lib)
        count = lib.OpenDevices()
        if count <= 0:
            raise DacError(
                "no Helios DAC found. Check the USB cable and that the device "
                "enumerates, or run with --dry-run to use the simulator."
            )
        if self._device >= count:
            lib.CloseDevices()
            raise DacError(f"device {self._device} requested but only {count} found")
        self._lib = lib
        self._opened = True

        buf = ctypes.create_string_buffer(32)
        if lib.GetName(self._device, buf) == 1:
            self._name = buf.value.decode("ascii", "replace").strip()
        fw = lib.GetFirmwareVersion(self._device)
        log.info("opened %s (firmware %s), %d device(s) present", self._name, fw, count)
        return count

    def ready(self) -> bool:
        if not self._opened or self._lib is None:
            return False
        return self._lib.GetStatus(self._device) == 1

    def _wait_ready(self) -> bool:
        deadline = time.monotonic() + STATUS_TIMEOUT
        while time.monotonic() < deadline:
            if self.ready():
                return True
            time.sleep(STATUS_POLL_INTERVAL)
        return False

    def write_frame(self, frame: LaserFrame, pps: int, flags: int = FLAGS_DEFAULT) -> None:
        if not self._opened or self._lib is None:
            raise DacError("write_frame() called on a closed device")
        pps = clamp_pps(pps)
        validate_frame_shape(frame, pps)

        n = len(frame)
        buf = (HeliosPointStruct * n)()
        for idx, p in enumerate(frame.points):
            s = buf[idx]
            s.x = p.x
            s.y = p.y
            if p.blank:
                s.r = s.g = s.b = s.i = 0
            else:
                s.r, s.g, s.b, s.i = p.r, p.g, p.b, p.intensity

        if not self._wait_ready():
            # The device is wedged. Stop resets its frame state; one retry, then
            # surface it -- the pump turns an exception into a blanked output.
            log.warning("DAC not ready after %.0f ms; issuing Stop and retrying", STATUS_TIMEOUT * 1e3)
            self._lib.Stop(self._device)
            if not self._wait_ready():
                raise DacError("DAC did not become ready; output blanked")

        rc = self._lib.WriteFrame(self._device, pps, flags, buf, n)
        if rc != 1:
            raise DacError(f"WriteFrame failed with code {rc}")

    def stop(self) -> None:
        if self._opened and self._lib is not None:
            self._lib.Stop(self._device)

    def set_shutter(self, open_: bool) -> None:
        if self._opened and self._lib is not None:
            self._lib.SetShutter(self._device, bool(open_))

    def close(self) -> None:
        if self._opened and self._lib is not None:
            try:
                self._lib.Stop(self._device)
                self._lib.SetShutter(self._device, False)
            finally:
                self._lib.CloseDevices()
        self._opened = False
        self._lib = None


__all__ = ["HeliosDllDac", "HeliosPointStruct", "MAX_POINTS", "VENDOR_DLL"]
