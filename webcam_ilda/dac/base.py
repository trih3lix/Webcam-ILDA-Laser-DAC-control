"""Backend-independent Helios constants, the backend protocol, and wire encoding.

The numbers here are the public Helios DAC interface (see the MIT-licensed
upstream SDK, `Grix/helios_dac`). They are shared by the ctypes backend, the
clean-room pyusb backend, and the simulator so that all three enforce identical
limits -- a frame that the simulator accepts is a frame the hardware accepts.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..frame import COORD_MAX, COORD_MIN, LaserFrame

# --- device identity ---------------------------------------------------------
HELIOS_VID = 0x1209
HELIOS_PID = 0xE500

# --- USB endpoints (pyusb backend) -------------------------------------------
EP_BULK_OUT = 0x02
EP_INT_OUT = 0x06
EP_INT_IN = 0x83
INTERFACE_NUM = 0
ALT_SETTING = 1

# --- limits ------------------------------------------------------------------
#: Maximum points in a single frame. The firmware's frame buffer is 4096 points.
MAX_POINTS = 4096
MAX_PPS = 65535
MIN_PPS = 7

# --- WriteFrame flags --------------------------------------------------------
#: Loop this frame until another is written. The steady-state choice: it keeps
#: the scanners fed even when the vision pipeline runs slower than the scan rate.
FLAGS_DEFAULT = 0
#: Interrupt the frame currently playing instead of waiting for it to finish.
#: Produces a visible discontinuity, so it is reserved for emergency blanking.
FLAGS_START_IMMEDIATELY = 1 << 0
#: Play once and stop, rather than looping.
FLAGS_SINGLE_MODE = 1 << 1
#: Return immediately rather than waiting for the previous frame to complete.
FLAGS_DONT_BLOCK = 1 << 2

#: SDK version the firmware is told we speak.
HELIOS_SDK_VERSION = 6

# --- control words (pyusb backend) -------------------------------------------
CMD_STOP = 0x0001
CMD_SHUTTER = 0x0002
CMD_GET_STATUS = 0x0003
CMD_GET_FWVERSION = 0x0004
CMD_GET_NAME = 0x0005
CMD_SET_NAME = 0x0006
CMD_SET_SDK_VERSION = 0x0007


class DacError(RuntimeError):
    """Raised when a DAC backend cannot open, or fails a write irrecoverably."""


def clamp_pps(pps: int) -> int:
    """Clamp a requested scan rate into the range the firmware accepts."""
    return max(MIN_PPS, min(MAX_PPS, int(pps)))


def validate_frame_shape(frame: LaserFrame, pps: int) -> None:
    """Raise :class:`DacError` if a frame could not be written at all.

    This is a *structural* check (would the firmware reject it?), distinct from
    the beam-safety validation in :mod:`webcam_ilda.safety`. Every backend runs
    it, so the simulator fails on exactly what the hardware would fail on.
    """
    n = len(frame)
    if n == 0:
        raise DacError("refusing to write an empty frame")
    if n > MAX_POINTS:
        raise DacError(f"frame has {n} points, maximum is {MAX_POINTS}")
    if not (MIN_PPS <= pps <= MAX_PPS):
        raise DacError(f"pps {pps} outside {MIN_PPS}..{MAX_PPS}")
    for i, p in enumerate(frame.points):
        if not (COORD_MIN <= p.x <= COORD_MAX and COORD_MIN <= p.y <= COORD_MAX):
            raise DacError(
                f"point {i} at ({p.x}, {p.y}) is outside the "
                f"{COORD_MIN}..{COORD_MAX} DAC coordinate space"
            )


def needs_packet_workaround(num_points: int) -> bool:
    """Does this point count land the USB transfer on an exact 64-byte multiple?

    The Helios MCU mis-handles a bulk transfer whose length is an exact multiple
    of the 64-byte USB packet size. A frame is ``7 * n + 5`` bytes on the wire,
    so the bad case is ``(7n + 5) % 64 == 0``, i.e. ``n % 64 == 45``. The stock
    SDK works around it by dropping the final point; the pyusb backend must do
    the same or those frames arrive corrupted.
    """
    return num_points % 64 == 45


def encode_point(x: int, y: int, r: int, g: int, b: int, i: int) -> bytes:
    """Pack one point into the 7-byte Helios wire format.

    X and Y are 12-bit, packed across three bytes as ``xxxxxxxx xxxxyyyy
    yyyyyyyy``, followed by one byte each of red, green, blue and intensity.
    """
    return bytes(
        (
            (x >> 4) & 0xFF,
            ((x & 0x0F) << 4) | ((y >> 8) & 0x0F),
            y & 0xFF,
            r & 0xFF,
            g & 0xFF,
            b & 0xFF,
            i & 0xFF,
        )
    )


def encode_frame(frame: LaserFrame, pps: int, flags: int = FLAGS_DEFAULT) -> bytes:
    """Pack a whole frame for the bulk endpoint, applying the packet workaround.

    Returns the payload only; the caller is responsible for the transfer. When
    the workaround trims a point, the scan rate is rescaled so the frame still
    takes the same wall-clock time and the visual result is unchanged.
    """
    points = frame.points
    n = len(points)
    if needs_packet_workaround(n):
        # Dropping one point shortens the frame; slow the scan rate by the same
        # ratio so the frame's duration -- and therefore its flicker rate -- holds.
        pps = clamp_pps(round(pps * (n - 1) / n))
        points = points[: n - 1]
        n -= 1

    out = bytearray()
    for p in points:
        if p.blank:
            out += encode_point(p.x, p.y, 0, 0, 0, 0)
        else:
            out += encode_point(p.x, p.y, p.r, p.g, p.b, p.intensity)
    out += bytes(
        (
            pps & 0xFF,
            (pps >> 8) & 0xFF,
            n & 0xFF,
            (n >> 8) & 0xFF,
            flags & 0xFF,
        )
    )
    return bytes(out)


@runtime_checkable
class DacBackend(Protocol):
    """What every DAC backend must provide.

    Deliberately small: open, ask if it can take a frame, write one, stop, close.
    Anything cleverer belongs in :mod:`webcam_ilda.pump`, which is backend-blind.
    """

    @property
    def name(self) -> str:
        """Human-readable device name, for logs and the preview HUD."""
        ...

    def open(self) -> int:
        """Open the device. Returns the number of devices found."""
        ...

    def ready(self) -> bool:
        """True when the device will accept a new frame right now."""
        ...

    def write_frame(self, frame: LaserFrame, pps: int, flags: int = FLAGS_DEFAULT) -> None:
        """Send a frame. Must call :func:`validate_frame_shape` first."""
        ...

    def stop(self) -> None:
        """Halt output immediately and blank."""
        ...

    def set_shutter(self, open_: bool) -> None:
        """Drive the DAC's shutter line."""
        ...

    def close(self) -> None:
        """Release the device."""
        ...
