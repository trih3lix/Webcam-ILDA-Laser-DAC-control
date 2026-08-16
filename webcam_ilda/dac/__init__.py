"""DAC backends and the resolver that picks one."""

from __future__ import annotations

import logging

from .base import (
    FLAGS_DEFAULT,
    FLAGS_SINGLE_MODE,
    FLAGS_START_IMMEDIATELY,
    MAX_POINTS,
    MAX_PPS,
    MIN_PPS,
    DacBackend,
    DacError,
    clamp_pps,
    encode_frame,
    encode_point,
    needs_packet_workaround,
    validate_frame_shape,
)
from .simulator import SimulatorDac

log = logging.getLogger(__name__)

BACKENDS = ("auto", "dll", "pyusb", "sim")


def open_dac(backend: str = "auto", device: int = 0, dll_path: str | None = None) -> DacBackend:
    """Return an *opened* DAC backend.

    ``auto`` prefers the vendor DLL, falls back to pyusb, and never silently
    falls back to the simulator -- a run that thinks it is driving a laser and
    is not would be worse than an error. Ask for ``sim`` (or ``--dry-run``)
    explicitly.
    """
    backend = backend.lower()
    if backend not in BACKENDS:
        raise DacError(f"unknown backend {backend!r}; choose from {', '.join(BACKENDS)}")

    if backend == "sim":
        dac = SimulatorDac()
        dac.open()
        return dac

    if backend in ("auto", "dll"):
        from .helios_dll import HeliosDllDac

        try:
            dac = HeliosDllDac(device=device, dll_path=dll_path)
            dac.open()
            return dac
        except DacError:
            if backend == "dll":
                raise
            log.info("SDK DLL backend unavailable, trying pyusb", exc_info=True)

    from .helios_usb import HeliosUsbDac

    dac = HeliosUsbDac(device=device)
    dac.open()
    return dac


__all__ = [
    "BACKENDS",
    "DacBackend",
    "DacError",
    "FLAGS_DEFAULT",
    "FLAGS_SINGLE_MODE",
    "FLAGS_START_IMMEDIATELY",
    "MAX_POINTS",
    "MAX_PPS",
    "MIN_PPS",
    "SimulatorDac",
    "clamp_pps",
    "encode_frame",
    "encode_point",
    "needs_packet_workaround",
    "open_dac",
    "validate_frame_shape",
]
