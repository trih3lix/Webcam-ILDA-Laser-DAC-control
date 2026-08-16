"""Secondary DAC backend: pure pyusb, no vendor binary.

Clean-room implementation against the publicly documented Helios USB protocol.
Exists for Linux and macOS, where the vendored Windows DLL is useless, and as
insurance if the DLL ever fails to load. Needs ``pip install .[usb]`` plus a
libusb-1.0 runtime.

Unlike the SDK path, this backend must implement the 64-byte packet workaround
itself -- see :func:`webcam_ilda.dac.base.needs_packet_workaround`.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

from ..frame import LaserFrame
from .base import (
    ALT_SETTING,
    CMD_GET_FWVERSION,
    CMD_GET_NAME,
    CMD_GET_STATUS,
    CMD_SET_SDK_VERSION,
    CMD_SHUTTER,
    CMD_STOP,
    EP_BULK_OUT,
    EP_INT_IN,
    EP_INT_OUT,
    FLAGS_DEFAULT,
    HELIOS_PID,
    HELIOS_SDK_VERSION,
    HELIOS_VID,
    INTERFACE_NUM,
    DacError,
    clamp_pps,
    encode_frame,
    validate_frame_shape,
)

log = logging.getLogger(__name__)

CONTROL_TIMEOUT_MS = 32
BULK_TIMEOUT_MS = 512
STATUS_POLL_INTERVAL = 0.0005
STATUS_TIMEOUT = 0.5


def pyusb_available() -> bool:
    try:
        import usb.core  # noqa: F401
    except Exception:
        return False
    return True


class HeliosUsbDac:
    """Helios DAC driven directly over USB with pyusb."""

    def __init__(self, device: int = 0) -> None:
        self._index = device
        self._dev = None
        self._name = f"Helios #{device} (pyusb)"
        self._opened = False

    @property
    def name(self) -> str:
        return self._name

    def open(self) -> int:
        try:
            import usb.core
            import usb.util
        except ImportError as exc:
            raise DacError(
                "pyusb is not installed. Install it with: pip install .[usb]"
            ) from exc

        devices = list(usb.core.find(find_all=True, idVendor=HELIOS_VID, idProduct=HELIOS_PID))
        if not devices:
            raise DacError("no Helios DAC found on the USB bus")
        if self._index >= len(devices):
            raise DacError(f"device {self._index} requested but only {len(devices)} found")

        dev = devices[self._index]
        dev.set_configuration()
        # Interface 0 alt-setting 1 is the bulk/interrupt pair used for output;
        # alt-setting 0 is the low-bandwidth control-only mode.
        usb.util.claim_interface(dev, INTERFACE_NUM)
        dev.set_interface_altsetting(interface=INTERFACE_NUM, alternate_setting=ALT_SETTING)
        self._dev = dev
        self._opened = True

        self._send_control(CMD_SET_SDK_VERSION, HELIOS_SDK_VERSION)
        name = self._get_name()
        if name:
            self._name = name
        log.info("opened %s (firmware %s) over pyusb", self._name, self._get_fw_version())
        return len(devices)

    # -- control channel ------------------------------------------------------

    def _send_control(self, command: int, arg: int = 0) -> None:
        if self._dev is None:
            raise DacError("device is closed")
        # Control words go out as a little-endian uint16 with the argument in
        # the high byte.
        payload = bytes(((command | (arg << 8)) & 0xFF, ((command | (arg << 8)) >> 8) & 0xFF))
        self._dev.write(EP_INT_OUT, payload, CONTROL_TIMEOUT_MS)

    def _read_control(self, expect: int, length: int = 32) -> Optional[bytes]:
        if self._dev is None:
            return None
        try:
            data = bytes(self._dev.read(EP_INT_IN, length, CONTROL_TIMEOUT_MS))
        except Exception:
            return None
        # Responses echo the command with bit 7 set in the first byte.
        if not data or data[0] != expect:
            return None
        return data

    def _get_name(self) -> Optional[str]:
        self._send_control(CMD_GET_NAME)
        data = self._read_control(0x85)
        if not data:
            return None
        raw = bytes(data[1:])
        return raw.split(b"\x00", 1)[0].decode("ascii", "replace").strip() or None

    def _get_fw_version(self) -> Optional[int]:
        self._send_control(CMD_GET_FWVERSION)
        data = self._read_control(0x84, length=5)
        if not data or len(data) < 5:
            return None
        return int.from_bytes(data[1:5], "little")

    # -- backend protocol -----------------------------------------------------

    def ready(self) -> bool:
        if not self._opened:
            return False
        self._send_control(CMD_GET_STATUS)
        data = self._read_control(0x83, length=2)
        return bool(data and data[1] == 1)

    def _wait_ready(self) -> bool:
        deadline = time.monotonic() + STATUS_TIMEOUT
        while time.monotonic() < deadline:
            if self.ready():
                return True
            time.sleep(STATUS_POLL_INTERVAL)
        return False

    def write_frame(self, frame: LaserFrame, pps: int, flags: int = FLAGS_DEFAULT) -> None:
        if not self._opened or self._dev is None:
            raise DacError("write_frame() called on a closed device")
        pps = clamp_pps(pps)
        validate_frame_shape(frame, pps)

        payload = encode_frame(frame, pps, flags)
        if not self._wait_ready():
            log.warning("DAC not ready after %.0f ms; issuing Stop and retrying", STATUS_TIMEOUT * 1e3)
            self._send_control(CMD_STOP)
            if not self._wait_ready():
                raise DacError("DAC did not become ready; output blanked")
        self._dev.write(EP_BULK_OUT, payload, BULK_TIMEOUT_MS)

    def stop(self) -> None:
        if self._opened:
            try:
                self._send_control(CMD_STOP)
            except Exception:
                log.debug("Stop failed", exc_info=True)

    def set_shutter(self, open_: bool) -> None:
        if self._opened:
            try:
                self._send_control(CMD_SHUTTER, 1 if open_ else 0)
            except Exception:
                log.debug("SetShutter failed", exc_info=True)

    def close(self) -> None:
        if self._opened and self._dev is not None:
            try:
                self.stop()
                self.set_shutter(False)
            finally:
                try:
                    import usb.util

                    usb.util.release_interface(self._dev, INTERFACE_NUM)
                    usb.util.dispose_resources(self._dev)
                except Exception:
                    log.debug("USB cleanup failed", exc_info=True)
        self._opened = False
        self._dev = None
