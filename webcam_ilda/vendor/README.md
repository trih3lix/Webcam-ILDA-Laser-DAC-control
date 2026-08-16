# Vendored binaries

## `HeliosLaserDAC.dll`

The official Helios Laser DAC SDK for Windows, vendored so a fresh clone can
drive real hardware without a separate download.

| | |
| --- | --- |
| Upstream | <https://github.com/Grix/helios_dac> |
| Author | Gitle Mikkelsen (Bitlasers) |
| License | MIT |
| Architecture | x86-64 (PE machine `0x8664`) |
| SHA-256 | `51a6cf77e9ba48dead64c7bfa8653788eb57f2d2d664dc2f48bb2938e03ef83b` |

Loaded via `ctypes` in [`../dac/helios_dll.py`](../dac/helios_dll.py). The
search order is `--dll-path`, then `$HELIOS_DLL`, then this file, then the
system loader — so a newer SDK can be substituted without editing the package.

Functions used: `OpenDevices`, `CloseDevices`, `GetStatus`, `WriteFrame`,
`Stop`, `SetShutter`, `GetName`, `GetFirmwareVersion`.

Verified against firmware version 5 on device `Helios 875966520`.
