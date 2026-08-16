# Third-party components

## Helios Laser DAC SDK

`webcam_ilda/vendor/HeliosLaserDAC.dll` is a compiled binary from the official
Helios Laser DAC SDK by Gitle Mikkelsen (Bitlasers).

- Upstream: <https://github.com/Grix/helios_dac>
- License: MIT
- Architecture: x86-64 (PE machine `0x8664`)
- SHA-256: `51a6cf77e9ba48dead64c7bfa8653788eb57f2d2d664dc2f48bb2938e03ef83b`

It is vendored so that a fresh clone can drive real hardware on Windows with no
additional downloads. It is loaded through `ctypes` by
`webcam_ilda/dac/helios_dll.py`; no SDK source is copied into this project.

The alternative backend `webcam_ilda/dac/helios_usb.py` is a **clean-room**
implementation written against the publicly documented Helios USB protocol
(vendor/product IDs, endpoint numbers, point wire format, control words). It
shares no code with the SDK.

MIT License text for the Helios SDK:

```
MIT License

Copyright (c) 2018 Gitle Mikkelsen

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Test fixture

`tests/data/ILDATEST.ILD` is a small ILDA Image Data Transfer Format file
(format 0, single frame, 1191 points) used to exercise the ILDA reader. It is
included as a binary test fixture only.
