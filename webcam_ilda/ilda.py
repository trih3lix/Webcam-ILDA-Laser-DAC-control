"""Reader and writer for the ILDA Image Data Transfer Format.

Supports the five record formats in common use:

===== ====================================== =========
Code  Meaning                                Bytes/rec
===== ====================================== =========
0     3D coordinates, indexed colour         8
1     2D coordinates, indexed colour         6
2     Colour palette                         3
4     3D coordinates, true colour            10
5     2D coordinates, true colour            8
===== ====================================== =========

Two robustness notes, both learned from real files rather than the spec:

* The reserved header bytes are *not* reliably zero. They must be ignored, not
  validated.
* A file is not guaranteed to end with the spec's zero-record trailer header.
  End-of-file is a legitimate end-of-stream.

Formats 4 and 5 store colour as **blue, green, red** in that order -- getting
this backwards silently swaps red and blue on every true-colour file.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterable, Optional, Sequence

from .frame import COORD_MAX, LaserFrame, LaserPoint

HEADER_STRUCT = ">4s3sB8s8sHHHBB"
HEADER_SIZE = struct.calcsize(HEADER_STRUCT)  # 32
MAGIC = b"ILDA"

#: Bytes per record, keyed by format code.
RECORD_SIZE = {0: 8, 1: 6, 2: 3, 4: 10, 5: 8}

STATUS_BLANK = 0x40
STATUS_LAST = 0x80

#: The ILDA default 64-colour palette, used when a file carries indexed colour
#: but no preceding format-2 palette section.
DEFAULT_PALETTE: tuple[tuple[int, int, int], ...] = (
    (255, 0, 0), (255, 16, 0), (255, 32, 0), (255, 48, 0),
    (255, 64, 0), (255, 80, 0), (255, 96, 0), (255, 112, 0),
    (255, 128, 0), (255, 144, 0), (255, 160, 0), (255, 176, 0),
    (255, 192, 0), (255, 208, 0), (255, 224, 0), (255, 240, 0),
    (255, 255, 0), (224, 255, 0), (192, 255, 0), (160, 255, 0),
    (128, 255, 0), (96, 255, 0), (64, 255, 0), (32, 255, 0),
    (0, 255, 0), (0, 255, 36), (0, 255, 73), (0, 255, 109),
    (0, 255, 146), (0, 255, 182), (0, 255, 219), (0, 255, 255),
    (0, 227, 255), (0, 198, 255), (0, 170, 255), (0, 142, 255),
    (0, 113, 255), (0, 85, 255), (0, 56, 255), (0, 28, 255),
    (0, 0, 255), (32, 0, 255), (64, 0, 255), (96, 0, 255),
    (128, 0, 255), (160, 0, 255), (192, 0, 255), (224, 0, 255),
    (255, 0, 255), (255, 32, 255), (255, 64, 255), (255, 96, 255),
    (255, 128, 255), (255, 160, 255), (255, 192, 255), (255, 224, 255),
    (255, 255, 255), (255, 224, 224), (255, 192, 192), (255, 160, 160),
    (255, 128, 128), (255, 96, 96), (255, 64, 64), (255, 32, 32),
)


class IldaError(ValueError):
    """Raised on a malformed ILDA stream."""


@dataclass(slots=True)
class IldaRecord:
    """One point as stored in the file, in native ILDA signed-16-bit space."""

    x: int
    y: int
    z: int = 0
    status: int = 0
    color_index: int = 0
    rgb: Optional[tuple[int, int, int]] = None

    @property
    def blank(self) -> bool:
        return bool(self.status & STATUS_BLANK)

    @property
    def last(self) -> bool:
        return bool(self.status & STATUS_LAST)


@dataclass(slots=True)
class IldaFrame:
    """A frame section: its header metadata plus its records."""

    format: int
    name: str
    company: str
    frame_number: int
    total_frames: int
    projector: int
    records: list[IldaRecord]

    def __len__(self) -> int:
        return len(self.records)


def ilda_to_dac(value: int) -> int:
    """Map a signed 16-bit ILDA coordinate onto the 12-bit DAC space.

    ``-32768..32767`` becomes ``0..4095``: shift the origin, then drop the low
    four bits the DAC cannot resolve.
    """
    return max(0, min(COORD_MAX, (int(value) + 32768) >> 4))


def dac_to_ilda(value: int) -> int:
    """Inverse of :func:`ilda_to_dac`, for writing frames back out."""
    return max(-32768, min(32767, (int(value) << 4) - 32768))


def _unpack_record(fmt: int, buf: bytes) -> IldaRecord:
    if fmt == 0:
        x, y, z, status, cindex = struct.unpack(">hhhBB", buf)
        return IldaRecord(x, y, z, status, cindex)
    if fmt == 1:
        x, y, status, cindex = struct.unpack(">hhBB", buf)
        return IldaRecord(x, y, 0, status, cindex)
    if fmt == 4:
        # True colour, 3D. Colour bytes are blue, green, red.
        x, y, z, status, b, g, r = struct.unpack(">hhhBBBB", buf)
        return IldaRecord(x, y, z, status, 0, (r, g, b))
    if fmt == 5:
        x, y, status, b, g, r = struct.unpack(">hhBBBB", buf)
        return IldaRecord(x, y, 0, status, 0, (r, g, b))
    raise IldaError(f"unsupported record format {fmt}")


def read(path: str | Path) -> list[IldaFrame]:
    """Read every frame section from an ILDA file."""
    with open(path, "rb") as fh:
        return read_stream(fh)


def read_stream(fh: BinaryIO) -> list[IldaFrame]:
    frames: list[IldaFrame] = []
    palette: list[tuple[int, int, int]] = []

    while True:
        head = fh.read(HEADER_SIZE)
        if len(head) < HEADER_SIZE:
            # End of file is a valid end of stream; the spec's zero-record
            # trailer header is frequently absent in the wild.
            break

        magic, _reserved, fmt, name_b, comp_b, count, num, total, projector, _pad = struct.unpack(
            HEADER_STRUCT, head
        )
        if magic != MAGIC:
            if not frames:
                raise IldaError(f"not an ILDA file: magic was {magic!r}")
            break
        if fmt not in RECORD_SIZE:
            raise IldaError(f"unsupported ILDA format code {fmt}")
        if count == 0:
            # The explicit end-of-file marker.
            break

        size = RECORD_SIZE[fmt]
        body = fh.read(count * size)
        if len(body) < count * size:
            raise IldaError(
                f"truncated frame: header declares {count} records "
                f"({count * size} bytes) but only {len(body)} remain"
            )

        name = name_b.decode("ascii", "replace").rstrip("\x00 ")
        company = comp_b.decode("ascii", "replace").rstrip("\x00 ")

        if fmt == 2:
            palette = [
                tuple(body[i * 3 : i * 3 + 3]) for i in range(count)  # type: ignore[misc]
            ]
            continue

        records = [_unpack_record(fmt, body[i * size : (i + 1) * size]) for i in range(count)]
        frames.append(
            IldaFrame(
                format=fmt,
                name=name,
                company=company,
                frame_number=num,
                total_frames=total,
                projector=projector,
                records=records,
            )
        )

    # Resolve indexed colours once the whole file has been seen, so a palette
    # section anywhere in the file applies to the frames that reference it.
    table = palette or list(DEFAULT_PALETTE)
    for f in frames:
        if f.format in (0, 1):
            for rec in f.records:
                rec.rgb = table[rec.color_index % len(table)] if table else (255, 255, 255)
    return frames


def to_laser_frame(frame: IldaFrame) -> LaserFrame:
    """Convert a parsed ILDA frame into the pipeline's DAC-space representation."""
    out = LaserFrame()
    for rec in frame.records:
        r, g, b = rec.rgb or (255, 255, 255)
        out.append(
            LaserPoint(
                x=ilda_to_dac(rec.x),
                y=ilda_to_dac(rec.y),
                r=r,
                g=g,
                b=b,
                blank=rec.blank,
            )
        )
    return out


def write(
    path: str | Path,
    frames: Sequence[LaserFrame],
    name: str = "WEBCAM",
    company: str = "JSLADE",
) -> None:
    """Write frames as ILDA format 5 (2D, true colour).

    Format 5 is the most faithful target for this pipeline: the colours are
    per-point RGB rather than palette indices, and the Z axis is unused.
    """
    with open(path, "wb") as fh:
        write_stream(fh, frames, name=name, company=company)


def write_stream(
    fh: BinaryIO,
    frames: Sequence[LaserFrame],
    name: str = "WEBCAM",
    company: str = "JSLADE",
) -> None:
    total = len(frames)
    name_b = name.encode("ascii", "replace")[:8].ljust(8, b"\x00")
    comp_b = company.encode("ascii", "replace")[:8].ljust(8, b"\x00")

    for idx, frame in enumerate(frames):
        fh.write(
            struct.pack(
                HEADER_STRUCT,
                MAGIC, b"\x00\x00\x00", 5, name_b, comp_b,
                len(frame), idx, total, 0, 0,
            )
        )
        last = len(frame) - 1
        for i, p in enumerate(frame.points):
            status = 0
            if p.blank:
                status |= STATUS_BLANK
            if i == last:
                status |= STATUS_LAST
            # Blue, green, red -- the ILDA byte order for true-colour records.
            fh.write(
                struct.pack(
                    ">hhBBBB",
                    dac_to_ilda(p.x), dac_to_ilda(p.y), status,
                    0 if p.blank else p.b,
                    0 if p.blank else p.g,
                    0 if p.blank else p.r,
                )
            )

    # Spec-compliant end-of-file marker: a header declaring zero records.
    fh.write(
        struct.pack(HEADER_STRUCT, MAGIC, b"\x00\x00\x00", 5, name_b, comp_b, 0, 0, total, 0, 0)
    )


def load_laser_frames(path: str | Path) -> list[LaserFrame]:
    """Convenience: read a file straight into projectable frames."""
    return [to_laser_frame(f) for f in read(path)]


__all__ = [
    "DEFAULT_PALETTE",
    "HEADER_SIZE",
    "IldaError",
    "IldaFrame",
    "IldaRecord",
    "RECORD_SIZE",
    "STATUS_BLANK",
    "STATUS_LAST",
    "dac_to_ilda",
    "ilda_to_dac",
    "load_laser_frames",
    "read",
    "read_stream",
    "to_laser_frame",
    "write",
    "write_stream",
]
