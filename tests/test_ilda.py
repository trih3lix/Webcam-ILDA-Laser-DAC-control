"""ILDA reader/writer, checked against a real file rather than a synthetic one."""

from __future__ import annotations

import io
import struct
from pathlib import Path

import pytest

from webcam_ilda import ilda
from webcam_ilda.frame import LaserFrame, LaserPoint

FIXTURE = Path(__file__).parent / "data" / "ILDATEST.ILD"


def test_fixture_is_exactly_header_plus_records():
    """The file has no end-of-file trailer -- EOF is the end of the stream."""
    size = FIXTURE.stat().st_size
    assert size == ilda.HEADER_SIZE + 1191 * ilda.RECORD_SIZE[0] == 9560


def test_reads_single_format0_frame():
    frames = ilda.read(FIXTURE)
    assert len(frames) == 1
    frame = frames[0]
    assert frame.format == 0
    assert len(frame) == 1191


def test_tolerates_junk_in_reserved_header_bytes():
    """Real files do not zero the reserved bytes; validating them would fail."""
    head = FIXTURE.read_bytes()[: ilda.HEADER_SIZE]
    assert head[4:7] != b"\x00\x00\x00"
    assert ilda.read(FIXTURE)  # parses anyway


def test_first_point_is_blanked():
    rec = ilda.read(FIXTURE)[0].records[0]
    assert rec.x == 12800
    assert rec.y == -32767
    assert rec.status & ilda.STATUS_BLANK
    assert rec.blank is True


def test_indexed_colors_resolve_through_default_palette():
    frame = ilda.read(FIXTURE)[0]
    assert all(r.rgb is not None for r in frame.records)
    assert frame.records[0].rgb in ilda.DEFAULT_PALETTE


def test_conversion_to_dac_space_stays_in_bounds():
    laser = ilda.to_laser_frame(ilda.read(FIXTURE)[0])
    assert len(laser) == 1191
    assert all(0 <= p.x <= 4095 and 0 <= p.y <= 4095 for p in laser)
    assert laser.lit_points  # the file is not entirely blanked


@pytest.mark.parametrize(
    "ilda_value,expected",
    [(-32768, 0), (0, 2048), (32767, 4095), (32752, 4095)],
)
def test_ilda_to_dac_endpoints(ilda_value, expected):
    assert ilda.ilda_to_dac(ilda_value) == expected


def test_dac_to_ilda_round_trips_within_quantisation():
    for dac in (0, 1, 2048, 4000, 4095):
        assert ilda.ilda_to_dac(ilda.dac_to_ilda(dac)) == dac


def test_write_read_round_trip_preserves_geometry_and_colour():
    frame = LaserFrame(
        [
            LaserPoint(100, 200, 255, 0, 0, blank=True),
            LaserPoint(1000, 2000, 0, 255, 0),
            LaserPoint(4095, 4095, 0, 0, 255),
        ]
    )
    buf = io.BytesIO()
    ilda.write_stream(buf, [frame])
    buf.seek(0)
    parsed = ilda.read_stream(buf)

    assert len(parsed) == 1
    out = ilda.to_laser_frame(parsed[0])
    assert len(out) == 3
    for original, result in zip(frame.points, out.points):
        assert result.x == original.x
        assert result.y == original.y
        assert result.blank == original.blank
        if not original.blank:
            # Format 5 stores blue, green, red; a byte-order slip shows up here.
            assert (result.r, result.g, result.b) == (original.r, original.g, original.b)


def test_writer_emits_end_of_file_marker():
    buf = io.BytesIO()
    ilda.write_stream(buf, [LaserFrame([LaserPoint(0, 0), LaserPoint(10, 10)])])
    tail = buf.getvalue()[-ilda.HEADER_SIZE :]
    magic, _res, fmt, _n, _c, count, _num, _total, _pid, _pad = struct.unpack(
        ilda.HEADER_STRUCT, tail
    )
    assert magic == b"ILDA"
    assert count == 0


def test_truncated_file_is_rejected():
    data = FIXTURE.read_bytes()[: ilda.HEADER_SIZE + 40]
    with pytest.raises(ilda.IldaError, match="truncated"):
        ilda.read_stream(io.BytesIO(data))


def test_non_ilda_input_is_rejected():
    with pytest.raises(ilda.IldaError, match="not an ILDA file"):
        ilda.read_stream(io.BytesIO(b"NOPE" + b"\x00" * 64))
