"""Wire-format encoding, structure layout, and the firmware packet workaround."""

from __future__ import annotations

import ctypes

import pytest

from webcam_ilda.dac.base import (
    FLAGS_DEFAULT,
    MAX_POINTS,
    MAX_PPS,
    MIN_PPS,
    DacError,
    clamp_pps,
    encode_frame,
    encode_point,
    needs_packet_workaround,
    validate_frame_shape,
)
from webcam_ilda.dac.helios_dll import HeliosPointStruct
from webcam_ilda.frame import LaserFrame, LaserPoint


def test_helios_point_struct_is_eight_packed_bytes():
    assert ctypes.sizeof(HeliosPointStruct) == 8


def test_point_encoding_packs_twelve_bit_coordinates():
    # x = 4095 (0xFFF), y = 4095 -> FF F0 FF ...
    assert encode_point(4095, 4095, 1, 2, 3, 4) == bytes((0xFF, 0xFF, 0xFF, 1, 2, 3, 4))
    # x = 0, y = 0
    assert encode_point(0, 0, 0, 0, 0, 0) == bytes(7)
    # x = 4095, y = 0 -> high nibble of byte 1 set, low nibble clear
    assert encode_point(4095, 0, 0, 0, 0, 0)[:3] == bytes((0xFF, 0xF0, 0x00))
    # x = 0, y = 4095 -> low nibble of byte 1 set
    assert encode_point(0, 4095, 0, 0, 0, 0)[:3] == bytes((0x00, 0x0F, 0xFF))


def test_point_encoding_is_seven_bytes():
    assert len(encode_point(1, 2, 3, 4, 5, 6)) == 7


def test_frame_encoding_length_and_trailer():
    frame = LaserFrame([LaserPoint(10, 20, 1, 2, 3) for _ in range(10)])
    data = encode_frame(frame, 30000, FLAGS_DEFAULT)
    assert len(data) == 10 * 7 + 5
    pps_lo, pps_hi, n_lo, n_hi, flags = data[-5:]
    assert pps_lo | (pps_hi << 8) == 30000
    assert n_lo | (n_hi << 8) == 10
    assert flags == FLAGS_DEFAULT


def test_blanked_points_encode_as_dark():
    frame = LaserFrame([LaserPoint(10, 20, 255, 255, 255, blank=True)])
    assert encode_frame(frame, 1000)[:7] == encode_point(10, 20, 0, 0, 0, 0)


@pytest.mark.parametrize("n,expected", [(45, True), (109, True), (44, False), (46, False), (64, False)])
def test_packet_workaround_detects_the_bad_lengths(n, expected):
    """(7n + 5) must not be an exact multiple of the 64-byte USB packet."""
    assert needs_packet_workaround(n) is expected
    if expected:
        assert (7 * n + 5) % 64 == 0


def test_packet_workaround_drops_a_point_and_rescales_pps():
    frame = LaserFrame([LaserPoint(i, i, 0, 255, 0) for i in range(45)])
    data = encode_frame(frame, 30000)
    n = data[-3] | (data[-2] << 8)
    pps = data[-5] | (data[-4] << 8)
    assert n == 44
    assert len(data) == 44 * 7 + 5
    # Duration is preserved: 44/pps' should match 45/30000.
    assert pps == round(30000 * 44 / 45)


def test_pps_clamping():
    assert clamp_pps(0) == MIN_PPS
    assert clamp_pps(10**9) == MAX_PPS
    assert clamp_pps(30000) == 30000


def test_validate_rejects_empty_and_oversized_frames():
    with pytest.raises(DacError, match="empty"):
        validate_frame_shape(LaserFrame(), 30000)
    big = LaserFrame([LaserPoint(0, 0) for _ in range(MAX_POINTS + 1)])
    with pytest.raises(DacError, match="maximum"):
        validate_frame_shape(big, 30000)


def test_validate_rejects_out_of_range_coordinates_and_pps():
    with pytest.raises(DacError, match="coordinate space"):
        validate_frame_shape(LaserFrame([LaserPoint(4096, 0)]), 30000)
    with pytest.raises(DacError, match="pps"):
        validate_frame_shape(LaserFrame([LaserPoint(0, 0)]), 999999)
