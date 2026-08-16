"""The controls that decide whether a frame is allowed to reach the projector."""

from __future__ import annotations

import pytest

from webcam_ilda.frame import LaserFrame, LaserPoint, blank_frame
from webcam_ilda.safety import (
    FrameRejected,
    SafetyConfig,
    check_bounds,
    check_extent,
    check_point_count,
    check_static_beam,
    validate,
)


def _moving_frame(n: int = 40) -> LaserFrame:
    return LaserFrame([LaserPoint(500 + i * 40, 500 + i * 40, 0, 255, 0) for i in range(n)])


def test_a_normal_moving_frame_passes():
    frame = _moving_frame()
    out = validate(frame, SafetyConfig(max_brightness=1.0), 4096)
    assert len(out) == len(frame)
    assert out.lit_points


def test_parked_beam_is_rejected():
    """The actual hazard: a lit beam that stops moving."""
    frame = LaserFrame([LaserPoint(2048, 2048, 0, 255, 0) for _ in range(500)])
    with pytest.raises(FrameRejected, match="stationary"):
        check_static_beam(frame, SafetyConfig())
    assert not validate(frame, SafetyConfig(), 4096).lit_points


def test_legitimate_corner_dwell_is_not_rejected():
    """Dwell points are short by design and must survive the check."""
    cfg = SafetyConfig(max_static_run=12)
    pts = [LaserPoint(100 + i * 50, 100, 0, 255, 0) for i in range(10)]
    pts += [LaserPoint(550, 100, 0, 255, 0) for _ in range(4)]  # a 4-point dwell
    pts += [LaserPoint(550, 100 + i * 50, 0, 255, 0) for i in range(1, 10)]
    check_static_beam(LaserFrame(pts), cfg)  # must not raise


def test_blanked_points_may_repeat_freely():
    """A blanked beam emits nothing, so parking it is harmless."""
    frame = LaserFrame([LaserPoint(2048, 2048, 0, 0, 0, blank=True) for _ in range(500)])
    check_static_beam(frame, SafetyConfig())  # must not raise


def test_collapsed_frame_is_rejected():
    frame = LaserFrame(
        [LaserPoint(2048 + (i % 3), 2048 + (i % 2), 0, 255, 0) for i in range(30)]
    )
    with pytest.raises(FrameRejected, match="spans only"):
        check_extent(frame, SafetyConfig())


def test_fully_blanked_frame_is_always_acceptable():
    check_extent(blank_frame(), SafetyConfig())  # must not raise
    out = validate(blank_frame(), SafetyConfig(), 4096)
    assert not out.lit_points


def test_out_of_bounds_point_is_rejected():
    frame = LaserFrame([LaserPoint(0, 0), LaserPoint(9999, 10)])
    with pytest.raises(FrameRejected, match="outside the projector field"):
        check_bounds(frame)


def test_empty_and_oversized_frames_are_rejected():
    with pytest.raises(FrameRejected, match="empty"):
        check_point_count(LaserFrame(), 4096)
    with pytest.raises(FrameRejected, match="limit"):
        check_point_count(_moving_frame(100), 50)


def test_brightness_cap_is_applied():
    frame = LaserFrame([LaserPoint(100 + i * 50, 100, 200, 100, 50) for i in range(20)])
    out = validate(frame, SafetyConfig(max_brightness=0.5), 4096)
    assert (out.points[0].r, out.points[0].g, out.points[0].b) == (100, 50, 25)


def test_brightness_cap_does_not_mutate_the_callers_frame():
    """Upstream must keep its full-intensity copy for the preview window."""
    frame = LaserFrame([LaserPoint(100 + i * 50, 100, 200, 200, 200) for i in range(20)])
    validate(frame, SafetyConfig(max_brightness=0.25), 4096)
    assert frame.points[0].r == 200


def test_max_brightness_is_clamped_to_a_sane_range():
    assert SafetyConfig(max_brightness=5.0).max_brightness == 1.0
    assert SafetyConfig(max_brightness=-1.0).max_brightness == 0.0


def test_validate_never_raises_it_blanks():
    """The pump must always have something safe to write."""
    hazardous = LaserFrame([LaserPoint(2048, 2048, 255, 255, 255) for _ in range(400)])
    out = validate(hazardous, SafetyConfig(), 4096)
    assert len(out) > 0
    assert not out.lit_points


def test_blanked_points_report_zero_intensity():
    assert LaserPoint(0, 0, 255, 255, 255, blank=True).intensity == 0
    assert LaserPoint(0, 0, 255, 128, 0).intensity == 255
