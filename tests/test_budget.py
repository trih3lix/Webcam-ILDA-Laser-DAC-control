"""The point budget is a hard limit, not a target."""

from __future__ import annotations

import numpy as np
import pytest

from webcam_ilda.dac.base import MAX_POINTS
from webcam_ilda.pathopt import LaserPath, OptimiserConfig, build_frame, point_budget


def _blob(cx: float, cy: float, r: float) -> LaserPath:
    t = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    return LaserPath(np.stack([cx + r * np.cos(t), cy + r * np.sin(t)], axis=1), closed=True)


def test_budget_is_scan_rate_over_frame_rate():
    assert point_budget(30000, 25) == 1200
    assert point_budget(20000, 30) == 666


def test_budget_never_exceeds_the_firmware_frame_buffer():
    assert point_budget(65535, 1) == MAX_POINTS


def test_budget_respects_an_explicit_lower_cap():
    assert point_budget(30000, 25, max_points=500) == 500


@pytest.mark.parametrize("bad", [(0, 25), (30000, 0), (-1, 25)])
def test_budget_rejects_nonsense(bad):
    with pytest.raises(ValueError):
        point_budget(*bad)


def test_busy_scene_is_trimmed_to_budget():
    paths = [_blob(200 + (i % 10) * 380, 200 + (i // 10) * 380, 120) for i in range(100)]
    frame, stats = build_frame(paths, budget=1200)
    assert len(frame) <= 1200
    assert stats["dropped"] > 0
    assert stats["kept"] < 100


def test_the_largest_contours_are_the_ones_kept():
    big = _blob(2048, 2048, 900)
    smalls = [_blob(300 + i * 60, 300, 20) for i in range(40)]
    frame, stats = build_frame([*smalls, big], budget=600)
    # The dominant shape survives: its extent should still be visible.
    min_x, min_y, max_x, max_y = frame.bounds()
    assert (max_x - min_x) > 1000
    assert len(frame) <= 600


def test_a_single_oversized_contour_is_coarsened_rather_than_dropped():
    """One huge shape cannot be dropped -- there is nothing else to show."""
    huge = _blob(2048, 2048, 1900)
    cfg = OptimiserConfig(step_draw=5.0)  # absurdly fine: guarantees overflow
    frame, stats = build_frame([huge], budget=400, cfg=cfg)
    assert 0 < len(frame) <= 400
    assert stats["step"] > cfg.step_draw
    assert frame.lit_points


def test_budget_of_one_still_produces_a_valid_frame():
    frame, _ = build_frame([_blob(2048, 2048, 500)], budget=1)
    assert len(frame) <= 1


def test_frame_never_exceeds_hardware_maximum():
    paths = [_blob(200 + (i % 10) * 380, 200 + (i // 10) * 380, 150) for i in range(80)]
    frame, _ = build_frame(paths, budget=MAX_POINTS, cfg=OptimiserConfig(step_draw=2.0))
    assert len(frame) <= MAX_POINTS
