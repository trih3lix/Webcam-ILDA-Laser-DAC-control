"""Invariants the point stream must satisfy for the projection to look right."""

from __future__ import annotations

import math

import numpy as np
import pytest

from webcam_ilda.pathopt import (
    LaserPath,
    OptimiserConfig,
    build_frame,
    order_paths,
    travel_distance,
)

SQUARE = np.array([[1000, 1000], [3000, 1000], [3000, 3000], [1000, 3000]], dtype=float)


def _line(x0, y0, x1, y1):
    return LaserPath(np.array([[x0, y0], [x1, y1]], dtype=float))


def test_frame_starts_blanked():
    """The travel from wherever the last frame ended must never be lit."""
    frame, _ = build_frame([LaserPath(SQUARE, closed=True)], budget=2000)
    assert frame.points[0].blank is True


def test_consecutive_lit_points_are_within_the_step():
    cfg = OptimiserConfig(step_draw=60.0)
    frame, _ = build_frame([LaserPath(SQUARE, closed=True)], budget=4096, cfg=cfg)
    prev = None
    for p in frame.points:
        if p.lit() and prev is not None:
            d = math.dist((p.x, p.y), (prev.x, prev.y))
            assert d <= cfg.step_draw + 1.5
        prev = p if p.lit() else None


def test_travel_between_paths_is_fully_blanked():
    """Two separated shapes: everything between them must be dark."""
    a = LaserPath(np.array([[500, 500], [800, 500]], dtype=float))
    b = LaserPath(np.array([[3000, 3000], [3300, 3000]], dtype=float))
    frame, _ = build_frame([a, b], budget=4096)

    lit_x = [p.x for p in frame.points if p.lit()]
    # Nothing may be lit in the empty middle of the field.
    assert not [x for x in lit_x if 1000 < x < 2900]


def test_closed_contour_returns_to_its_start():
    frame, _ = build_frame([LaserPath(SQUARE, closed=True)], budget=4096)
    lit = frame.lit_points
    start, end = lit[0], lit[-1]
    assert math.dist((start.x, start.y), (end.x, end.y)) <= 2.0


def test_square_corners_get_dwell_points():
    """Every 90-degree corner must be repeated so the galvos can turn."""
    cfg = OptimiserConfig(corner_threshold_deg=25.0)
    frame, _ = build_frame([LaserPath(SQUARE, closed=True)], budget=4096, cfg=cfg)
    dwells = [p for p in frame.points if p.dwell and p.lit()]
    corners = {(1000, 1000), (3000, 1000), (3000, 3000), (1000, 3000)}
    dwell_positions = {(p.x, p.y) for p in dwells}
    assert corners <= dwell_positions


def test_a_straight_line_gets_no_corner_dwell():
    straight = LaserPath(np.array([[500, 2000], [1500, 2000], [2500, 2000]], dtype=float))
    frame, _ = build_frame([straight], budget=4096)
    interior = [p for p in frame.points if p.dwell and p.lit() and p.x == 1500]
    assert interior == []


def test_path_has_blanked_anchor_dwell_at_entry():
    """The beam must sit dark on the first vertex while the galvos settle."""
    cfg = OptimiserConfig(dwell_blank_start=8)
    path = LaserPath(SQUARE, closed=True)
    # Ordering rotates a closed contour to begin at the vertex nearest the beam,
    # so the entry point is whatever ordering chose -- not necessarily SQUARE[0].
    entry_pt = order_paths([path])[0].points[0]
    entry = (int(entry_pt[0]), int(entry_pt[1]))

    frame, _ = build_frame([path], budget=4096, cfg=cfg)
    first_lit = next(i for i, p in enumerate(frame.points) if p.lit())
    # The settling dwell must all happen *before* the beam comes on. (On a closed
    # contour the exit dwell shares these coordinates, so only count the ones
    # ahead of first light.)
    anchors = [
        p for p in frame.points[:first_lit]
        if p.blank and p.dwell and (p.x, p.y) == entry
    ]
    assert len(anchors) >= cfg.dwell_blank_start


def test_ordering_reduces_blanked_travel():
    """Three clusters given in a deliberately awful order."""
    paths = [
        _line(4000, 4000, 4050, 4000),
        _line(100, 100, 150, 100),
        _line(3900, 3900, 3950, 3900),
        _line(200, 200, 250, 200),
    ]
    before = travel_distance(paths)
    after = travel_distance(order_paths(paths))
    assert after < before


def test_ordering_reverses_open_paths_when_the_tail_is_nearer():
    path = LaserPath(np.array([[4000, 4000], [100, 100]], dtype=float))
    ordered = order_paths([path], start=(0, 0))
    assert tuple(ordered[0].points[0]) == (100.0, 100.0)


def test_ordering_rotates_closed_paths_to_the_nearest_vertex():
    ordered = order_paths([LaserPath(SQUARE, closed=True)], start=(3100, 3100))
    assert tuple(ordered[0].points[0]) == (3000.0, 3000.0)


def test_degenerate_single_point_path_is_not_emitted():
    """A one-point contour would park the beam. It must be dropped, not drawn."""
    frame, _ = build_frame([LaserPath(np.array([[2000, 2000]], dtype=float))], budget=1000)
    assert len(frame) == 0


def test_empty_input_produces_an_empty_frame():
    frame, stats = build_frame([], budget=1000)
    assert len(frame) == 0
    assert stats["kept"] == 0


def test_all_points_land_inside_the_dac_field():
    paths = [LaserPath(SQUARE, closed=True), _line(0, 0, 4095, 4095)]
    frame, _ = build_frame(paths, budget=4096)
    assert all(0 <= p.x <= 4095 and 0 <= p.y <= 4095 for p in frame.points)
