"""Image-to-projector mapping and the calibration transform."""

from __future__ import annotations

import numpy as np
import pytest

from webcam_ilda.geometry import Calibration, fit_transform, image_to_dac, polyline_length


def test_image_centre_maps_to_dac_centre():
    out = image_to_dac(np.array([[320.0, 240.0]]), 640, 480)
    assert tuple(out[0]) == (2048, 2048)


def test_output_always_within_dac_bounds():
    corners = np.array([[0, 0], [639, 0], [0, 479], [639, 479]], dtype=float)
    out = image_to_dac(corners, 640, 480, Calibration(scale_pct=100.0))
    assert out.min() >= 0
    assert out.max() <= 4095


def test_aspect_ratio_is_preserved():
    """A square drawn in a 16:9 frame must still be square on the wall."""
    square = np.array([[100, 100], [200, 100], [200, 200], [100, 200]], dtype=float)
    out = image_to_dac(square, 1920, 1080).astype(float)
    width = out[:, 0].max() - out[:, 0].min()
    height = out[:, 1].max() - out[:, 1].min()
    assert width == pytest.approx(height, abs=1.0)


def test_y_axis_is_inverted():
    """Image y grows downwards; DAC y grows upwards."""
    top = image_to_dac(np.array([[320.0, 100.0]]), 640, 480)[0]
    bottom = image_to_dac(np.array([[320.0, 380.0]]), 640, 480)[0]
    assert top[1] > bottom[1]


def test_flip_x_mirrors_about_the_centre():
    pt = np.array([[500.0, 240.0]])
    plain = image_to_dac(pt, 640, 480)[0]
    flipped = image_to_dac(pt, 640, 480, Calibration(flip_x=True))[0]
    assert plain[0] + flipped[0] == pytest.approx(2 * 2048, abs=1)
    assert plain[1] == pytest.approx(flipped[1], abs=1)


def test_flip_y_mirrors_about_the_centre():
    pt = np.array([[320.0, 100.0]])
    plain = image_to_dac(pt, 640, 480)[0]
    flipped = image_to_dac(pt, 640, 480, Calibration(flip_y=True))[0]
    assert plain[1] + flipped[1] == pytest.approx(2 * 2048, abs=1)


def test_rotate_180_equals_both_flips():
    pts = np.array([[100.0, 120.0], [500.0, 400.0]])
    rotated = image_to_dac(pts, 640, 480, Calibration(rotate=180))
    flipped = image_to_dac(pts, 640, 480, Calibration(flip_x=True, flip_y=True))
    assert np.allclose(rotated, flipped, atol=1)


def test_rotate_90_four_times_is_identity():
    pts = np.array([[100.0, 120.0], [500.0, 400.0]])
    base = image_to_dac(pts, 640, 480)
    assert np.allclose(image_to_dac(pts, 640, 480, Calibration(rotate=360)), base, atol=1)


def test_scale_pct_controls_projected_size():
    corners = np.array([[0, 0], [639, 479]], dtype=float)
    small = image_to_dac(corners, 640, 480, Calibration(scale_pct=25.0))
    large = image_to_dac(corners, 640, 480, Calibration(scale_pct=75.0))
    small_span = abs(small[1][0] - small[0][0])
    large_span = abs(large[1][0] - large[0][0])
    assert large_span > small_span * 2.5


def test_empty_input_returns_empty_array():
    out = image_to_dac(np.zeros((0, 2)), 640, 480)
    assert out.shape == (0, 2)


def test_fit_transform_rejects_degenerate_size():
    with pytest.raises(ValueError):
        fit_transform(0, 480)


def test_polyline_length_of_unit_square():
    square = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=float)
    assert polyline_length(square) == pytest.approx(30.0)
