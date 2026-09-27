"""Unit tests for `LaneGeometricTruth` (plan §7.1).

Standalone — no imports from lane_boundary or physlanenet.
"""

from __future__ import annotations

import numpy as np

from lcms_lane_benchmark.data.lane_gt import LaneGeometricTruth


# ----------------------------------------------------------------------
# from_lane_dict / from_clean_gt_dict
# ----------------------------------------------------------------------
def test_from_lane_dict_round_trip():
    """Build a raw-shaped lane dict, run from_lane_dict, confirm fields."""
    raw = {
        'has_marking': True,
        'polynomial_coefficients': [0.0001, 0.02, 150.0],
        'polynomial_degree': 2,
        'clicked_points': [[150, 100], [180, 1500], [210, 2400]],
        'y_range': {'min': 100, 'max': 2400},
        'pixel_positions': [],  # irrelevant for the loader
    }
    gt = LaneGeometricTruth.from_lane_dict(raw)
    assert gt is not None
    assert gt.has_marking is True
    assert gt.poly_degree == 2
    assert np.allclose(gt.poly_coefs, [0.0001, 0.02, 150.0])
    assert gt.clicked_y_range == (100, 2400)
    assert gt.clicked_points == [(150, 100), (180, 1500), (210, 2400)]


def test_from_lane_dict_no_marking():
    """has_marking=False with null fields should return a placeholder GT."""
    raw = {
        'has_marking': False,
        'polynomial_coefficients': None,
        'polynomial_degree': None,
        'clicked_points': [],
        'y_range': {'min': None, 'max': None},
        'pixel_positions': [],
    }
    gt = LaneGeometricTruth.from_lane_dict(raw)
    assert gt is not None
    assert gt.has_marking is False
    assert gt.poly_degree == 0
    assert gt.clicked_y_range is None
    assert gt.clicked_points == []


def test_from_clean_gt_dict_uses_renamed_field():
    """Sidecar uses `clicked_y_range`; loader must accept it."""
    sidecar = {
        'has_marking': True,
        'polynomial_coefficients': [0.05, 200.0],
        'polynomial_degree': 1,
        'clicked_points': [[210, 200], [260, 1200]],
        'clicked_y_range': {'min': 200, 'max': 1200},
        # x_full and in_frame_mask are ignored at load time.
        'x_full': [], 'in_frame_mask': [],
        'off_frame_top_rows': 0, 'off_frame_bottom_rows': 0,
    }
    gt = LaneGeometricTruth.from_clean_gt_dict(sidecar)
    assert gt is not None
    assert gt.poly_degree == 1
    assert gt.clicked_y_range == (200, 1200)


# ----------------------------------------------------------------------
# eval_x_at: clamping vs NaN regimes
# ----------------------------------------------------------------------
def test_eval_x_at_clamp_false_returns_nan_off_frame():
    """A degree-2 polynomial that extrapolates beyond [0, W) gives NaN."""
    # x = -1e-3 * y^2 + 0 * y + 100 → at y=0: x=100. At y=600: x = -260 (off-left).
    gt = LaneGeometricTruth(True, np.array([-1e-3, 0.0, 100.0]), 2, None, [])
    ys = np.arange(0, 1000, 100, dtype=np.float64)
    xs = gt.eval_x_at(ys, W=500, clamp_to_frame=False)
    # y=0 → x=100 (in-frame). y=600 → x=-260 (off-frame → NaN).
    assert np.isfinite(xs[0])
    assert np.isnan(xs[6])
    # clamp=True converts off-frame to 0.0 (left edge).
    xs_clamp = gt.eval_x_at(ys, W=500, clamp_to_frame=True)
    assert xs_clamp[6] == 0.0


def test_eval_x_at_no_marking_clamp_returns_zeros():
    gt = LaneGeometricTruth(False, np.zeros(1), 0, None, [])
    xs_nan = gt.eval_x_at(np.arange(10), W=100, clamp_to_frame=False)
    xs_clamp = gt.eval_x_at(np.arange(10), W=100, clamp_to_frame=True)
    assert np.all(np.isnan(xs_nan))
    assert np.all(xs_clamp == 0.0)


# ----------------------------------------------------------------------
# in_frame_mask
# ----------------------------------------------------------------------
def test_in_frame_mask_matches_manual_check():
    """A line that exits the frame at y > 800 should have False after row 800."""
    # x = 0.5 * y + 100 → at y=800: x=500. At y=801: x=500.5 (out for W=500).
    gt = LaneGeometricTruth(True, np.array([0.5, 100.0]), 1, None, [])
    mask = gt.in_frame_mask(H=1000, W=500)
    # x at row y is 0.5*y + 100; in-frame: 0 <= x <= 499  →  y <= 798.
    expected_true = np.arange(1000) <= 798
    assert np.array_equal(mask, expected_true)


def test_in_frame_mask_no_marking_all_false():
    gt = LaneGeometricTruth(False, np.zeros(1), 0, None, [])
    assert not gt.in_frame_mask(H=100, W=50).any()


# ----------------------------------------------------------------------
# to_crop_view: closed-form coordinate substitution (degree 1 + 2)
# ----------------------------------------------------------------------
def test_to_crop_view_degree1_round_trip():
    """For degree 1: evaluating the crop-local poly should equal the
    original polynomial evaluated at corresponding full-image y."""
    # Original: x_full = 0.02 * y_full + 150.
    gt = LaneGeometricTruth(True, np.array([0.02, 150.0]), 1, None, [])
    crop_y0 = 500
    crop_x0_padded = 50
    pad_left = 128
    crop_h, crop_w = 800, 600

    crop_gt = gt.to_crop_view(crop_y0, crop_x0_padded, pad_left, crop_h, crop_w)

    # Evaluate both forms at the same row in crop coords.
    y_crop = np.array([0.0, 200.0, 500.0, 799.0])
    x_crop = np.polyval(crop_gt.poly_coefs, y_crop)
    # Manual closed-form check.
    y_full = y_crop + crop_y0
    x_full = np.polyval(gt.poly_coefs, y_full)
    x_expected = x_full + pad_left - crop_x0_padded
    assert np.allclose(x_crop, x_expected, atol=1e-6)


def test_to_crop_view_degree2_round_trip():
    """Degree 2 should also satisfy x_crop(y_crop) = x_full(y_full) + delta."""
    # Original: x_full = 1e-4 * y_full^2 + 0.02 * y_full + 150.
    gt = LaneGeometricTruth(True, np.array([1e-4, 0.02, 150.0]), 2, None, [])
    crop_y0 = 800
    crop_x0_padded = 70
    pad_left = 64
    crop_h, crop_w = 1024, 1024

    crop_gt = gt.to_crop_view(crop_y0, crop_x0_padded, pad_left, crop_h, crop_w)

    y_crop = np.linspace(0, crop_h - 1, 10)
    x_crop = np.polyval(crop_gt.poly_coefs, y_crop)
    y_full = y_crop + crop_y0
    x_full = np.polyval(gt.poly_coefs, y_full)
    x_expected = x_full + pad_left - crop_x0_padded
    assert np.allclose(x_crop, x_expected, atol=1e-3)


def test_to_crop_view_no_marking_passes_through():
    gt = LaneGeometricTruth(False, np.zeros(1), 0, None, [])
    crop_gt = gt.to_crop_view(0, 0, 0, 100, 100)
    assert crop_gt.has_marking is False
    assert crop_gt.poly_degree == 0


if __name__ == '__main__':
    test_from_lane_dict_round_trip()
    test_from_lane_dict_no_marking()
    test_from_clean_gt_dict_uses_renamed_field()
    test_eval_x_at_clamp_false_returns_nan_off_frame()
    test_eval_x_at_no_marking_clamp_returns_zeros()
    test_in_frame_mask_matches_manual_check()
    test_in_frame_mask_no_marking_all_false()
    test_to_crop_view_degree1_round_trip()
    test_to_crop_view_degree2_round_trip()
    test_to_crop_view_no_marking_passes_through()
    print('All LaneGeometricTruth tests PASSED')
