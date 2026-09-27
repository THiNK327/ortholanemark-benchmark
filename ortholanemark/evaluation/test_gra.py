"""Unit tests for the GRA headline metric (plan §4.4).

GRA = operational region IoU, but the image scores 0 unless BOTH per-side
existence flags are exactly correct (false negative OR false positive
zeros it). Contrast with strict_operational_region_iou, which only
penalises false negatives (abstention).
"""

from __future__ import annotations

import numpy as np

from ortholanemark.data.lane_gt import LaneGeometricTruth
from ortholanemark.evaluation.metrics import (
    gated_region_accuracy,
    misassigned_area_fraction,
    within_sigma_fractions,
    per_image_boundary_errors_full,
    MM_PER_PX_TRANSVERSE,
)


def _line_gt(slope_x: float, intercept: float):
    """A simple in-frame near-vertical GT line x = slope*y + intercept."""
    # numpy-polyfit order: [slope, intercept] for degree-1 x = a*y + b.
    return LaneGeometricTruth(True, np.array([slope_x, intercept]), 1,
                              (0, 2499), [])


def test_gra_both_correct_equals_op_iou():
    iou = 0.93
    g = _line_gt(0.0, 200.0)
    out = gated_region_accuracy(iou, True, True, g, _line_gt(0.0, 800.0))
    assert out == 0.93


def test_gra_false_negative_zeros():
    """Pred misses a present side -> GRA = 0."""
    g = _line_gt(0.0, 200.0)
    out = gated_region_accuracy(0.93, False, True, g, _line_gt(0.0, 800.0))
    assert out == 0.0


def test_gra_false_positive_zeros():
    """Pred invents a marking on an absent side -> GRA = 0 (the property
    that distinguishes GRA from the FN-only strict metric)."""
    absent = LaneGeometricTruth(False, np.zeros(1), 0, None, [])
    present = _line_gt(0.0, 800.0)
    out = gated_region_accuracy(0.99, True, True, absent, present)
    assert out == 0.0


def test_gra_both_absent_correct_keeps_iou():
    """none/none correctly predicted -> existence correct -> keep IoU."""
    absent = LaneGeometricTruth(False, np.zeros(1), 0, None, [])
    out = gated_region_accuracy(1.0, False, False, absent, absent)
    assert out == 1.0


def test_per_image_driver_includes_gra_and_phantom_behaviour():
    """End-to-end through per_image_boundary_errors_full: a near-edge
    phantom keeps high op_iou and strict_FN but GRA must be 0."""
    H, W = 100, 200
    # GT: right side present near x=150, left side ABSENT.
    gt_R = _line_gt(0.0, 150.0)
    gt_L = LaneGeometricTruth(False, np.zeros(1), 0, None, [])
    # Pred: correct right curve, plus a phantom LEFT curve hugging x=2
    # (near the edge -> barely changes region).
    pred_R = np.full(H, 150.0)
    pred_L = np.full(H, 2.0)
    rec = per_image_boundary_errors_full(
        pred_xs_L=pred_L, pred_exists_L=True,
        pred_xs_R=pred_R, pred_exists_R=True,
        gt_L=gt_L, gt_R=gt_R, H=H, W=W)
    assert 'gated_region_accuracy' in rec
    # Phantom on the left (GT absent) -> existence wrong -> GRA zeroed.
    assert rec['gated_region_accuracy'] == 0.0
    # ...while ordinary op_iou stays high (phantom hugs the edge).
    assert rec['operational_iou'] > 0.8
    # MAF + within-σ are populated in the per-image record.
    assert 'misassigned_area_fraction' in rec
    assert 'within_50mm' in rec


def test_maf_perfect_prediction_is_zero():
    """Identical pred and GT regions → symmetric difference 0 → MAF 0."""
    H, W = 50, 200
    g = _line_gt(0.0, 150.0)
    pred = np.full(H, 150.0)
    maf = misassigned_area_fraction(
        pred_xs_L=np.full(H, 50.0), pred_exists_L=True,
        pred_xs_R=pred, pred_exists_R=True,
        gt_L=_line_gt(0.0, 50.0), gt_R=g, H=H, W=W)
    assert maf == 0.0


def test_within_sigma_perfect_and_offset():
    """Perfect pred → 100% within every σ; a +30 mm constant offset →
    0% within 10/25 mm but 100% within 50/100 mm."""
    H, W = 40, 300
    gt_R = _line_gt(0.0, 150.0)
    absent = LaneGeometricTruth(False, np.zeros(1), 0, None, [])
    perfect = within_sigma_fractions(
        None, False, np.full(H, 150.0), True, absent, gt_R, H, W)
    assert perfect[10.0] == 1.0 and perfect[100.0] == 1.0
    off_px = 30.0 / MM_PER_PX_TRANSVERSE + 0.5   # ~30 mm lateral offset
    shifted = within_sigma_fractions(
        None, False, np.full(H, 150.0 + off_px), True, absent, gt_R, H, W)
    assert shifted[10.0] == 0.0 and shifted[25.0] == 0.0
    assert shifted[50.0] == 1.0 and shifted[100.0] == 1.0
