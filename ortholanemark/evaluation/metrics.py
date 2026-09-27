"""Clean evaluation metrics for the OrthoLaneMark benchmark.

Replaces `lane_boundary.evaluation.metrics`. Two crucial differences:

1. The polynomial extends across the full image; we never gate on
   `clicked_y_range`. The only mask we apply is the operational
   in-frame mask (polynomial value lies inside [0, W)).

2. Operational closure (§5.2): off-frame x values are CLAMPED to the
   image edge for region/area/width metrics (matches the inference
   rasterizer); and SKIPPED for boundary MAE on GT (matches the
   training-target convention).

Top-level entrypoint: `per_image_boundary_errors_full` runs every
metric per image, returning a dict the train/eval scripts aggregate.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ortholanemark.data.lane_gt import LaneGeometricTruth

# Transverse (lateral / x) resolution: 1 px = 4 mm (sensor ground
# resolution, as stated in the paper). Boundary errors are lateral, so
# they convert with this factor. Within-σ thresholds are in mm.
MM_PER_PX_TRANSVERSE = 4.0
SIGMA_MM = (10.0, 25.0, 50.0, 100.0)


# ----------------------------------------------------------------------
# Per-side polynomial sources
# ----------------------------------------------------------------------
def _eval_full(gt: Optional[LaneGeometricTruth], H: int, W: int,
               clamp: bool) -> np.ndarray:
    """Evaluate per-row x for the full image. Returns NaN per row when
    `gt` is None or has_marking=False AND clamp=False; returns sentinel
    image-edge values when clamp=True and the side is absent."""
    if gt is None or not gt.has_marking:
        if clamp:
            return np.zeros(H, dtype=np.float64)   # caller chooses edge
        return np.full(H, np.nan, dtype=np.float64)
    ys = np.arange(H, dtype=np.float64)
    return gt.eval_x_at(ys, W, clamp_to_frame=clamp)


# ----------------------------------------------------------------------
# Boundary MAE on GT (training-target convention: skip off-frame rows)
# ----------------------------------------------------------------------
def boundary_mae_on_gt(pred_xs: np.ndarray,
                       gt: Optional[LaneGeometricTruth],
                       H: int, W: int) -> Optional[float]:
    """Mean |pred_x − gt_x| over rows where pred is finite AND GT poly
    is in-frame. Returns None if no qualifying rows exist (the side is
    absent or entirely off-frame).
    No abstention penalty — this is pure boundary localization."""
    if gt is None or not gt.has_marking:
        return None
    gt_x = _eval_full(gt, H, W, clamp=False)
    if pred_xs is None:
        return None
    pred = np.asarray(pred_xs, dtype=np.float64)
    if pred.shape[0] != H:
        return None
    valid = np.isfinite(pred) & np.isfinite(gt_x)
    if not valid.any():
        return None
    return float(np.mean(np.abs(pred[valid] - gt_x[valid])))


# ----------------------------------------------------------------------
# Operational closure helpers
# ----------------------------------------------------------------------
def _operational_xs(
    pred_xs_L: Optional[np.ndarray], pred_exists_L: bool,
    pred_xs_R: Optional[np.ndarray], pred_exists_R: bool,
    H: int, W: int,
):
    """Build per-row operational left/right edges from predictions.
    Missing/absent side → image edge (0 for left, W-1 for right).
    `pred_xs_*` may contain NaN at rows where the pred poly is off-frame
    — those are clamped to the edge here (operational eval semantics)."""
    op_L = np.zeros(H, dtype=np.float64)
    op_R = np.full(H, W - 1, dtype=np.float64)
    if pred_exists_L and pred_xs_L is not None:
        x = np.asarray(pred_xs_L, dtype=np.float64)
        x = np.where(np.isfinite(x), x, 0.0)
        op_L = np.clip(x, 0.0, float(W - 1))
    if pred_exists_R and pred_xs_R is not None:
        x = np.asarray(pred_xs_R, dtype=np.float64)
        x = np.where(np.isfinite(x), x, float(W - 1))
        op_R = np.clip(x, 0.0, float(W - 1))
    return op_L, op_R


def _operational_xs_from_gt(gt_L: Optional[LaneGeometricTruth],
                            gt_R: Optional[LaneGeometricTruth],
                            H: int, W: int):
    """Same shape as `_operational_xs` but for GT — image-edge fallback
    when a side is absent. Existing sides are clamped to [0, W-1]
    (operational closure)."""
    if gt_L is not None and gt_L.has_marking:
        gt_x_L = gt_L.eval_x_at(np.arange(H, dtype=np.float64), W,
                                clamp_to_frame=True)
    else:
        gt_x_L = np.zeros(H, dtype=np.float64)
    if gt_R is not None and gt_R.has_marking:
        gt_x_R = gt_R.eval_x_at(np.arange(H, dtype=np.float64), W,
                                clamp_to_frame=True)
    else:
        gt_x_R = np.full(H, W - 1, dtype=np.float64)
    return gt_x_L, gt_x_R


def _rasterize_region(x_L: np.ndarray, x_R: np.ndarray,
                      H: int, W: int) -> np.ndarray:
    """Mask=1 where x_L[r] <= col <= x_R[r]. Both inputs already in [0,W-1].

    Vectorized — replaces a per-row Python loop that dominated val-time
    metric cost. (H, W) bool tensor; for 2504x1040 that's ~2.6 MB.
    """
    cols = np.arange(W, dtype=np.float64)
    lo = np.ceil(x_L).astype(np.float64)[:, None]   # (H, 1)
    hi = np.floor(x_R).astype(np.float64)[:, None]  # (H, 1)
    mask = (cols[None, :] >= lo) & (cols[None, :] <= hi)
    return mask


# ----------------------------------------------------------------------
# Operational metrics
# ----------------------------------------------------------------------
def operational_region_iou(
    pred_xs_L, pred_exists_L,
    pred_xs_R, pred_exists_R,
    gt_L, gt_R, H, W,
) -> float:
    op_pL, op_pR = _operational_xs(pred_xs_L, pred_exists_L,
                                   pred_xs_R, pred_exists_R, H, W)
    op_gL, op_gR = _operational_xs_from_gt(gt_L, gt_R, H, W)
    pred_mask = _rasterize_region(op_pL, op_pR, H, W)
    gt_mask = _rasterize_region(op_gL, op_gR, H, W)
    union = pred_mask | gt_mask
    if not union.any():
        return 1.0
    inter = pred_mask & gt_mask
    return float(inter.sum()) / float(union.sum())


def strict_operational_region_iou(
    pred_xs_L, pred_exists_L,
    pred_xs_R, pred_exists_R,
    gt_L, gt_R, H, W,
) -> float:
    """Strict version of `operational_region_iou` that does NOT reward
    abstention.

    Behaviour vs the standard version:
      - If pred misses a side that GT has → pred region for those rows
        is treated as EMPTY (no positive prediction). This kills the
        "abstain → image-edge fallback → trivial IoU because the lane
        dominates the image" loophole.
      - If both pred AND GT abstain on a side → image-edge fallback still
        applies (joint correct abstention).
      - If pred predicts a side that GT lacks → pred uses its predicted
        curve (counts as a false-positive; the standard IoU computation
        naturally penalises it).

    The pred-region rows where this rule invalidates the prediction
    contribute 0 to intersection and `gt_row_width` to union, so
    abstention on a present GT side hurts IoU exactly as it should.
    """
    op_pL, op_pR = _operational_xs(pred_xs_L, pred_exists_L,
                                   pred_xs_R, pred_exists_R, H, W)
    op_gL, op_gR = _operational_xs_from_gt(gt_L, gt_R, H, W)

    # Per-row "is the predicted region valid?"
    # Strict rule (no-abstention-reward, by design): a SINGLE missed present
    # side invalidates the ENTIRE predicted region (all rows below), not just
    # that side's rows — so abstaining on a present marking cannot game IoU.
    pred_valid = np.ones(H, dtype=bool)
    gt_has_L = bool(gt_L is not None and gt_L.has_marking)
    gt_has_R = bool(gt_R is not None and gt_R.has_marking)
    if (not pred_exists_L) and gt_has_L:
        pred_valid[:] = False
    if (not pred_exists_R) and gt_has_R:
        pred_valid[:] = False

    pred_mask = _rasterize_region(op_pL, op_pR, H, W)
    gt_mask = _rasterize_region(op_gL, op_gR, H, W)
    # Zero out pred_mask on invalid rows — they contribute nothing.
    if not pred_valid.all():
        pred_mask = pred_mask & pred_valid[:, None]

    union = pred_mask | gt_mask
    if not union.any():
        return 1.0
    inter = pred_mask & gt_mask
    return float(inter.sum()) / float(union.sum())


def gated_region_accuracy(
    operational_iou: float,
    pred_exists_L, pred_exists_R,
    gt_L, gt_R,
) -> float:
    """GRA (plan §4.4): the image scores its operational region IoU ONLY
    if BOTH per-side existence flags are exactly correct; otherwise 0.

    This is the strict-from-both-directions headline gate. Unlike
    `strict_operational_region_iou` (which only empties pred rows on a
    missed present side, i.e. punishes false NEGATIVES), GRA also zeros
    the image on a false POSITIVE — a phantom marking on a GT-absent
    side. Phantom markings that hug the image edge barely dent ordinary
    IoU (the near-edge degeneracy), so without this gate a trigger-happy
    model can game the leaderboard; GRA closes that loophole.
    """
    gt_has_L = bool(gt_L is not None and gt_L.has_marking)
    gt_has_R = bool(gt_R is not None and gt_R.has_marking)
    exist_correct = ((bool(pred_exists_L) == gt_has_L) and
                     (bool(pred_exists_R) == gt_has_R))
    return float(operational_iou) if exist_correct else 0.0


def operational_area_error_pct(
    pred_xs_L, pred_exists_L,
    pred_xs_R, pred_exists_R,
    gt_L, gt_R, H, W,
) -> float:
    """|pred_area − gt_area| / max(gt_area, 100 px²)."""
    op_pL, op_pR = _operational_xs(pred_xs_L, pred_exists_L,
                                   pred_xs_R, pred_exists_R, H, W)
    op_gL, op_gR = _operational_xs_from_gt(gt_L, gt_R, H, W)
    pred_area = float((op_pR - op_pL).clip(min=0.0).sum())
    gt_area = float((op_gR - op_gL).clip(min=0.0).sum())
    denom = max(gt_area, 100.0)
    return abs(pred_area - gt_area) / denom


def operational_width_mae(
    pred_xs_L, pred_exists_L,
    pred_xs_R, pred_exists_R,
    gt_L, gt_R, H, W,
) -> float:
    op_pL, op_pR = _operational_xs(pred_xs_L, pred_exists_L,
                                   pred_xs_R, pred_exists_R, H, W)
    op_gL, op_gR = _operational_xs_from_gt(gt_L, gt_R, H, W)
    pred_w = (op_pR - op_pL).clip(min=0.0)
    gt_w = (op_gR - op_gL).clip(min=0.0)
    return float(np.mean(np.abs(pred_w - gt_w)))


def misassigned_area_fraction(
    pred_xs_L, pred_exists_L,
    pred_xs_R, pred_exists_R,
    gt_L, gt_R, H, W,
) -> float:
    """MAF = |pred_region Δ gt_region| / (W·H) — symmetric-difference area
    over total image area, operational closure (default-to-edge) on both.

    Downstream reading: for distresses uniformly distributed in the image,
    MAF is the probability a distress pixel is mis-assigned in/out of the
    travel lane — the expected distress-localization error the lane model
    induces. Lower is better."""
    op_pL, op_pR = _operational_xs(pred_xs_L, pred_exists_L,
                                   pred_xs_R, pred_exists_R, H, W)
    op_gL, op_gR = _operational_xs_from_gt(gt_L, gt_R, H, W)
    pred_mask = _rasterize_region(op_pL, op_pR, H, W)
    gt_mask = _rasterize_region(op_gL, op_gR, H, W)
    sym = np.logical_xor(pred_mask, gt_mask)
    return float(sym.sum()) / float(H * W)


def _per_row_abs_err_px(pred_xs, gt: Optional[LaneGeometricTruth],
                        H: int, W: int) -> np.ndarray:
    """|pred − gt| (px) over rows where pred is finite AND GT poly in-frame.
    Empty array if the side is absent / no qualifying rows."""
    if gt is None or not gt.has_marking or pred_xs is None:
        return np.empty(0, dtype=np.float64)
    gt_x = _eval_full(gt, H, W, clamp=False)
    pred = np.asarray(pred_xs, dtype=np.float64)
    if pred.shape[0] != H:
        return np.empty(0, dtype=np.float64)
    valid = np.isfinite(pred) & np.isfinite(gt_x)
    if not valid.any():
        return np.empty(0, dtype=np.float64)
    return np.abs(pred[valid] - gt_x[valid])


def within_sigma_fractions(
    pred_xs_L, pred_exists_L,
    pred_xs_R, pred_exists_R,
    gt_L, gt_R, H, W,
) -> dict:
    """Layer-B accuracy: fraction of rows with lateral error < σ mm, pooled
    over the two sides, counting only TRUE-POSITIVE existence sides (GT
    present AND pred_exists). Returns {σ_mm: frac or None}. None when no
    qualifying rows (no TP side in the image)."""
    errs = []
    if pred_exists_L:
        errs.append(_per_row_abs_err_px(pred_xs_L, gt_L, H, W))
    if pred_exists_R:
        errs.append(_per_row_abs_err_px(pred_xs_R, gt_R, H, W))
    e = (np.concatenate(errs) if errs else np.empty(0, dtype=np.float64))
    if e.size == 0:
        return {s: None for s in SIGMA_MM}
    e_mm = e * MM_PER_PX_TRANSVERSE
    return {s: float((e_mm < s).mean()) for s in SIGMA_MM}


# ----------------------------------------------------------------------
# Existence F1 (macro)
# ----------------------------------------------------------------------
def existence_f1_accumulator():
    """Returns a small state dict the caller updates per image."""
    return {
        'L': {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0},
        'R': {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0},
    }


def existence_f1_update(accum: dict, side: str,
                        pred_exists: bool, gt_exists: bool) -> None:
    cell = accum[side]
    if pred_exists and gt_exists:
        cell['tp'] += 1
    elif pred_exists and not gt_exists:
        cell['fp'] += 1
    elif (not pred_exists) and gt_exists:
        cell['fn'] += 1
    else:
        cell['tn'] += 1


def existence_f1_finalize(accum: dict) -> dict:
    out = {}
    f1_sides = []
    for side in ('L', 'R'):
        c = accum[side]
        prec_d = c['tp'] + c['fp']
        rec_d = c['tp'] + c['fn']
        prec = c['tp'] / prec_d if prec_d > 0 else 1.0
        rec = c['tp'] / rec_d if rec_d > 0 else 1.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        out[side] = {'precision': prec, 'recall': rec, 'f1': f1, **c}
        f1_sides.append(f1)
    out['macro_f1'] = float(np.mean(f1_sides))
    return out


# ----------------------------------------------------------------------
# Top-level per-image driver
# ----------------------------------------------------------------------
def per_image_boundary_errors_full(
    pred_xs_L, pred_exists_L,
    pred_xs_R, pred_exists_R,
    gt_L: Optional[LaneGeometricTruth],
    gt_R: Optional[LaneGeometricTruth],
    H: int, W: int,
) -> dict:
    """Compute all per-image clean metrics in one pass.

    `pred_xs_L/R` are per-row pred x positions of length H (NaN allowed
    where the pred polynomial extrapolates off-frame). `pred_exists_*`
    are scalar booleans (the model's per-side existence call).
    """
    # True-positive convention: localization is reported only for sides
    # the model detected AND GT has present, same gate as within-σ.
    mae_L = (boundary_mae_on_gt(pred_xs_L, gt_L, H, W)
             if pred_exists_L else None)
    mae_R = (boundary_mae_on_gt(pred_xs_R, gt_R, H, W)
             if pred_exists_R else None)

    iou = operational_region_iou(pred_xs_L, pred_exists_L,
                                 pred_xs_R, pred_exists_R,
                                 gt_L, gt_R, H, W)
    iou_strict = strict_operational_region_iou(
        pred_xs_L, pred_exists_L,
        pred_xs_R, pred_exists_R,
        gt_L, gt_R, H, W)
    gra = gated_region_accuracy(
        iou, pred_exists_L, pred_exists_R, gt_L, gt_R)
    area_err = operational_area_error_pct(pred_xs_L, pred_exists_L,
                                          pred_xs_R, pred_exists_R,
                                          gt_L, gt_R, H, W)
    w_mae = operational_width_mae(pred_xs_L, pred_exists_L,
                                  pred_xs_R, pred_exists_R,
                                  gt_L, gt_R, H, W)
    maf = misassigned_area_fraction(pred_xs_L, pred_exists_L,
                                    pred_xs_R, pred_exists_R,
                                    gt_L, gt_R, H, W)
    wsig = within_sigma_fractions(pred_xs_L, pred_exists_L,
                                  pred_xs_R, pred_exists_R,
                                  gt_L, gt_R, H, W)

    return {
        'boundary_mae_on_gt_left': mae_L,
        'boundary_mae_on_gt_right': mae_R,
        'boundary_mae_on_gt_macro': (
            float(np.nanmean([m for m in (mae_L, mae_R) if m is not None]))
            if (mae_L is not None or mae_R is not None) else None
        ),
        'operational_iou': iou,
        'strict_operational_iou': iou_strict,
        'gated_region_accuracy': gra,
        'operational_area_error_pct': area_err,
        'operational_width_mae': w_mae,
        'misassigned_area_fraction': maf,
        'within_10mm': wsig[10.0],
        'within_25mm': wsig[25.0],
        'within_50mm': wsig[50.0],
        'within_100mm': wsig[100.0],
    }


def aggregate_per_image(per_image: list) -> dict:
    """Aggregate a list of `per_image_boundary_errors_full` dicts into
    dataset-level summaries (means over per-image values, NaNs ignored).
    """
    def _mean(key):
        vals = [r[key] for r in per_image if r.get(key) is not None]
        return float(np.mean(vals)) if vals else None

    return {
        'boundary_mae_on_gt_left': _mean('boundary_mae_on_gt_left'),
        'boundary_mae_on_gt_right': _mean('boundary_mae_on_gt_right'),
        'boundary_mae_on_gt_macro': _mean('boundary_mae_on_gt_macro'),
        'operational_iou': _mean('operational_iou'),
        'strict_operational_iou': _mean('strict_operational_iou'),
        'gated_region_accuracy': _mean('gated_region_accuracy'),
        'operational_area_error_pct': _mean('operational_area_error_pct'),
        'operational_width_mae': _mean('operational_width_mae'),
        'misassigned_area_fraction': _mean('misassigned_area_fraction'),
        'within_10mm': _mean('within_10mm'),
        'within_25mm': _mean('within_25mm'),
        'within_50mm': _mean('within_50mm'),
        'within_100mm': _mean('within_100mm'),
        'n_images': len(per_image),
    }
