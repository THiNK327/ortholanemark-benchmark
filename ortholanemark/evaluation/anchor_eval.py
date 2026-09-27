# vendored from physlanenet/utils/evaluation.py on 2026-05-30
# Part of the ortholanemark standalone project (no runtime imports from
# lane_boundary/ or physlanenet/). Refresh policy: manual re-vendor only —
# upstream changes do NOT auto-sync. Modify locally as needed.

"""
Evaluation metrics for lane marking position detection.

Includes smoothness metrics that catch zigzag failures invisible to F1.
Composite score ranks methods holistically.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class PositionMetrics:
    mae_pixels: float = float('inf')
    rmse_pixels: float = float('inf')
    max_error_pixels: float = float('inf')
    in_range_mae: float = float('inf')
    existence_correct: bool = False
    false_positive: bool = False
    false_negative: bool = False
    smoothness_d2: float = float('inf')
    max_step: float = float('inf')
    mean_step: float = float('inf')
    lateral_range: float = float('inf')


@dataclass
class AggregateMetrics:
    n_samples: int = 0
    mean_mae: float = 0.0
    mean_rmse: float = 0.0
    mean_max_error: float = 0.0
    median_mae: float = 0.0
    existence_accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    mean_inference_ms: float = 0.0
    all_maes: List[float] = field(default_factory=list)
    mean_smoothness_d2: float = 0.0
    mean_max_step: float = 0.0
    mean_lateral_range: float = 0.0
    composite_score: float = float('inf')


def compute_smoothness(x_positions):
    """Smoothness metrics for a sequence of x predictions.

    Returns: (smoothness_d2, max_step, mean_step, lateral_range)
    """
    if x_positions is None or len(x_positions) < 3:
        return float('inf'), float('inf'), float('inf'), float('inf')
    x = np.asarray(x_positions, dtype=np.float64)
    dx = np.abs(np.diff(x))
    d2x = np.abs(x[2:] - 2 * x[1:-1] + x[:-2])
    return (float(np.mean(d2x)), float(np.max(dx)),
            float(np.mean(dx)), float(np.max(x) - np.min(x)))


def compute_position_metrics(pred_x, gt_x, anchor_ys,
                              pred_exists, gt_exists, gt_y_range=None):
    m = PositionMetrics()
    m.existence_correct = (pred_exists == gt_exists)
    if gt_exists and not pred_exists:
        m.false_negative = True
        return m
    if not gt_exists and pred_exists:
        m.false_positive = True
        if pred_x is not None:
            m.smoothness_d2, m.max_step, m.mean_step, m.lateral_range = \
                compute_smoothness(pred_x)
        return m
    if not gt_exists and not pred_exists:
        m.mae_pixels = m.rmse_pixels = m.max_error_pixels = m.in_range_mae = 0.0
        m.smoothness_d2 = m.max_step = m.mean_step = m.lateral_range = 0.0
        return m
    if pred_x is None or gt_x is None:
        return m

    pred_arr = np.asarray(pred_x, dtype=np.float64)
    gt_arr = np.asarray(gt_x, dtype=np.float64)
    # Filter abstention sentinels: methods may write -1 (or any negative value)
    # at anchors where they choose not to predict. Treating those as predictions
    # at x=-1 silently inflates MAE for methods that selectively abstain
    # (e.g. lane_boundary) versus methods that always predict (e.g. PhysLaneNet).
    # F1 already captures per-image existence; MAE should only average over
    # anchors with an actual prediction.
    valid_anchors = pred_arr >= 0
    if not valid_anchors.any():
        return m   # no valid anchors -> leave MAE at inf (false negative below)

    errors = np.abs(pred_arr - gt_arr)
    errors_valid = errors[valid_anchors]
    m.mae_pixels = float(np.mean(errors_valid))
    m.rmse_pixels = float(np.sqrt(np.mean(errors_valid ** 2)))
    m.max_error_pixels = float(np.max(errors_valid))
    if gt_y_range is not None:
        in_range_mask = ((anchor_ys >= gt_y_range[0])
                         & (anchor_ys <= gt_y_range[1]) & valid_anchors)
        m.in_range_mae = (float(np.mean(errors[in_range_mask]))
                          if in_range_mask.sum() > 0 else m.mae_pixels)
    else:
        m.in_range_mae = m.mae_pixels
    m.smoothness_d2, m.max_step, m.mean_step, m.lateral_range = \
        compute_smoothness(pred_arr[valid_anchors])
    return m


def aggregate_metrics(metrics_list):
    agg = AggregateMetrics(n_samples=len(metrics_list))
    if not metrics_list:
        return agg

    n_fp = sum(m.false_positive for m in metrics_list)
    n_fn = sum(m.false_negative for m in metrics_list)
    n_tp = sum(m.existence_correct and 0 < m.mae_pixels < float('inf')
               for m in metrics_list)

    agg.existence_accuracy = sum(m.existence_correct for m in metrics_list) / agg.n_samples
    agg.precision = n_tp / (n_tp + n_fp) if (n_tp + n_fp) > 0 else 0
    agg.recall = n_tp / (n_tp + n_fn) if (n_tp + n_fn) > 0 else 0
    if agg.precision + agg.recall > 0:
        agg.f1 = 2 * agg.precision * agg.recall / (agg.precision + agg.recall)

    valid_mae = [m.mae_pixels for m in metrics_list if 0 < m.mae_pixels < float('inf')]
    if valid_mae:
        agg.mean_mae = float(np.mean(valid_mae))
        agg.median_mae = float(np.median(valid_mae))
        agg.all_maes = valid_mae
    valid_rmse = [m.rmse_pixels for m in metrics_list if 0 < m.rmse_pixels < float('inf')]
    if valid_rmse:
        agg.mean_rmse = float(np.mean(valid_rmse))
    valid_max = [m.max_error_pixels for m in metrics_list if 0 < m.max_error_pixels < float('inf')]
    if valid_max:
        agg.mean_max_error = float(np.mean(valid_max))

    # Smoothness (include 0.0 = perfect)
    for attr, src in [('mean_smoothness_d2', 'smoothness_d2'),
                       ('mean_max_step', 'max_step'),
                       ('mean_lateral_range', 'lateral_range')]:
        vals = [getattr(m, src) for m in metrics_list if getattr(m, src) < float('inf')]
        if vals:
            setattr(agg, attr, float(np.mean(vals)))

    # Composite score (lower = better)
    miss_pen = 100.0 * (1.0 - agg.recall) if agg.recall > 0 else 100.0
    fp_pen = 50.0 * (1.0 - agg.precision) if agg.precision > 0 else 50.0
    pos_term = agg.mean_mae if agg.mean_mae < float('inf') else 500.0
    smooth_term = agg.mean_smoothness_d2 * 2.0
    agg.composite_score = miss_pen + fp_pen + pos_term + smooth_term
    return agg


def gt_x_at_anchors(gt_lane, anchor_ys_img, ann_w, ann_h, img_w, img_h):
    """Evaluate GT polynomial at anchor rows, annotation→image coords."""
    if not gt_lane.get('has_marking') or gt_lane.get('polynomial_coefficients') is None:
        return None, None
    sx, sy = img_w / ann_w, img_h / ann_h
    c = gt_lane['polynomial_coefficients']
    d = gt_lane['polynomial_degree']
    ys_ann = anchor_ys_img / sy
    if d == 1:
        xs_ann = c[0] * ys_ann + c[1]
    elif d == 2:
        xs_ann = c[0] * ys_ann**2 + c[1] * ys_ann + c[2]
    else:
        return None, None
    yr = gt_lane.get('y_range', {})
    y_min, y_max = yr.get('min'), yr.get('max')
    gt_yr = (y_min * sy, y_max * sy) if y_min is not None else None
    return xs_ann * sx, gt_yr


def evaluate_method_on_dataset(predictions, ground_truths, n_anchors=64,
                                image_w=832, image_h=2000):
    anchor_ys = np.linspace(0, image_h - 1, n_anchors)
    left_list, right_list = [], []
    for pred, gt_ann in zip(predictions, ground_truths):
        ann_w = gt_ann.get('image_size', {}).get('width', image_w)
        ann_h = gt_ann.get('image_size', {}).get('height', image_h)
        for side, lk in [('left', 'left_lane'), ('right', 'right_lane')]:
            gt_x, gt_yr = gt_x_at_anchors(gt_ann[lk], anchor_ys, ann_w, ann_h,
                                            image_w, image_h)
            ps = pred.get(side, {})
            m = compute_position_metrics(
                ps.get('x_positions'), gt_x, anchor_ys,
                ps.get('exists', False), gt_ann[lk]['has_marking'], gt_yr)
            (left_list if side == 'left' else right_list).append(m)
    return {'left': aggregate_metrics(left_list),
            'right': aggregate_metrics(right_list),
            'combined': aggregate_metrics(left_list + right_list)}


def print_comparison_table(results: Dict[str, Dict]):
    """Print table sorted by composite score. Smoothness columns included."""
    sorted_m = sorted(results.keys(),
                       key=lambda m: results[m]['combined'].composite_score)
    print("\n" + "=" * 120)
    print(f"{'#':<3} {'Method':<22} {'MAE(px)':<10} {'Smooth_d2':<11} "
          f"{'MaxStep':<10} {'LatRange':<10} "
          f"{'Prec':<7} {'Rec':<7} {'F1':<7} {'Score(L)':<10} {'ms':<7}")
    print("=" * 120)
    for rank, method in enumerate(sorted_m, 1):
        c = results[method]['combined']
        fmt = lambda v: f"{v:.1f}" if v < float('inf') else "—"
        print(f"{rank:<3} {method:<22} {fmt(c.mean_mae):<10} "
              f"{fmt(c.mean_smoothness_d2):<11} {fmt(c.mean_max_step):<10} "
              f"{fmt(c.mean_lateral_range):<10} "
              f"{c.precision:<7.3f} {c.recall:<7.3f} {c.f1:<7.3f} "
              f"{fmt(c.composite_score):<10} {c.mean_inference_ms:<7.1f}")
    print("=" * 120)
    print("\n  Score(L) = composite (lower=better): miss_penalty + fp_penalty + MAE + 2x Smooth_d2")
