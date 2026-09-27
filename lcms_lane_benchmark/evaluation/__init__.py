"""Clean evaluation module for the LCMS lane-boundary benchmark."""

from lcms_lane_benchmark.evaluation.metrics import (
    boundary_mae_on_gt,
    operational_region_iou,
    strict_operational_region_iou,
    gated_region_accuracy,
    operational_area_error_pct,
    operational_width_mae,
    existence_f1_accumulator,
    existence_f1_update,
    existence_f1_finalize,
    per_image_boundary_errors_full,
    aggregate_per_image,
)
from lcms_lane_benchmark.evaluation.predictions_io import (
    save_predictions_npz,
    load_predictions_npz,
)

__all__ = [
    'boundary_mae_on_gt',
    'operational_region_iou',
    'strict_operational_region_iou',
    'gated_region_accuracy',
    'operational_area_error_pct',
    'operational_width_mae',
    'existence_f1_accumulator',
    'existence_f1_update',
    'existence_f1_finalize',
    'per_image_boundary_errors_full',
    'aggregate_per_image',
    'save_predictions_npz',
    'load_predictions_npz',
]
