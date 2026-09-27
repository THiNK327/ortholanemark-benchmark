"""Shared helper: compute mean `strict_operational_iou` over a val split,
with **batched** GPU inference.

Per-epoch GRA selects best_gra.pth; strict operational IoU is also recorded
for the historical best.pth checkpoint. Running inference one image at a time was the
dominant cost (~15 min/epoch for LaneATT at batch=1). Batching the
forward pass collapses that to ~2 min/epoch — same model, same val set,
same metric, just one CUDA kernel launch per batch instead of one per
image.

API
---
    predict_batch_fn(model, image_grays_list, **predict_kwargs) -> list[dict]
        Each element in image_grays_list is a numpy uint8 grayscale image
        of arbitrary HxW. Returned list has the same length, with each
        dict matching the single-image `predict_with_model` output:
            {'pred_x_L', 'pred_x_R', 'pred_exists_L', 'pred_exists_R'}
"""

from __future__ import annotations

import json
from typing import Callable

import cv2
import numpy as np

from lcms_lane_benchmark.data.lane_gt import LaneGeometricTruth
from lcms_lane_benchmark.evaluation.metrics import per_image_boundary_errors_full


def _load_val_entries(manifest_path: str):
    with open(manifest_path, 'r', encoding='utf-8') as f:
        m = json.load(f)
    return list(m['splits'].get('val', []))


def _selection_presence(output, presence_rule):
    """Keep checkpoint selection independent of optional decoder metadata."""
    if presence_rule not in ('confidence', 'native'):
        raise ValueError('presence_rule must be confidence or native')
    decisions = []
    for side in ('L', 'R'):
        score = output.get('pred_conf_' + side)
        decisions.append(bool(output['pred_exists_' + side])
                         if presence_rule == 'native' or score is None
                         else float(score) >= 0.5)
    return tuple(decisions)


def compute_val_strict_iou(model,
                           manifest_path: str,
                           predict_batch_fn: Callable,
                           predict_kwargs: dict | None = None,
                           batch_size: int = 8,
                           presence_rule: str = 'confidence') -> dict:
    """Mean strict + standard op_iou over the val split, with batched
    inference. `predict_batch_fn(model, [imgs], **kwargs) -> [outs]`.

    Also returns ``val_gra`` using the explicitly chosen presence rule:
    continuous confidence >= 0.5, or the adapter's native decisions.
    Native selection preserves the recorded SCNN, PolyLaneNet, UFLDv2,
    and LaneATT protocol even when their batch adapters expose confidence
    metadata for consistency with individual-image inference.
    This is the per-epoch selection metric for the GRA-selected
    checkpoint in the checkpoint-selection sensitivity analysis. It is
    computed in a SEPARATE metric pass so the strict-IoU selection values
    (and thus best.pth) stay byte-identical to before.
    """
    entries = _load_val_entries(manifest_path)
    predict_kwargs = predict_kwargs or {}
    strict_vals = []
    op_vals = []
    gra_vals = []

    for start in range(0, len(entries), batch_size):
        batch = entries[start:start + batch_size]
        imgs, gts, sizes = [], [], []
        for e in batch:
            img = cv2.imread(e['image_path'], cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            with open(e['clean_gt_path'], 'r', encoding='utf-8') as f:
                d = json.load(f)
            gL = LaneGeometricTruth.from_clean_gt_dict(d['left_lane'])
            gR = LaneGeometricTruth.from_clean_gt_dict(d['right_lane'])
            if gL is None:
                gL = LaneGeometricTruth(False, np.zeros(1), 0, None, [])
            if gR is None:
                gR = LaneGeometricTruth(False, np.zeros(1), 0, None, [])
            imgs.append(img)
            gts.append((gL, gR))
            sizes.append(img.shape[:2])

        if not imgs:
            continue

        # Single batched model forward + per-image post-processing inside.
        results = predict_batch_fn(model, imgs, **predict_kwargs)

        for (H, W), (gL, gR), out in zip(sizes, gts, results):
            rec = per_image_boundary_errors_full(
                pred_xs_L=out['pred_x_L'],
                pred_exists_L=out['pred_exists_L'],
                pred_xs_R=out['pred_x_R'],
                pred_exists_R=out['pred_exists_R'],
                gt_L=gL, gt_R=gR, H=H, W=W,
            )
            s = rec.get('strict_operational_iou')
            o = rec.get('operational_iou')
            if s is not None and np.isfinite(s):
                strict_vals.append(s)
            if o is not None and np.isfinite(o):
                op_vals.append(o)

            # The explicit rule prevents new confidence metadata from silently
            # changing checkpoint selection. This pass leaves strict/op values
            # above untouched.
            eL05, eR05 = _selection_presence(out, presence_rule)
            rec_g = per_image_boundary_errors_full(
                pred_xs_L=out['pred_x_L'],
                pred_exists_L=eL05,
                pred_xs_R=out['pred_x_R'],
                pred_exists_R=eR05,
                gt_L=gL, gt_R=gR, H=H, W=W,
            )
            g = rec_g.get('gated_region_accuracy')
            if g is not None and np.isfinite(g):
                gra_vals.append(g)

    return {
        'val_strict_iou': float(np.mean(strict_vals)) if strict_vals else 0.0,
        'val_op_iou': float(np.mean(op_vals)) if op_vals else 0.0,
        'val_gra': float(np.mean(gra_vals)) if gra_vals else 0.0,
        'n_images': len(entries),
        'n_with_strict': len(strict_vals),
    }
