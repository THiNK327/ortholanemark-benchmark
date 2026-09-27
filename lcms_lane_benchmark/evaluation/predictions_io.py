"""Persist raw per-row predictions to NPZ for offline metric recomputation.

Each saved `predictions.npz` contains arrays parallel-indexed by image
position in the input report:

    stems            : (N,) str            — image stems
    projects         : (N,) str            — project names
    image_h          : (N,) int32          — image height (H)
    image_w          : (N,) int32          — image width (W)
    pred_exists_L    : (N,) bool           — model's per-side existence call
    pred_exists_R    : (N,) bool
    gt_exists_L      : (N,) bool           — GT existence (has_marking)
    gt_exists_R      : (N,) bool
    pred_x_L         : object array of (H,) float64 — per-row pred x (NaN off-frame)
    pred_x_R         : object array of (H,) float64

Using object arrays keeps variable-length per-row predictions in one
file without padding. Downstream consumers can recompute any metric
without re-running the full inference pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np


def save_predictions_npz(path: Path, per_image: list) -> None:
    """Dump raw per-row predictions for a list of per-image records.

    Each record in `per_image` must expose:
        stem, project, image_h, image_w,
        pred_x_L, pred_x_R, pred_exists_L, pred_exists_R,
        gt_L, gt_R (LaneGeometricTruth or None)
    """
    n = len(per_image)
    stems = np.array([str(r.get('stem', '')) for r in per_image])
    projects = np.array([str(r.get('project', 'unknown')) for r in per_image])
    image_h = np.array([int(r['image_h']) for r in per_image], dtype=np.int32)
    image_w = np.array([int(r['image_w']) for r in per_image], dtype=np.int32)
    pred_exists_L = np.array([bool(r['pred_exists_L']) for r in per_image])
    pred_exists_R = np.array([bool(r['pred_exists_R']) for r in per_image])
    # Continuous per-side existence confidence for PR-over-τ. Defaults to
    # the boolean call (degenerate flat PR) when an adapter doesn't expose
    # a score.
    pred_conf_L = np.array([float(r.get('pred_conf_L',
                                        float(r['pred_exists_L'])))
                            for r in per_image], dtype=np.float64)
    pred_conf_R = np.array([float(r.get('pred_conf_R',
                                        float(r['pred_exists_R'])))
                            for r in per_image], dtype=np.float64)

    def _gt_has(rec, key):
        gt = rec.get(key)
        return bool(gt is not None and getattr(gt, 'has_marking', False))

    gt_exists_L = np.array([_gt_has(r, 'gt_L') for r in per_image])
    gt_exists_R = np.array([_gt_has(r, 'gt_R') for r in per_image])

    pred_x_L = np.empty(n, dtype=object)
    pred_x_R = np.empty(n, dtype=object)
    for i, r in enumerate(per_image):
        pred_x_L[i] = np.asarray(r['pred_x_L'], dtype=np.float64)
        pred_x_R[i] = np.asarray(r['pred_x_R'], dtype=np.float64)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        stems=stems,
        projects=projects,
        image_h=image_h,
        image_w=image_w,
        pred_exists_L=pred_exists_L,
        pred_exists_R=pred_exists_R,
        pred_conf_L=pred_conf_L,
        pred_conf_R=pred_conf_R,
        gt_exists_L=gt_exists_L,
        gt_exists_R=gt_exists_R,
        pred_x_L=pred_x_L,
        pred_x_R=pred_x_R,
    )


def load_predictions_npz(path: Path) -> dict:
    """Load a `predictions.npz` written by `save_predictions_npz`.

    Returns a dict with the same keys as the saved arrays. `pred_x_L`
    and `pred_x_R` are object arrays of per-row float64 vectors.
    """
    d = np.load(Path(path), allow_pickle=True)
    return {
        'stems': d['stems'],
        'projects': d['projects'],
        'image_h': d['image_h'],
        'image_w': d['image_w'],
        'pred_exists_L': d['pred_exists_L'],
        'pred_exists_R': d['pred_exists_R'],
        'pred_conf_L': d['pred_conf_L'] if 'pred_conf_L' in d else
        d['pred_exists_L'].astype(np.float64),
        'pred_conf_R': d['pred_conf_R'] if 'pred_conf_R' in d else
        d['pred_exists_R'].astype(np.float64),
        'gt_exists_L': d['gt_exists_L'],
        'gt_exists_R': d['gt_exists_R'],
        'pred_x_L': d['pred_x_L'],
        'pred_x_R': d['pred_x_R'],
    }
