"""Trivial 'always predict no markings' baseline (plan §3.3).

Predicts existence = False on both sides for every image, so the
operational region defaults to the full image width. Its purpose is
diagnostic, not competitive: on images whose markings sit near the frame
edge, the default-to-edge region barely differs from the true travel
lane, so this no-detection baseline scores a deceptively high region
IoU while finding nothing. Pairing it with existence-gated metrics
(strict operational IoU / existence F1) is the motivating exhibit for
the metric design — any metric under which this baseline ranks well is
not measuring detection.
"""

from __future__ import annotations

import time

import numpy as np

from ortholanemark.literature import (
    BaseLiteratureMethod, register_method,
)


@register_method('trivial_none')
class TrivialNoneMethod(BaseLiteratureMethod):
    """Always predicts that no marking exists on either side."""

    def predict_image(self, image_gray: np.ndarray) -> dict:
        t0 = time.perf_counter()
        H = int(image_gray.shape[0])
        nan = np.full(H, np.nan, dtype=np.float64)
        return {
            'pred_x_L': nan,
            'pred_x_R': nan.copy(),
            'pred_exists_L': False,
            'pred_exists_R': False,
            'inference_ms': (time.perf_counter() - t0) * 1000.0,
        }
