"""LSD — Line Segment Detector (Grompone von Gioi et al., TPAMI 2010).

Parameter-free line detection. Uses `cv2.createLineSegmentDetector()`.
On grayscale pavement imagery the detector finds many short segments
along paint stripes; the shared post-processing in `_common.py`
filters to near-vertical orientation and fits degree-1 curves per side.
"""

from __future__ import annotations

import time

import cv2
import numpy as np

from ortholanemark.literature import (
    BaseLiteratureMethod, register_method,
)
from ortholanemark.literature._common import (
    filter_near_vertical, segments_to_points, edge_inner_curves,
)


@register_method('lsd')
class LSDMethod(BaseLiteratureMethod):
    """OpenCV's Line Segment Detector with the standard post-processing
    pipeline (near-vertical filter → L/R grouping → degree-1 fit)."""

    def __init__(self,
                 max_angle_from_vertical_deg: float = 25.0,
                 min_segment_length_px: float = 30.0,
                 min_total_length_px: float = 80.0):
        self.detector = cv2.createLineSegmentDetector()
        self.max_angle = float(max_angle_from_vertical_deg)
        self.min_seg_len = float(min_segment_length_px)
        self.min_total_len = float(min_total_length_px)

    def predict_image(self, image_gray: np.ndarray) -> dict:
        t0 = time.perf_counter()
        H, W = image_gray.shape[:2]
        # LSD wants uint8 grayscale; image_gray already is.
        # detect() returns (N, 1, 4) of (x1, y1, x2, y2) plus widths/prec/nfa.
        lines, _, _, _ = self.detector.detect(image_gray)
        if lines is None:
            segs = np.zeros((0, 4), dtype=np.float64)
        else:
            segs = lines.reshape(-1, 4).astype(np.float64)
        # Inner-edge selection from the detector's own edges: near-vertical
        # filter → discard non-marking edges → per-side RANSAC location →
        # lane-facing inner-edge envelope + fit (see _common.py).
        filtered = filter_near_vertical(
            segs, max_angle_from_vertical_deg=self.max_angle,
            min_length_px=self.min_seg_len)
        pred_x_L, pred_x_R, exists_L, exists_R = edge_inner_curves(
            filtered, image_gray, min_total_length_px=self.min_total_len)
        t1 = time.perf_counter()
        return {
            'pred_x_L': pred_x_L,
            'pred_x_R': pred_x_R,
            'pred_exists_L': exists_L,
            'pred_exists_R': exists_R,
            'inference_ms': (t1 - t0) * 1000.0,
        }
