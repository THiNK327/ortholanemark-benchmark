"""Canny + Probabilistic Hough (Canny TPAMI 1986; Matas et al. CVIU 2000).

Textbook reference baseline for line detection in grayscale images:

  1. Canny edge map (`cv2.Canny`) with auto-tuned thresholds based on
     image-statistics (Otsu-derived).
  2. Probabilistic Hough transform (`cv2.HoughLinesP`) extracts line
     segments from the edge map.
  3. Shared post-processing (`_common.py`) filters near-vertical
     segments, groups L/R, fits degree-1 polynomials.
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


@register_method('canny_hough')
class CannyHoughMethod(BaseLiteratureMethod):
    """Canny edges → probabilistic Hough → polynomial fit per side.

    Threshold strategy: derive the Canny high threshold from Otsu of
    the input image (good empirical default; no manual tuning per image).
    """

    def __init__(self,
                 hough_threshold: int = 60,
                 hough_min_line_length: int = 40,
                 hough_max_line_gap: int = 12,
                 max_angle_from_vertical_deg: float = 25.0,
                 min_segment_length_px: float = 30.0,
                 min_total_length_px: float = 80.0):
        self.hough_threshold = int(hough_threshold)
        self.hough_min_line_length = int(hough_min_line_length)
        self.hough_max_line_gap = int(hough_max_line_gap)
        self.max_angle = float(max_angle_from_vertical_deg)
        self.min_seg_len = float(min_segment_length_px)
        self.min_total_len = float(min_total_length_px)

    def predict_image(self, image_gray: np.ndarray) -> dict:
        t0 = time.perf_counter()
        H, W = image_gray.shape[:2]

        # Auto-threshold via Otsu on a blurred image (more robust than
        # fixed 50/150 thresholds for our wide-dynamic-range pavement imagery).
        blurred = cv2.GaussianBlur(image_gray, (5, 5), 1.2)
        otsu_high, _ = cv2.threshold(
            blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        canny_high = max(40, int(otsu_high))
        canny_low = int(canny_high * 0.4)

        edges = cv2.Canny(blurred, canny_low, canny_high, L2gradient=True)

        # Probabilistic Hough — angle resolution 1 degree.
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180.0,
            threshold=self.hough_threshold,
            minLineLength=self.hough_min_line_length,
            maxLineGap=self.hough_max_line_gap,
        )
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
