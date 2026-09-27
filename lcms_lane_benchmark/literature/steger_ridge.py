"""Steger ridge detection (Steger, TPAMI 1998).

Direct Hessian-eigenvalue ridge detector for bright-line-on-dark-background
pavement markings:

  1. Smooth the image at scale σ (Gaussian filter).
  2. Compute the Hessian at each pixel via Sobel-based second derivatives.
  3. Per-pixel ridge measure = the larger of the two negated eigenvalues
     of the Hessian (positive when the local Hessian is negative-definite,
     which is the case at the centerline of a bright ridge).
  4. Multi-scale: take the max ridge response over σ ∈ {2, 4, 6, 8}.
  5. Threshold to a binary ridge mask.
  6. Pixel-based fit: group ALL ridge pixels into left/right by x position,
     fit one polynomial per group through the pixel cloud. Bypasses the
     "segment" abstraction the other classical methods use because Steger
     naturally outputs a pixel mask rather than line segments.
"""

from __future__ import annotations

import time

import cv2
import numpy as np

from lcms_lane_benchmark.literature import (
    BaseLiteratureMethod, register_method,
)
from lcms_lane_benchmark.literature._common import robust_inner_curves


def _hessian_ridge_response(image: np.ndarray, sigma: float) -> np.ndarray:
    """Per-pixel ridge measure at scale σ.

    Computes the Hessian via Gaussian-smoothed Sobel second derivatives,
    then returns the larger of -λ1, -λ2 (positive at bright ridges).

    Returns (H, W) float32, non-negative.
    """
    # Smooth at this scale.
    k = max(3, int(2 * round(3 * sigma) + 1))
    img = cv2.GaussianBlur(image.astype(np.float32), (k, k), sigma)
    # Second derivatives via repeated Sobel.
    Ix = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    Iy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    Ixx = cv2.Sobel(Ix, cv2.CV_32F, 1, 0, ksize=3)
    Iyy = cv2.Sobel(Iy, cv2.CV_32F, 0, 1, ksize=3)
    Ixy = cv2.Sobel(Ix, cv2.CV_32F, 0, 1, ksize=3)
    # Eigenvalues of H = [[Ixx, Ixy], [Ixy, Iyy]].
    half_trace = 0.5 * (Ixx + Iyy)
    half_diff = 0.5 * (Ixx - Iyy)
    discriminant = np.sqrt(half_diff * half_diff + Ixy * Ixy)
    lam1 = half_trace + discriminant
    lam2 = half_trace - discriminant
    # Bright ridges: Hessian is negative-definite. We want where both
    # eigenvalues are negative; the "ridge strength" is min(-lam1, -lam2)
    # (we need BOTH to be strongly negative).
    neg_lam1 = -lam1
    neg_lam2 = -lam2
    ridge = np.minimum(neg_lam1, neg_lam2)
    ridge = np.maximum(ridge, 0.0)
    # Scale-normalised: multiply by σ² (Lindeberg's normalised derivative).
    return ridge * (sigma * sigma)


def _fit_pixels_to_curve(ys: np.ndarray, xs: np.ndarray, H: int, W: int,
                         min_pixels: int = 50,
                         min_y_span: float = 100.0) -> tuple:
    """Fit a degree-1 polynomial x = a1*y + a0 through a set of (y, x)
    pixel coordinates. Returns (pred_x_per_row (H,), exists bool).

    Returns NaN-array + False if:
      - too few pixels (<min_pixels), or
      - pixels are too clustered vertically (y-span < min_y_span — likely
        a tight ridge cluster, not a long line).
    """
    if len(ys) < min_pixels:
        return (np.full(H, np.nan, dtype=np.float64), False)
    y_span = float(ys.max() - ys.min())
    if y_span < min_y_span:
        return (np.full(H, np.nan, dtype=np.float64), False)
    coefs = np.polyfit(ys, xs, 1)
    rows = np.arange(H, dtype=np.float64)
    x_per_row = np.polyval(coefs, rows)
    in_frame = (x_per_row >= 0) & (x_per_row <= W - 1)
    x_per_row = np.where(in_frame, x_per_row, np.nan)
    return (x_per_row, True)


@register_method('steger_ridge')
class StegerRidgeMethod(BaseLiteratureMethod):
    """Multi-scale Hessian-based ridge detector. Yields contours that the
    shared post-processing converts into degree-1 per-side curves."""

    def __init__(self,
                 sigmas=(2.0, 4.0, 6.0, 8.0),
                 response_threshold_percentile: float = 97.0,
                 left_zone: float = 0.40,
                 right_zone: float = 0.60,
                 min_pixels: int = 100,
                 min_y_span: float = 200.0):
        self.sigmas = tuple(float(s) for s in sigmas)
        self.percentile = float(response_threshold_percentile)
        self.left_zone = float(left_zone)
        self.right_zone = float(right_zone)
        self.min_pixels = int(min_pixels)
        self.min_y_span = float(min_y_span)

    def predict_image(self, image_gray: np.ndarray) -> dict:
        t0 = time.perf_counter()
        H, W = image_gray.shape[:2]
        # Multi-scale: max over σ.
        ridge_max = np.zeros((H, W), dtype=np.float32)
        for sigma in self.sigmas:
            r = _hessian_ridge_response(image_gray, sigma)
            np.maximum(ridge_max, r, out=ridge_max)
        # Threshold via percentile (parameter-light: keep top ~3 %).
        if ridge_max.max() < 1e-6:
            t1 = time.perf_counter()
            return {
                'pred_x_L': np.full(H, np.nan, dtype=np.float64),
                'pred_x_R': np.full(H, np.nan, dtype=np.float64),
                'pred_exists_L': False, 'pred_exists_R': False,
                'inference_ms': (t1 - t0) * 1000.0,
            }
        thresh = float(np.percentile(ridge_max, self.percentile))
        mask = ridge_max >= thresh
        # Light denoising — drop isolated single pixels.
        mask_u8 = mask.astype(np.uint8) * 255
        mask_u8 = cv2.morphologyEx(
            mask_u8, cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        ys, xs = np.where(mask_u8 > 0)

        # Robust inner-boundary conversion from the ridge pixel cloud:
        # intensity gating → RANSAC near-vertical line → inner-edge snap.
        # (Replaces the earlier direct centerline fit, which sat ~half a
        # marking width off the annotated inner edge; see _common.py.)
        points = np.stack([ys.astype(np.float64), xs.astype(np.float64)], axis=1)
        pred_x_L, pred_x_R, exists_L, exists_R = robust_inner_curves(
            points, image_gray)

        t1 = time.perf_counter()
        return {
            'pred_x_L': pred_x_L,
            'pred_x_R': pred_x_R,
            'pred_exists_L': exists_L,
            'pred_exists_R': exists_R,
            'inference_ms': (t1 - t0) * 1000.0,
        }
