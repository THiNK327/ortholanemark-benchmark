"""Shared uniform-augmentation helpers used by faithful dataset
shims, so the augmentation pipeline is identical across methods.

Current uniform-aug policy (Option A from FIDELITY_REVIEW.md):
    train: Rotation(±2°) applied to image AND polynomial GT samples
    val/test: identity (no augmentation)

The polynomial is rotated by transforming its sampled (y, x) points
around the image center, then methods can re-fit / re-bin / re-paint
from the rotated samples as their native loss expects.

Why no HorizontalFlip: would swap our positional L/R lane identities.
Why no Affine translate/scale: pavement survey frames are captured at fixed scale
and roughly axis-aligned; matches Pan et al.'s `Rotation(2)` philosophy.
"""

from __future__ import annotations

import cv2
import numpy as np

from ortholanemark.data.lane_gt import LaneGeometricTruth


def sample_rotation_angle(rotation_deg: float,
                          rng: np.random.Generator | None = None
                          ) -> float:
    """Sample a uniform random angle in [-rotation_deg, +rotation_deg]."""
    if rotation_deg <= 0.0:
        return 0.0
    if rng is None:
        return float(np.random.uniform(-rotation_deg, rotation_deg))
    return float(rng.uniform(-rotation_deg, rotation_deg))


def rotate_image(img: np.ndarray, angle_deg: float,
                 interpolation: int = cv2.INTER_LINEAR,
                 border_mode: int = cv2.BORDER_REFLECT,
                 border_value=0) -> np.ndarray:
    """Rotate a single-channel or 3-channel image around its center."""
    if angle_deg == 0.0:
        return img
    H, W = img.shape[:2]
    M = cv2.getRotationMatrix2D((W / 2.0, H / 2.0), angle_deg, 1.0)
    return cv2.warpAffine(img, M, (W, H), flags=interpolation,
                          borderMode=border_mode, borderValue=border_value)


def rotate_seg_mask(mask: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate an integer-valued segmentation mask using NEAREST so the
    class labels are preserved."""
    if angle_deg == 0.0:
        return mask
    H, W = mask.shape[:2]
    M = cv2.getRotationMatrix2D((W / 2.0, H / 2.0), angle_deg, 1.0)
    return cv2.warpAffine(mask.astype(np.uint8), M, (W, H),
                          flags=cv2.INTER_NEAREST,
                          borderMode=cv2.BORDER_CONSTANT,
                          borderValue=0).astype(mask.dtype)


def rotate_polynomial_samples(ys: np.ndarray, xs: np.ndarray,
                              angle_deg: float, H: int, W: int
                              ) -> tuple[np.ndarray, np.ndarray]:
    """Rotate (y, x) sample points around the image center by `angle_deg`.

    The convention matches `cv2.getRotationMatrix2D`: a positive angle
    rotates the image content **counter-clockwise**. We apply the SAME
    rotation to the polynomial samples so they match the rotated image.

    Off-frame points (NaN x) stay NaN. Points pushed off-frame by the
    rotation are also set to NaN.
    """
    if angle_deg == 0.0:
        return ys, xs
    theta = np.deg2rad(angle_deg)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    cx, cy = W / 2.0, H / 2.0
    # cv2's rotation is around (cx, cy) and CCW for positive angle.
    # The warpAffine matrix maps src→dst as M·p. We want to transform a
    # GT point in the unrotated image to its position in the rotated
    # image. That's the SAME transform cv2 applies to pixels.
    finite = np.isfinite(xs) & np.isfinite(ys)
    xs_c = xs - cx
    ys_c = ys - cy
    # cv2.getRotationMatrix2D(angle, scale=1):
    #   alpha = cos(angle); beta = sin(angle)
    #   M = [[ alpha, beta,  (1-alpha)*cx - beta*cy],
    #        [-beta,  alpha, beta*cx + (1-alpha)*cy]]
    # Applied to (x, y): (x', y') = (alpha*x + beta*y + tx, -beta*x + alpha*y + ty)
    # In our recentered space:
    xs_new = cos_t * xs_c + sin_t * ys_c + cx
    ys_new = -sin_t * xs_c + cos_t * ys_c + cy
    # Mark points pushed off the frame as NaN
    off = (xs_new < 0) | (xs_new >= W) | (ys_new < 0) | (ys_new >= H)
    xs_out = np.where(finite & ~off, xs_new, np.nan)
    ys_out = np.where(finite & ~off, ys_new, np.nan)
    return ys_out, xs_out


def evaluate_and_rotate_lane(gt: LaneGeometricTruth, H: int, W: int,
                              angle_deg: float,
                              n_samples: int = 200
                              ) -> tuple[np.ndarray, np.ndarray]:
    """Convenience: evaluate the GT polynomial at `n_samples` rows
    spanning [0, H), then rotate the (y, x) sample points by `angle_deg`.

    Returns (ys_rot, xs_rot), both length n_samples. NaN entries mean
    the polynomial was off-frame OR the rotation pushed the point off-
    frame.
    """
    if gt is None or not gt.has_marking:
        return (np.full(n_samples, np.nan, dtype=np.float64),
                np.full(n_samples, np.nan, dtype=np.float64))
    ys = np.linspace(0.0, H - 1, n_samples, dtype=np.float64)
    xs = gt.eval_x_at(ys, W, clamp_to_frame=False)
    return rotate_polynomial_samples(ys, xs, angle_deg, H, W)
