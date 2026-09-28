"""Dataset shim feeding our clean_gt into PolyLaneNet's expected sample
format.

Upstream's `lib/datasets/lane_dataset.py:transform_annotation` produces
labels of shape (max_lanes, 1 + 2 + 2*max_points):
    [:, 0]                              = category (0=no lane, ≥1=lane id)
    [:, 1]                              = lower_y normalized
    [:, 2]                              = upper_y normalized
    [:, 3 : 3+max_points]               = x_normalized at sample rows
    [:, 3+max_points : 3+2*max_points]  = y_normalized at those rows

`max_points` is upstream's `dataset.max_points` (typically 56 for
TuSimple, equal to len(h_samples)).

Our adaptation: lane index 0 = LEFT, lane index 1 = RIGHT — positional,
not sorted-by-leftmost as upstream's TuSimple loader does. Justified:
our clean_gt has explicit L/R fields; we want the model's confidence head
to learn "this is the left lane" and "this is the right lane" specifically,
matching the LLAMAS / CULane convention where lane indices are positional.
This is purely a label-assignment choice; the model architecture, loss,
and `num_outputs=2*7` layout are unchanged.

Per-sample dict (matches upstream `__getitem__` return signature
`(images, labels, img_idxs)`):
    images : float32 (3, img_h, img_w) ImageNet-normalized
    labels : float32 (max_lanes, 1 + 2 + 2*max_points)
    img_idxs: int (sample index in the manifest)
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from ortholanemark.data.lane_gt import LaneGeometricTruth
from ortholanemark.literature._uniform_aug import (
    sample_rotation_angle, rotate_image, rotate_polynomial_samples,
)


_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _load_gt_pair(clean_gt_path: str):
    with open(clean_gt_path, 'r', encoding='utf-8') as f:
        d = json.load(f)
    H = int(d['image_size']['height'])
    W = int(d['image_size']['width'])
    gL = LaneGeometricTruth.from_clean_gt_dict(d['left_lane'])
    gR = LaneGeometricTruth.from_clean_gt_dict(d['right_lane'])
    if gL is None:
        gL = LaneGeometricTruth(False, np.zeros(1), 0, None, [])
    if gR is None:
        gR = LaneGeometricTruth(False, np.zeros(1), 0, None, [])
    return gL, gR, H, W


def _build_lane_label(gt: LaneGeometricTruth, max_points: int,
                       H_orig: int, W_orig: int) -> tuple:
    """Returns (category, lower_y_norm, upper_y_norm, xs_norm, ys_norm).
    xs/ys length = max_points, filled with -1 for missing rows.
    """
    xs = np.full(max_points, -1.0, dtype=np.float32)
    ys = np.full(max_points, -1.0, dtype=np.float32)
    if not gt.has_marking:
        return 0.0, 0.0, 0.0, xs, ys
    # Sample y at uniformly-spaced rows across the full image (we use
    # the full image because OrthoLaneMark annotations go top-to-bottom).
    ys_orig = np.linspace(0.0, H_orig - 1, max_points)
    xs_orig = gt.eval_x_at(ys_orig, W_orig, clamp_to_frame=False)
    in_frame = np.isfinite(xs_orig) & (xs_orig >= 0) & (xs_orig <= W_orig - 1)
    if not in_frame.any():
        return 0.0, 0.0, 0.0, xs, ys
    # xs / ys are normalized to [0, 1].
    xs_norm = (xs_orig[in_frame] / max(float(W_orig - 1), 1.0)).astype(np.float32)
    ys_norm = (ys_orig[in_frame] / max(float(H_orig - 1), 1.0)).astype(np.float32)
    n = len(xs_norm)
    xs[:n] = xs_norm
    ys[:n] = ys_norm
    lower = float(ys_norm.min())
    upper = float(ys_norm.max())
    return 1.0, lower, upper, xs, ys


class PolyLaneNetDataset(Dataset):
    """Reads our manifest, returns (images, labels, img_idx) per upstream
    PolyLaneNet's train loop contract."""

    def __init__(self,
                 manifest_path: str,
                 split: str,
                 img_h: int = 360,
                 img_w: int = 640,
                 max_lanes: int = 2,
                 max_points: int = 56,
                 rotation_deg: float = 0.0):
        with open(manifest_path, 'r') as f:
            m = json.load(f)
        self.entries = list(m['splits'].get(split, []))
        self.img_h = int(img_h)
        self.img_w = int(img_w)
        self.max_lanes = int(max_lanes)
        self.max_points = int(max_points)
        self.rotation_deg = float(rotation_deg)

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int):
        e = self.entries[idx]
        img = cv2.imread(e['image_path'], cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(e['image_path'])
        H_orig, W_orig = img.shape[:2]
        img_r = cv2.resize(img, (self.img_w, self.img_h),
                           interpolation=cv2.INTER_AREA)

        gL, gR, _, _ = _load_gt_pair(e['clean_gt_path'])

        # Sample rotation (train only via rotation_deg > 0; rotation_deg=0
        # → no-op). Image is rotated; labels (xs, ys at max_points
        # uniformly-spaced y) are built from polynomial samples that we
        # rotate by the same angle so they stay aligned with the image.
        angle = sample_rotation_angle(self.rotation_deg)
        if angle != 0.0:
            img_r = rotate_image(img_r, angle)

        img_3 = np.stack([img_r, img_r, img_r], axis=-1).astype(np.float32) / 255.0
        img_3 = (img_3 - _IMAGENET_MEAN) / _IMAGENET_STD
        images = torch.from_numpy(
            np.ascontiguousarray(img_3.transpose(2, 0, 1))).float()

        label = np.ones((self.max_lanes, 3 + 2 * self.max_points),
                        dtype=np.float32) * -1e5
        # Mirror upstream: lanes[:, 0] = 0 is the "no lane" default,
        # then we set fields for actual lanes. Our positional convention:
        # lane 0 = L, lane 1 = R.
        label[:, 0] = 0.0
        for lane_idx, gt in enumerate((gL, gR)):
            if lane_idx >= self.max_lanes:
                break
            cat, lower, upper, xs, ys = _build_rotated_lane_label(
                gt, self.max_points, H_orig, W_orig,
                self.img_h, self.img_w, angle)
            label[lane_idx, 0] = cat
            label[lane_idx, 1] = lower
            label[lane_idx, 2] = upper
            label[lane_idx, 3:3 + self.max_points] = xs
            label[lane_idx, 3 + self.max_points:3 + 2 * self.max_points] = ys

        return images, torch.from_numpy(label), idx


def _build_rotated_lane_label(gt: LaneGeometricTruth, max_points: int,
                               H_orig: int, W_orig: int,
                               img_h: int, img_w: int, angle: float
                               ) -> tuple:
    """Same as `_build_lane_label`, but applies the same rotation that
    was applied to the image. Samples the polynomial at the model's
    INPUT resolution (img_h, img_w) — upstream PolyLaneNet operates on
    the resized image, and our normalized [0, 1] labels are with respect
    to the input image, not the original.
    """
    xs = np.full(max_points, -1.0, dtype=np.float32)
    ys = np.full(max_points, -1.0, dtype=np.float32)
    if not gt.has_marking:
        return 0.0, 0.0, 0.0, xs, ys

    # Sample at max_points y rows in the resized image; map via H_orig/W_orig
    # to evaluate the polynomial.
    ys_res = np.linspace(0.0, img_h - 1, max_points, dtype=np.float64)
    ys_orig = ys_res * (H_orig - 1) / max(float(img_h - 1), 1.0)
    xs_orig = gt.eval_x_at(ys_orig, W_orig, clamp_to_frame=False)
    xs_res = xs_orig * (img_w - 1) / max(float(W_orig - 1), 1.0)

    # Rotate sample points by the same angle as the image.
    if angle != 0.0:
        ys_res, xs_res = rotate_polynomial_samples(
            ys_res, xs_res, angle, img_h, img_w)

    in_frame = (np.isfinite(xs_res) & np.isfinite(ys_res)
                & (xs_res >= 0) & (xs_res <= img_w - 1)
                & (ys_res >= 0) & (ys_res <= img_h - 1))
    if not in_frame.any():
        return 0.0, 0.0, 0.0, xs, ys

    xs_norm = (xs_res[in_frame] / max(float(img_w - 1), 1.0)).astype(np.float32)
    ys_norm = (ys_res[in_frame] / max(float(img_h - 1), 1.0)).astype(np.float32)
    n = len(xs_norm)
    xs[:n] = xs_norm
    ys[:n] = ys_norm
    lower = float(ys_norm.min())
    upper = float(ys_norm.max())
    return 1.0, lower, upper, xs, ys
