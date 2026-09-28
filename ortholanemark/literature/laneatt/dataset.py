"""Dataset shim feeding our clean_gt into LaneATT's expected sample
format (per upstream `lib/datasets/lane_dataset.py:transform_annotation`).

Per-image label shape: (max_lanes, 5 + S) where S = n_offsets = 72.
  [:, 0] = cls0  (1 = invalid lane, 0 = valid)
  [:, 1] = cls1  (0 = invalid, 1 = valid)
  [:, 2] = start_y  = num_offsets_outside_image / n_strips
  [:, 3] = start_x  = pixel x at the first inside-image y-position
  [:, 4] = length   = number of in-frame x values
  [:, 5:5+S] = x values in PIXEL units (NOT normalized) at the S
              `offsets_ys` y-positions (img_h down to 0 in strides of
              strip_size = img_h / (S-1))

Adaptations from upstream:
  * lane index 0 = LEFT, lane index 1 = RIGHT (positional). Upstream's
    TuSimple loader sorts lanes by leftmost x; for our task with explicit
    L/R semantics in clean_gt, we always place L at [0] and R at [1].
    Identical to upstream's behaviour when GT has exactly two lanes
    that already obey L<R (the common case), but more robust at the
    edges (e.g., one side present, or near-centered marking).
  * Per-sample dict shape unchanged. Returns (img_tensor, label_array,
    img_idx), matching upstream's __getitem__ contract used by the
    training loop.

Image preprocessing: resize 2500x1040 portrait grayscale → 360x640
landscape RGB in [0, 1], matching upstream LaneATT configs where
`normalize: false`. Lanes remain vertical after resize.
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


def _sample_lane(gt: LaneGeometricTruth, H_orig: int, W_orig: int,
                  img_h: int, img_w: int, offsets_ys: np.ndarray
                  ) -> tuple:
    """Mirror upstream's sample_lane logic.
    Returns (xs_outside_image, xs_inside_image, lane_visible_bool).
    xs are in PIXEL units of the resized image (img_w-wide).
    """
    if not gt.has_marking:
        return np.array([]), np.array([]), False
    # Eval poly at the offsets_ys (which run from img_h down to 0).
    # Need original-image y for poly eval.
    ys_orig = offsets_ys * (H_orig - 1) / max(float(img_h - 1), 1.0)
    xs_orig = gt.eval_x_at(ys_orig, W_orig, clamp_to_frame=False)
    # Map to resized pixel-x.
    xs_resized = xs_orig * (img_w - 1) / max(float(W_orig - 1), 1.0)
    in_frame = (xs_resized >= 0) & (xs_resized < img_w) & np.isfinite(xs_resized)
    xs_inside = xs_resized[in_frame]
    xs_outside = xs_resized[~in_frame]
    if len(xs_inside) == 0:
        return xs_outside, xs_inside, False
    return xs_outside, xs_inside, True


class LaneATTDataset(Dataset):
    """Produces samples in upstream LaneATT's expected format.

    The training loop, focal loss, and matching code consume these
    unchanged.
    """

    def __init__(self,
                 manifest_path: str,
                 split: str,
                 img_h: int = 360,
                 img_w: int = 640,
                 S: int = 72,
                 max_lanes: int = 2,
                 rotation_deg: float = 0.0):
        with open(manifest_path, 'r') as f:
            m = json.load(f)
        self.entries = list(m['splits'].get(split, []))
        self.img_h = int(img_h)
        self.img_w = int(img_w)
        self.S = int(S)
        self.n_strips = self.S - 1
        self.n_offsets = self.S
        self.strip_size = self.img_h / self.n_strips
        # offsets_ys: img_h, img_h - strip_size, ..., 0 (descending)
        self.offsets_ys = np.arange(self.img_h, -1, -self.strip_size,
                                     dtype=np.float64)
        # In upstream, arange's behavior here gives exactly S+1 entries
        # sometimes — clip to S to match the (5+S) label width.
        self.offsets_ys = self.offsets_ys[:self.S]
        self.max_lanes = int(max_lanes)
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

        # Sample rotation and apply to image. Labels are built below
        # from polynomial samples that we rotate by the same angle.
        angle = sample_rotation_angle(self.rotation_deg)
        if angle != 0.0:
            img_r = rotate_image(img_r, angle)

        # Upstream LaneATT configs use normalize: false; keep images in
        # [0, 1] after grayscale-to-RGB replication.
        img_3 = np.stack([img_r, img_r, img_r], axis=-1).astype(np.float32) / 255.0
        images = torch.from_numpy(
            np.ascontiguousarray(img_3.transpose(2, 0, 1))).float()

        # Label array: max_lanes × (5 + S).
        lanes = np.ones((self.max_lanes, 5 + self.S), dtype=np.float32) * -1e5
        # Defaults: invalid (cls0=1, cls1=0)
        lanes[:, 0] = 1
        lanes[:, 1] = 0

        for lane_idx, gt in enumerate((gL, gR)):
            if lane_idx >= self.max_lanes:
                break
            xs_out, xs_in, visible = _sample_lane_with_rotation(
                gt, H_orig, W_orig, self.img_h, self.img_w,
                self.offsets_ys, angle)
            if not visible:
                continue
            all_xs = np.hstack((xs_out, xs_in))
            lanes[lane_idx, 0] = 0
            lanes[lane_idx, 1] = 1
            lanes[lane_idx, 2] = len(xs_out) / self.n_strips
            lanes[lane_idx, 3] = float(xs_in[0])
            lanes[lane_idx, 4] = len(xs_in)
            lanes[lane_idx, 5:5 + len(all_xs)] = all_xs

        return images, torch.from_numpy(lanes), idx


def _sample_lane_with_rotation(gt: LaneGeometricTruth,
                                H_orig: int, W_orig: int,
                                img_h: int, img_w: int,
                                offsets_ys: np.ndarray, angle: float
                                ) -> tuple:
    """Same as `_sample_lane`, but applies the same rotation as the image.

    The rotation operates on (y, x) sample points: we evaluate the
    polynomial at upstream's `offsets_ys` y values, then rotate the
    sample points around (img_w/2, img_h/2). Because rotation changes y
    too, we then RE-INTERPOLATE x values at the original `offsets_ys`
    so the (5+S) label layout (one x per strip y) still matches.
    """
    if not gt.has_marking:
        return np.array([]), np.array([]), False

    # Eval poly at the strip y positions (img_h coords).
    ys_orig = offsets_ys * (H_orig - 1) / max(float(img_h - 1), 1.0)
    xs_orig = gt.eval_x_at(ys_orig, W_orig, clamp_to_frame=False)
    # Map to resized pixel-x.
    xs_resized = xs_orig * (img_w - 1) / max(float(W_orig - 1), 1.0)
    ys_resized = offsets_ys.copy()

    if angle != 0.0:
        # Rotate the sample points by the SAME rotation as the image.
        ys_rot, xs_rot = rotate_polynomial_samples(
            ys_resized, xs_resized, angle, img_h, img_w)
        # Re-interpolate x at the strip y positions (`offsets_ys`).
        finite = np.isfinite(ys_rot) & np.isfinite(xs_rot)
        if finite.sum() < 2:
            return np.array([]), np.array([]), False
        ys_pts = ys_rot[finite]
        xs_pts = xs_rot[finite]
        order = np.argsort(ys_pts)
        ys_pts, xs_pts = ys_pts[order], xs_pts[order]
        # Linear interp; outside extent → NaN.
        xs_resized = np.interp(offsets_ys, ys_pts, xs_pts,
                                left=np.nan, right=np.nan)

    in_frame = (xs_resized >= 0) & (xs_resized < img_w) & np.isfinite(xs_resized)
    xs_inside = xs_resized[in_frame]
    xs_outside = xs_resized[~in_frame]
    # Upstream label invariant: every label slot must be finite. NaN at
    # off-frame strips poisons match_proposals_with_targets because
    # `0 * NaN = NaN` propagates through the masked sum, turning all
    # anchor distances into NaN (neither positive nor negative). Replace
    # NaN with 0 — these positions are masked out by start_y during
    # matching, so the actual value is irrelevant; only finiteness matters.
    xs_outside = np.where(np.isfinite(xs_outside), xs_outside, 0.0)
    if len(xs_inside) == 0:
        return xs_outside, xs_inside, False
    return xs_outside, xs_inside, True
