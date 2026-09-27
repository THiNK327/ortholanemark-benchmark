"""Dataset shim feeding our clean_gt into UFLDv2's expected sample
format.

Upstream uses NVIDIA DALI via `data/dali_data.py`. We replace it with
pure-PyTorch label generation; the per-sample output dict is exactly
what upstream's `inference_culane_tusimple` (utils/common.py:205-216)
expects, so the training loop, loss dict, and model are unchanged.

Per-sample dict produced (batch-collated by `collate()`):
    images           : float32 (3, H, W), ImageNet-normalized
    labels_row       : int64  (num_cls_row, num_lanes),
                       col-bin index ∈ [0, num_cell_row-1] ∪ {-1}
    labels_col       : int64  (num_cls_col, num_lanes),
                       row-bin index ∈ [0, num_cell_col-1] ∪ {-1}
    labels_row_float : float32 (num_cls_row, num_lanes),
                       x_norm ∈ [0, 1] ∪ {-1}
    labels_col_float : float32 (num_cls_col, num_lanes),
                       y_norm ∈ [0, 1] ∪ {-1}
    seg_label        : int64  (seg_h, seg_w),
                       multi-class lane id at stride-8 of input
                       (0=bg, 1=left, 2=right). Only present when
                       `paint_seg=True`; consumed by SegHead's CE loss.

Adaptations from upstream (each justified in the comment block):
  * **Pure-Python anchor sampling** instead of DALI + custom CUDA interp.
    The numerical recipe is identical (eval polynomial at anchor y for
    the row head; find y where x crosses anchor x for the col head).
    This avoids DALI build/Linux dependencies; output values match.
  * **Row anchors spread across full image height** (y_norm linspace
    0..1) instead of upstream's bottom-78% TuSimple anchors. Justified:
    in TuSimple, the camera-forward view's top-22% of pixels is the sky
    (no lanes ever). Our overhead pavement view has lanes top-to-bottom,
    so anchoring only at the bottom would discard half the supervision
    signal. Upstream's `data/constant.py:1-4` explicitly says
    "you can modify these row anchors according to your training image
    resolution."
  * **Col anchors at x_norm linspace 0..1** for the same reason.
  * **Lane indexing**: lane 0 = LEFT, lane 1 = RIGHT — positional, by
    construction. Upstream's TuSimple uses positional indexing too
    (lane 0 = leftmost ego line).
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from lcms_lane_benchmark.data.lane_gt import LaneGeometricTruth
from lcms_lane_benchmark.literature._uniform_aug import (
    sample_rotation_angle, rotate_image, evaluate_and_rotate_lane,
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


def _row_labels_from_samples(ys_res: np.ndarray, xs_res: np.ndarray,
                              input_h: int, input_w: int,
                              row_anchors_norm: np.ndarray,
                              num_cell_row: int):
    """Build row-anchor labels from (already-rotated) (y, x) sample points
    in resized image coords. For each row anchor y position, interpolate
    the lane x; convert to column bin.
    """
    n = len(row_anchors_norm)
    bins = np.full(n, -1, dtype=np.int64)
    fxs = np.full(n, -1.0, dtype=np.float32)
    finite = np.isfinite(ys_res) & np.isfinite(xs_res)
    if finite.sum() < 2:
        return bins, fxs
    ys_pts = ys_res[finite]
    xs_pts = xs_res[finite]
    # Sort by y for monotone interpolation
    order = np.argsort(ys_pts)
    ys_pts, xs_pts = ys_pts[order], xs_pts[order]
    for i, ya_norm in enumerate(row_anchors_norm):
        y_target = ya_norm * (input_h - 1)
        if y_target < ys_pts[0] or y_target > ys_pts[-1]:
            continue
        x_at = float(np.interp(y_target, ys_pts, xs_pts))
        if x_at < 0 or x_at > input_w - 1:
            continue
        x_norm = x_at / max(float(input_w - 1), 1.0)
        bi = int(round(x_norm * (num_cell_row - 1)))
        if 0 <= bi < num_cell_row:
            bins[i] = bi
            fxs[i] = x_norm
    return bins, fxs


def _paint_seg_label(per_lane_samples: list,
                      input_h: int, input_w: int,
                      seg_stride: int = 8,
                      line_width: int = 8) -> np.ndarray:
    """Build multi-class lane seg label at stride-`seg_stride` of input.

    Mirrors upstream CULane's seg label: 0=bg, lane_idx+1 for each lane.
    `per_lane_samples` is a list of (ys_res, xs_res) in resized+rotated
    input-image coords. We polyline each lane on a full-resolution mask
    with `line_width` pixels, then INTER_NEAREST-downsample to the
    SegHead's output resolution.

    line_width=8 matches SCNN's faithful painter — both reuse the same
    polynomials, so the supervision strengths are comparable.
    """
    full = np.zeros((input_h, input_w), dtype=np.uint8)
    for lane_idx, (ys_res, xs_res) in enumerate(per_lane_samples):
        finite = np.isfinite(ys_res) & np.isfinite(xs_res)
        if finite.sum() < 2:
            continue
        ys_pts = ys_res[finite]
        xs_pts = xs_res[finite]
        order = np.argsort(ys_pts)
        ys_pts, xs_pts = ys_pts[order], xs_pts[order]
        pts = np.stack([xs_pts, ys_pts], axis=1).round().astype(np.int32)
        in_frame = ((pts[:, 0] >= 0) & (pts[:, 0] < input_w)
                    & (pts[:, 1] >= 0) & (pts[:, 1] < input_h))
        pts = pts[in_frame]
        if len(pts) < 2:
            continue
        cv2.polylines(full, [pts], False, color=lane_idx + 1,
                       thickness=line_width, lineType=cv2.LINE_8)
    seg_h = input_h // seg_stride
    seg_w = input_w // seg_stride
    seg = cv2.resize(full, (seg_w, seg_h), interpolation=cv2.INTER_NEAREST)
    return seg.astype(np.int64)


def _col_labels_from_samples(ys_res: np.ndarray, xs_res: np.ndarray,
                              input_h: int, input_w: int,
                              col_anchors_norm: np.ndarray,
                              num_cell_col: int,
                              x_tol_norm: float = 0.5 / 100.0):
    """Build col-anchor labels from (already-rotated) sample points. For
    each col anchor x position, find the (y) where the lane crosses
    that x in the rotated/resized image.
    """
    n = len(col_anchors_norm)
    bins = np.full(n, -1, dtype=np.int64)
    fys = np.full(n, -1.0, dtype=np.float32)
    finite = np.isfinite(ys_res) & np.isfinite(xs_res)
    if not finite.any():
        return bins, fys
    xs_norm = xs_res[finite] / max(float(input_w - 1), 1.0)
    ys_norm = ys_res[finite] / max(float(input_h - 1), 1.0)
    for j, xa in enumerate(col_anchors_norm):
        d = np.abs(xs_norm - xa)
        k = int(d.argmin())
        if d[k] > x_tol_norm:
            continue
        y = float(ys_norm[k])
        bins[j] = int(round(y * (num_cell_col - 1)))
        fys[j] = y
    return bins, fys


class UFLDv2FaithfulDataset(Dataset):
    """Produces samples in the upstream UFLDv2 `inference_culane_tusimple`
    expected format. The training loop, loss dict, and `parsingNet`
    consume these unchanged.
    """

    def __init__(self,
                 manifest_path: str,
                 split: str,
                 input_h: int = 800,
                 input_w: int = 320,
                 num_cls_row: int = 56,
                 num_cls_col: int = 41,
                 num_cell_row: int = 100,
                 num_cell_col: int = 100,
                 num_lanes: int = 2,
                 rotation_deg: float = 0.0,
                 paint_seg: bool = False,
                 seg_stride: int = 8,
                 seg_line_width: int = 8):
        with open(manifest_path, 'r') as f:
            m = json.load(f)
        self.entries = list(m['splits'].get(split, []))
        self.input_h = int(input_h)
        self.input_w = int(input_w)
        self.num_cls_row = int(num_cls_row)
        self.num_cls_col = int(num_cls_col)
        self.num_cell_row = int(num_cell_row)
        self.num_cell_col = int(num_cell_col)
        self.num_lanes = int(num_lanes)
        # Uniform augmentation across all faithful methods. `rotation_deg`
        # > 0 (train only) → ±deg random rotation of image + labels.
        self.rotation_deg = float(rotation_deg)
        self.paint_seg = bool(paint_seg)
        self.seg_stride = int(seg_stride)
        self.seg_line_width = int(seg_line_width)
        # Linearly spaced anchors across full image extent — see header
        # comment for the justification vs upstream's bottom-78% anchors.
        self.row_anchors_norm = np.linspace(
            0.0, 1.0, self.num_cls_row, dtype=np.float64)
        self.col_anchors_norm = np.linspace(
            0.0, 1.0, self.num_cls_col, dtype=np.float64)

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> dict:
        e = self.entries[idx]
        img = cv2.imread(e['image_path'], cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(e['image_path'])
        H_orig, W_orig = img.shape[:2]
        img_r = cv2.resize(img, (self.input_w, self.input_h),
                           interpolation=cv2.INTER_AREA)

        gL, gR, _, _ = _load_gt_pair(e['clean_gt_path'])
        lanes = [gL, gR][:self.num_lanes]

        # Sample rotation (train only; rotation_deg=0 → no-op).
        angle = sample_rotation_angle(self.rotation_deg)
        if angle != 0.0:
            img_r = rotate_image(img_r, angle)

        # Build per-lane (ys, xs) sample points in resized image coords,
        # rotating both image and labels by the same angle.
        per_lane_samples = []
        for gt in lanes:
            ys_orig_arr = np.linspace(0.0, H_orig - 1, 200, dtype=np.float64)
            if gt.has_marking:
                xs_orig_arr = gt.eval_x_at(ys_orig_arr, W_orig,
                                            clamp_to_frame=False)
                # Map to resized coords
                ys_res = ys_orig_arr * (self.input_h - 1) / max(float(H_orig - 1), 1.0)
                xs_res = xs_orig_arr * (self.input_w - 1) / max(float(W_orig - 1), 1.0)
                # Rotate by `angle` around (input_w/2, input_h/2)
                from lcms_lane_benchmark.literature._uniform_aug import (
                    rotate_polynomial_samples,
                )
                ys_res, xs_res = rotate_polynomial_samples(
                    ys_res, xs_res, angle, self.input_h, self.input_w)
            else:
                ys_res = np.full(200, np.nan, dtype=np.float64)
                xs_res = np.full(200, np.nan, dtype=np.float64)
            per_lane_samples.append((ys_res, xs_res))

        img_3 = np.stack([img_r, img_r, img_r], axis=-1).astype(np.float32) / 255.0
        img_3 = (img_3 - _IMAGENET_MEAN) / _IMAGENET_STD
        images = torch.from_numpy(
            np.ascontiguousarray(img_3.transpose(2, 0, 1))).float()

        # (num_cls_row, num_lanes)
        labels_row = np.full((self.num_cls_row, self.num_lanes),
                              -1, dtype=np.int64)
        labels_row_float = np.full((self.num_cls_row, self.num_lanes),
                                    -1.0, dtype=np.float32)
        # (num_cls_col, num_lanes)
        labels_col = np.full((self.num_cls_col, self.num_lanes),
                              -1, dtype=np.int64)
        labels_col_float = np.full((self.num_cls_col, self.num_lanes),
                                    -1.0, dtype=np.float32)

        for lane_idx, (ys_res, xs_res) in enumerate(per_lane_samples):
            rb, rf = _row_labels_from_samples(
                ys_res, xs_res, self.input_h, self.input_w,
                self.row_anchors_norm, self.num_cell_row)
            labels_row[:, lane_idx] = rb
            labels_row_float[:, lane_idx] = rf
            cb, cf = _col_labels_from_samples(
                ys_res, xs_res, self.input_h, self.input_w,
                self.col_anchors_norm, self.num_cell_col)
            labels_col[:, lane_idx] = cb
            labels_col_float[:, lane_idx] = cf

        out = {
            'images': images,
            'labels_row': torch.from_numpy(labels_row),
            'labels_col': torch.from_numpy(labels_col),
            'labels_row_float': torch.from_numpy(labels_row_float),
            'labels_col_float': torch.from_numpy(labels_col_float),
            'image_h_orig': H_orig,
            'image_w_orig': W_orig,
            'stem': e.get('stem', ''),
            'project': e.get('project', 'unknown'),
        }
        if self.paint_seg:
            seg = _paint_seg_label(per_lane_samples, self.input_h,
                                    self.input_w,
                                    seg_stride=self.seg_stride,
                                    line_width=self.seg_line_width)
            out['seg_label'] = torch.from_numpy(seg)
        return out


def collate(batch: list) -> dict:
    out = {}
    for k in batch[0]:
        if isinstance(batch[0][k], torch.Tensor):
            out[k] = torch.stack([b[k] for b in batch], dim=0)
        else:
            out[k] = [b[k] for b in batch]
    return out
