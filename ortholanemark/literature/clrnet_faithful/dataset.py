"""Dataset shim feeding our clean_gt sidecars to the vendored CLRNet.

Sample format mirrors what upstream's `GenerateLaneLine` process emits
(the format consumed by `CLRHead.loss`):

    img       : float32 tensor (3, 320, 800), raw /255 (upstream CLRNet
                normalizes by /255 only; the config's img_norm block is
                unused in the vendored pipeline version).
    lane_line : float32 tensor (max_lanes=2, 78) — per lane:
                [neg, pos, start_y/n_strips, start_x_px, theta_norm,
                 n_inside, x@offsets_ys...(px)], invalid slots -1e5.
    seg       : int64 tensor (320, 800), {0=bg, 1=left, 2=right} —
                auxiliary segmentation target (8 px line width, the
                same rasterization convention as the SCNN shim).

The lane encoding functions `_sample_lane` / `_transform_annotation`
are upstream-verbatim ports of
`clrnet/datasets/process/generate_lane_line.py` (spline interpolation
inside the annotated domain, linear extrapolation to the image bottom,
mean-theta). Only imgaug is replaced: augmentation is the
project-uniform Rotation(±2°) applied identically to image, lane
points, and seg mask (same policy as every other faithful baseline —
see FIDELITY_REVIEW.md).
"""

from __future__ import annotations

import json
import math

import cv2
import numpy as np
import torch
from scipy.interpolate import InterpolatedUnivariateSpline
from torch.utils.data import Dataset

from ortholanemark.data.lane_gt import LaneGeometricTruth


IMG_W, IMG_H = 800, 320
NUM_POINTS = 72
N_STRIPS = NUM_POINTS - 1
STRIP_SIZE = IMG_H / N_STRIPS
OFFSETS_YS = np.arange(IMG_H, -1, -STRIP_SIZE)
MAX_LANES = 2


def set_resolution(img_w: int, img_h: int) -> None:
    """Set the CLRNet input resolution (landscape convention: width=img_w,
    height=img_h). Mutates module globals used by the encoding helpers;
    safe because training runs single-process (num_workers=0). Call before
    constructing the dataset for a non-default resolution (e.g. the
    resolution ablation)."""
    global IMG_W, IMG_H, STRIP_SIZE, OFFSETS_YS
    IMG_W, IMG_H = int(img_w), int(img_h)
    STRIP_SIZE = IMG_H / N_STRIPS
    OFFSETS_YS = np.arange(IMG_H, -1, -STRIP_SIZE)


def _load_gt_pair(clean_gt_path: str):
    with open(clean_gt_path, 'r', encoding='utf-8') as f:
        d = json.load(f)
    H = int(d['image_size']['height'])
    W = int(d['image_size']['width'])
    gL = LaneGeometricTruth.from_clean_gt_dict(d['left_lane'])
    gR = LaneGeometricTruth.from_clean_gt_dict(d['right_lane'])
    return gL, gR, H, W


def _lane_points_input_space(gt, H_orig: int, W_orig: int) -> np.ndarray:
    """GT polynomial -> (N, 2) in-frame [x, y] points in input (800x320)
    coords, sorted bottom-to-top (y descending)."""
    if gt is None or not gt.has_marking:
        return np.zeros((0, 2), dtype=np.float64)
    ys_in = np.arange(IMG_H, dtype=np.float64)
    ys_orig = ys_in * (H_orig - 1) / max(IMG_H - 1, 1)
    xs_orig = gt.eval_x_at(ys_orig, W_orig, clamp_to_frame=False)
    xs_in = xs_orig * (IMG_W - 1) / max(W_orig - 1, 1)
    ok = np.isfinite(xs_in) & (xs_in >= 0) & (xs_in < IMG_W)
    pts = np.stack([xs_in[ok], ys_in[ok]], axis=1)
    return pts[np.argsort(-pts[:, 1])]


def _rotate_points(pts: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Apply a 2x3 affine to (N, 2) points, clip to frame, resort, dedupe y."""
    if len(pts) == 0:
        return pts
    ones = np.ones((len(pts), 1))
    out = (np.hstack([pts, ones]) @ M.T)
    ok = ((out[:, 0] >= 0) & (out[:, 0] < IMG_W)
          & (out[:, 1] >= 0) & (out[:, 1] < IMG_H))
    out = out[ok]
    if len(out) == 0:
        return out
    out = out[np.argsort(-out[:, 1])]
    # strictly decreasing y required by _sample_lane
    keep = np.concatenate([[True], np.diff(out[:, 1]) < 0])
    return out[keep]


# ---------------------------------------------------------------------------
# Upstream-verbatim lane encoding (generate_lane_line.py)
# ---------------------------------------------------------------------------
def _sample_lane(points: np.ndarray, sample_ys: np.ndarray):
    points = np.array(points)
    if not np.all(points[1:, 1] < points[:-1, 1]):
        raise AssertionError('Annotation points have to be sorted')
    x, y = points[:, 0], points[:, 1]

    assert len(points) > 1
    interp = InterpolatedUnivariateSpline(y[::-1], x[::-1],
                                          k=min(3, len(points) - 1))
    domain_min_y = y.min()
    domain_max_y = y.max()
    sample_ys_inside_domain = sample_ys[(sample_ys >= domain_min_y)
                                        & (sample_ys <= domain_max_y)]
    assert len(sample_ys_inside_domain) > 0
    interp_xs = interp(sample_ys_inside_domain)

    # extrapolate to the bottom of the image with a straight line from the
    # 2 points closest to the bottom
    two_closest_points = points[:2]
    extrap = np.polyfit(two_closest_points[:, 1], two_closest_points[:, 0],
                        deg=1)
    extrap_ys = sample_ys[sample_ys > domain_max_y]
    extrap_xs = np.polyval(extrap, extrap_ys)
    all_xs = np.hstack((extrap_xs, interp_xs))

    inside_mask = (all_xs >= 0) & (all_xs < IMG_W)
    xs_inside_image = all_xs[inside_mask]
    xs_outside_image = all_xs[~inside_mask]
    return xs_outside_image, xs_inside_image


def _transform_annotation(lanes_points: list) -> np.ndarray:
    """Port of GenerateLaneLine.transform_annotation for already-prepared
    input-space point lists."""
    lanes = np.ones((MAX_LANES, 2 + 1 + 1 + 2 + NUM_POINTS),
                    dtype=np.float32) * -1e5
    lanes[:, 0] = 1
    lanes[:, 1] = 0
    for lane_idx, lane in enumerate(lanes_points):
        if lane_idx >= MAX_LANES:
            break
        if len(lane) <= 1:
            continue
        try:
            xs_outside_image, xs_inside_image = _sample_lane(
                lane, OFFSETS_YS)
        except AssertionError:
            continue
        if len(xs_inside_image) <= 1:
            continue
        all_xs = np.hstack((xs_outside_image, xs_inside_image))
        lanes[lane_idx, 0] = 0
        lanes[lane_idx, 1] = 1
        lanes[lane_idx, 2] = len(xs_outside_image) / N_STRIPS
        lanes[lane_idx, 3] = xs_inside_image[0]

        thetas = []
        for i in range(1, len(xs_inside_image)):
            theta = math.atan(
                i * STRIP_SIZE /
                (xs_inside_image[i] - xs_inside_image[0] + 1e-5)) / math.pi
            theta = theta if theta > 0 else 1 - abs(theta)
            thetas.append(theta)
        theta_far = sum(thetas) / len(thetas)

        lanes[lane_idx, 4] = theta_far
        lanes[lane_idx, 5] = len(xs_inside_image)
        lanes[lane_idx, 6:6 + len(all_xs)] = all_xs
    return lanes


def _paint_seg(lanes_points: list, line_width: int = 8) -> np.ndarray:
    seg = np.zeros((IMG_H, IMG_W), dtype=np.int64)
    half = line_width // 2
    for class_id, pts in enumerate(lanes_points, start=1):
        for x, y in pts:
            yi = int(round(y))
            xi = int(round(x))
            if 0 <= yi < IMG_H and 0 <= xi < IMG_W:
                seg[yi, max(0, xi - half):min(IMG_W, xi + half + 1)] = \
                    class_id
    return seg


class CLRNetFaithfulDataset(Dataset):
    def __init__(self, manifest_path: str, split: str,
                 rotation_deg: float = 0.0,
                 img_w: int = 800, img_h: int = 320):
        # Set module resolution (default 800x320 preserves prior behavior).
        set_resolution(img_w, img_h)
        with open(manifest_path, 'r') as f:
            m = json.load(f)
        self.entries = list(m['splits'].get(split, []))
        self.rotation_deg = float(rotation_deg)

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx: int) -> dict:
        e = self.entries[idx]
        img = cv2.imread(e['image_path'], cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(e['image_path'])
        H_orig, W_orig = img.shape[:2]
        img_r = cv2.resize(img, (IMG_W, IMG_H),
                           interpolation=cv2.INTER_AREA)

        gL, gR, H_o, W_o = _load_gt_pair(e['clean_gt_path'])
        pts_L = _lane_points_input_space(gL, H_o, W_o)
        pts_R = _lane_points_input_space(gR, H_o, W_o)

        if self.rotation_deg > 0.0:
            ang = float(np.random.uniform(-self.rotation_deg,
                                          self.rotation_deg))
            center = (IMG_W / 2.0, IMG_H / 2.0)
            M = cv2.getRotationMatrix2D(center, ang, 1.0)
            img_r = cv2.warpAffine(img_r, M, (IMG_W, IMG_H),
                                   flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_REFLECT)
            pts_L = _rotate_points(pts_L, M)
            pts_R = _rotate_points(pts_R, M)

        # order matters for seg classes: index 0 -> class 1 (left),
        # index 1 -> class 2 (right). lane_line slots may be empty.
        lanes_points = [pts_L, pts_R]
        lane_line = _transform_annotation(
            [p for p in lanes_points if len(p) > 1])
        seg = _paint_seg(lanes_points)

        img_3 = np.stack([img_r, img_r, img_r],
                         axis=-1).astype(np.float32) / 255.0
        img_t = torch.from_numpy(
            np.ascontiguousarray(img_3.transpose(2, 0, 1))).float()

        return {
            'img': img_t,
            'lane_line': torch.from_numpy(lane_line).float(),
            'seg': torch.from_numpy(seg).long(),
            'img_name': e['image_path'],
            'stem': e.get('stem', ''),
            'project': e.get('project', 'unknown'),
            'image_h_orig': H_orig,
            'image_w_orig': W_orig,
        }


def collate(batch: list) -> dict:
    out = {}
    for k in batch[0].keys():
        if isinstance(batch[0][k], torch.Tensor):
            out[k] = torch.stack([b[k] for b in batch], dim=0)
        else:
            out[k] = [b[k] for b in batch]
    return out
