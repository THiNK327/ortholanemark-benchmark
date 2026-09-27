"""Shared deep-learning baseline infrastructure.

All four Tier-2 methods (PolyLaneNet, SCNN, LaneATT, UFLDv2) share:
  - A dataset that reads `manifest_paper.<split>`, loads each image as
    1-channel grayscale, resizes to (input_h, input_w), and produces
    per-side targets in **normalized resized coordinates** (y_norm,
    x_norm ∈ [0, 1]).
  - A training loop (AdamW + cosine schedule + history.json + best.pth
    + last.pth).
  - An inference helper that scales method outputs back to the original
    image's per-row x array suitable for
    `per_image_boundary_errors_full`.

The polynomial targets are fit by sampling the GT polynomial at
`n_anchors` rows in the resized image and least-squares fitting a
degree-2 polynomial. This guarantees the loss targets are well-defined
even for sides where the original GT polynomial is degree 1.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from ortholanemark.data.lane_gt import LaneGeometricTruth


# ----------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------

@dataclass
class DeepLaneSample:
    """All deep baselines see the same per-sample shape; methods then
    pick the subset of targets they actually consume."""

    image_path: str
    project: str
    stem: str
    image_h_orig: int
    image_w_orig: int

    # Resized inputs.
    image_tensor: torch.Tensor                  # (1, input_h, input_w)

    # Per-side existence (crop-aware: True iff GT has poly AND any of
    # the anchor rows produces an in-frame x value in the resized image).
    exist_L: bool
    exist_R: bool

    # Per-side polynomial coefficients in NORMALIZED resized coords:
    #   y_norm ∈ [0, 1], x_norm ∈ [0, 1]
    # numpy polyfit order: [a2, a1, a0] for x_norm = a2*y² + a1*y + a0.
    poly_norm_L: torch.Tensor                   # (3,) float32
    poly_norm_R: torch.Tensor                   # (3,) float32

    # Per-row x targets in normalized coords at a fixed anchor grid
    # y_anchor_norm = linspace(0, 1, n_anchors). NaN where off-frame.
    x_at_anchor_L: torch.Tensor                 # (n_anchors,) float32
    x_at_anchor_R: torch.Tensor                 # (n_anchors,) float32
    valid_anchor_L: torch.Tensor                # (n_anchors,) bool
    valid_anchor_R: torch.Tensor                # (n_anchors,) bool


def _fit_poly_norm_from_gt(gt: Optional[LaneGeometricTruth],
                            H_orig: int, W_orig: int,
                            input_h: int, input_w: int,
                            n_anchors: int = 64
                            ) -> tuple[np.ndarray, np.ndarray,
                                       np.ndarray, bool]:
    """For a given side's GT polynomial, evaluate it at `n_anchors`
    rows spanning the resized image height, normalize, and least-squares
    fit a degree-2 polynomial in normalized coords.

    Returns (poly_coefs_norm, x_at_anchor_norm, valid_anchor, exists_in_resized).
    poly_coefs_norm is always length 3 (a2, a1, a0); zeros when there is
    no in-frame anchor row.
    """
    y_anchor_norm = np.linspace(0.0, 1.0, n_anchors)
    poly_coefs = np.zeros(3, dtype=np.float64)
    x_at_anchor = np.full(n_anchors, np.nan, dtype=np.float64)
    valid = np.zeros(n_anchors, dtype=bool)

    if gt is None or not gt.has_marking:
        return poly_coefs, x_at_anchor, valid, False

    # Anchor y in original image rows.
    y_orig = y_anchor_norm * (H_orig - 1)
    x_orig = gt.eval_x_at(y_orig, W_orig, clamp_to_frame=False)
    in_frame = np.isfinite(x_orig)
    if not in_frame.any():
        return poly_coefs, x_at_anchor, valid, False

    # Normalise.
    x_norm = x_orig / float(W_orig - 1)
    x_at_anchor = np.where(in_frame, x_norm, np.nan)
    valid = in_frame

    # LS-fit a deg-2 poly through valid (y_anchor_norm, x_norm) pairs.
    ys = y_anchor_norm[in_frame]
    xs = x_norm[in_frame]
    if len(ys) >= 3:
        coefs = np.polyfit(ys, xs, 2)
    else:
        # 2 points → degree-1; 1 point → constant.
        if len(ys) >= 2:
            c = np.polyfit(ys, xs, 1)
            coefs = np.array([0.0, c[0], c[1]], dtype=np.float64)
        else:
            coefs = np.array([0.0, 0.0, float(xs[0])], dtype=np.float64)
    poly_coefs = coefs.astype(np.float64)
    return poly_coefs, x_at_anchor, valid, True


def _load_gt_pair(clean_gt_path: str
                  ) -> tuple[LaneGeometricTruth, LaneGeometricTruth, int, int]:
    with open(clean_gt_path, 'r', encoding='utf-8') as f:
        d = json.load(f)
    H = int(d['image_size']['height'])
    W = int(d['image_size']['width'])
    gt_L = LaneGeometricTruth.from_clean_gt_dict(d['left_lane'])
    gt_R = LaneGeometricTruth.from_clean_gt_dict(d['right_lane'])
    if gt_L is None:
        gt_L = LaneGeometricTruth(False, np.zeros(1), 0, None, [])
    if gt_R is None:
        gt_R = LaneGeometricTruth(False, np.zeros(1), 0, None, [])
    return gt_L, gt_R, H, W


class DeepLaneDataset(Dataset):
    """Reads a manifest split, returns per-image tensors + per-side poly
    targets in normalized resized coords. Same shape every method uses.
    """

    def __init__(self, manifest_path: str, split: str,
                 input_h: int = 480, input_w: int = 200,
                 n_anchors: int = 64):
        with open(manifest_path, 'r') as f:
            m = json.load(f)
        self.entries = list(m['splits'].get(split, []))
        self.input_h = int(input_h)
        self.input_w = int(input_w)
        self.n_anchors = int(n_anchors)

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> dict:
        e = self.entries[idx]
        gt_L, gt_R, H_orig, W_orig = _load_gt_pair(e['clean_gt_path'])

        img = cv2.imread(e['image_path'], cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(e['image_path'])
        img_resized = cv2.resize(img, (self.input_w, self.input_h),
                                 interpolation=cv2.INTER_AREA)
        x = torch.from_numpy(img_resized).float().unsqueeze(0) / 255.0

        poly_L, xa_L, va_L, eL = _fit_poly_norm_from_gt(
            gt_L, H_orig, W_orig, self.input_h, self.input_w,
            n_anchors=self.n_anchors)
        poly_R, xa_R, va_R, eR = _fit_poly_norm_from_gt(
            gt_R, H_orig, W_orig, self.input_h, self.input_w,
            n_anchors=self.n_anchors)

        return {
            'image': x,
            'image_path': e['image_path'],
            'project': e.get('project', 'unknown'),
            'stem': e.get('stem', ''),
            'image_h_orig': H_orig,
            'image_w_orig': W_orig,
            'exist_L': torch.tensor(eL, dtype=torch.float32),
            'exist_R': torch.tensor(eR, dtype=torch.float32),
            'poly_norm_L': torch.from_numpy(poly_L).float(),
            'poly_norm_R': torch.from_numpy(poly_R).float(),
            'x_at_anchor_L': torch.from_numpy(
                np.nan_to_num(xa_L, nan=0.0)).float(),
            'x_at_anchor_R': torch.from_numpy(
                np.nan_to_num(xa_R, nan=0.0)).float(),
            'valid_anchor_L': torch.from_numpy(va_L),
            'valid_anchor_R': torch.from_numpy(va_R),
        }


def _collate(batch: list[dict]) -> dict:
    """Stacks tensors, keeps non-tensor metadata as lists."""
    out: dict = {}
    keys = batch[0].keys()
    for k in keys:
        if isinstance(batch[0][k], torch.Tensor):
            out[k] = torch.stack([b[k] for b in batch], dim=0)
        else:
            out[k] = [b[k] for b in batch]
    return out


# ----------------------------------------------------------------------
# Training loop (generic)
# ----------------------------------------------------------------------

def train_deep_baseline(
    *,
    model: nn.Module,
    loss_fn,                          # (out, batch) -> (loss, log_dict)
    manifest_path: str,
    save_dir: str,
    epochs: int = 30,
    batch_size: int = 8,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    num_workers: int = 0,
    input_h: int = 480,
    input_w: int = 200,
    n_anchors: int = 64,
    device: str = 'cuda',
    val_metric_fn=None,               # (out, batch) -> scalar (lower better)
):
    """Train a deep baseline. Saves best.pth + last.pth + history.json."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    train_ds = DeepLaneDataset(manifest_path, 'train', input_h, input_w, n_anchors)
    val_ds = DeepLaneDataset(manifest_path, 'val', input_h, input_w, n_anchors)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, collate_fn=_collate,
                          drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, collate_fn=_collate)

    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr,
                            weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    history: list[dict] = []
    best_val: float = float('inf')

    for ep in range(1, epochs + 1):
        model.train()
        tr_loss = 0.0
        n_tr = 0
        for batch in tqdm(train_dl, desc=f'ep {ep}/{epochs} train',
                          leave=False):
            batch_dev = _to_device(batch, device)
            out = model(batch_dev['image'])
            loss, _ = loss_fn(out, batch_dev)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tr_loss += float(loss.detach().cpu()) * batch_dev['image'].size(0)
            n_tr += batch_dev['image'].size(0)
        sched.step()
        tr_loss /= max(n_tr, 1)

        model.eval()
        val_loss = 0.0
        val_metric = 0.0
        n_val = 0
        n_val_metric = 0
        with torch.no_grad():
            for batch in tqdm(val_dl, desc=f'ep {ep}/{epochs} val',
                              leave=False):
                batch_dev = _to_device(batch, device)
                out = model(batch_dev['image'])
                loss, _ = loss_fn(out, batch_dev)
                val_loss += float(loss.cpu()) * batch_dev['image'].size(0)
                n_val += batch_dev['image'].size(0)
                if val_metric_fn is not None:
                    m = val_metric_fn(out, batch_dev)
                    if m is not None and np.isfinite(float(m)):
                        val_metric += float(m) * batch_dev['image'].size(0)
                        n_val_metric += batch_dev['image'].size(0)
        val_loss /= max(n_val, 1)
        val_metric_avg = (val_metric / n_val_metric
                          if n_val_metric > 0 else float('nan'))

        record = {
            'epoch': ep,
            'train_loss': tr_loss,
            'val_loss': val_loss,
            'val_metric': val_metric_avg,
            'lr': float(opt.param_groups[0]['lr']),
        }
        history.append(record)
        print(f"[ep {ep:3d}] train={tr_loss:.4f}  val={val_loss:.4f}  "
              f"val_metric={val_metric_avg:.4f}  lr={record['lr']:.6f}")

        # Save best (by val_metric if provided, else val_loss).
        selector = val_metric_avg if (val_metric_fn is not None
                                       and np.isfinite(val_metric_avg)) \
                                  else val_loss
        if selector < best_val:
            best_val = selector
            torch.save({
                'model': model.state_dict(),
                'epoch': ep,
                'val_metric': val_metric_avg,
                'val_loss': val_loss,
                'config': {
                    'input_h': input_h, 'input_w': input_w,
                    'n_anchors': n_anchors,
                },
            }, save_dir / 'best.pth')
        torch.save({
            'model': model.state_dict(),
            'epoch': ep,
            'val_loss': val_loss,
            'val_metric': val_metric_avg,
            'config': {
                'input_h': input_h, 'input_w': input_w,
                'n_anchors': n_anchors,
            },
        }, save_dir / 'last.pth')

        with open(save_dir / 'history.json', 'w') as f:
            json.dump({'history': history}, f, indent=2)

    print(f"Training done. best_val={best_val:.4f}")
    return history


def _to_device(batch: dict, device: str) -> dict:
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


# ----------------------------------------------------------------------
# Inference helpers (shared)
# ----------------------------------------------------------------------

def poly_norm_to_per_row_x(poly_norm: np.ndarray,
                            H_orig: int, W_orig: int) -> np.ndarray:
    """Evaluate a deg-2 polynomial defined in normalized resized coords
    at every original-image row, return per-row x in original pixel
    coords. NaN where the polynomial extrapolates off the original frame.
    """
    y_norm = np.arange(H_orig, dtype=np.float64) / max(float(H_orig - 1), 1.0)
    x_norm = np.polyval(poly_norm, y_norm)
    x_pix = x_norm * float(W_orig - 1)
    off = (x_pix < 0.0) | (x_pix > float(W_orig - 1))
    return np.where(off, np.nan, x_pix)
