"""Inference adapter for the plain U-Net baseline -> project per-row format.

Registers as the literature method `unet_seg`.

Decoding mirrors the `scnn` adapter exactly (softmax → 9x9 blur →
row-wise argmax above prob threshold → polynomial fit), so the comparison
isolates the architecture. The one necessary difference: U-Net has no
existence head, so existence is derived from segmentation support — a side
exists iff the fraction of map rows with a confident lane pixel is at
least `exist_row_frac`.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch

from ortholanemark.literature import (
    BaseLiteratureMethod, default_checkpoint_path, register_method,
)
from ortholanemark.literature.unet_seg.model import UNet


_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
DEFAULT_CKPT = default_checkpoint_path('unet_seg')


def _decode_side(prob_map: np.ndarray, H: int, W: int,
                 prob_threshold: float, exist_row_frac: float,
                 min_lane_rows: int) -> tuple:
    """One lane-class probability map -> (x_curve (H,), exists, support)."""
    prob_map = cv2.blur(prob_map, (9, 9), borderType=cv2.BORDER_REPLICATE)
    ih, iw = prob_map.shape
    row_x = prob_map.argmax(axis=1).astype(np.float64)
    row_prob = prob_map[np.arange(ih), row_x.astype(np.int64)]
    row_idx = np.where(row_prob > prob_threshold)[0]
    support = len(row_idx) / float(ih)
    exists = (support >= exist_row_frac) and (len(row_idx) >= min_lane_rows)
    # Low support may still pass a later tuned presence threshold.
    if len(row_idx) < min_lane_rows:
        return np.full(H, np.nan, dtype=np.float64), False, support
    y_norm = row_idx.astype(np.float64) / max(float(ih - 1), 1.0)
    x_norm = row_x[row_idx] / max(float(iw - 1), 1.0)
    deg = 2 if len(row_idx) >= 5 else 1
    coefs = np.polyfit(y_norm, x_norm, deg)
    y_full = np.arange(H, dtype=np.float64) / max(float(H - 1), 1.0)
    x_pix = np.polyval(coefs, y_full) * float(W - 1)
    off = (x_pix < 0.0) | (x_pix > float(W - 1))
    x_curve = np.where(off, np.nan, x_pix)
    if not np.isfinite(x_curve).any():
        return np.full(H, np.nan, dtype=np.float64), False, support
    return x_curve, bool(exists), support


def _preprocess(image_gray: np.ndarray, input_h: int,
                input_w: int) -> torch.Tensor:
    img_r = cv2.resize(image_gray, (input_w, input_h),
                       interpolation=cv2.INTER_AREA)
    img_3 = np.stack([img_r, img_r, img_r], axis=-1).astype(np.float32) / 255.0
    img_3 = (img_3 - _IMAGENET_MEAN) / _IMAGENET_STD
    return torch.from_numpy(
        np.ascontiguousarray(img_3.transpose(2, 0, 1))).float()


@torch.no_grad()
def predict_batch_with_model(model, image_grays, input_h: int = 800,
                             input_w: int = 320,
                             prob_threshold: float = 0.3,
                             exist_row_frac: float = 0.10,
                             min_lane_rows: int = 5) -> list:
    """Batched inference: one forward over a list of images, then per-image
    decoding. Same contract as scnn.predict_batch_with_model."""
    device = next(model.parameters()).device
    sizes = [img.shape[:2] for img in image_grays]
    x = torch.stack([_preprocess(img, input_h, input_w)
                     for img in image_grays], dim=0).to(device)
    seg_prob_all = torch.softmax(model(x), dim=1).cpu().numpy()

    results = []
    for (H, W), seg_prob in zip(sizes, seg_prob_all):
        x_L, eL, sL = _decode_side(seg_prob[1], H, W, prob_threshold,
                                   exist_row_frac, min_lane_rows)
        x_R, eR, sR = _decode_side(seg_prob[2], H, W, prob_threshold,
                                   exist_row_frac, min_lane_rows)
        results.append({
            'pred_x_L': x_L, 'pred_x_R': x_R,
            'pred_exists_L': eL, 'pred_exists_R': eR,
            # existence confidence = segmentation support fraction (rows
            # with a confident lane pixel); already in [0,1].
            'pred_conf_L': float(min(sL, 1.0)),
            'pred_conf_R': float(min(sR, 1.0)),
        })
    return results


@torch.no_grad()
def predict_with_model(model, image_gray, input_h: int = 800,
                       input_w: int = 320,
                       prob_threshold: float = 0.3,
                       exist_row_frac: float = 0.10,
                       min_lane_rows: int = 5) -> dict:
    t0 = time.perf_counter()
    out = predict_batch_with_model(
        model, [image_gray], input_h=input_h, input_w=input_w,
        prob_threshold=prob_threshold, exist_row_frac=exist_row_frac,
        min_lane_rows=min_lane_rows)[0]
    out['inference_ms'] = (time.perf_counter() - t0) * 1000.0
    return out


@register_method('unet_seg')
class UNetSegMethod(BaseLiteratureMethod):
    """Plain U-Net segmentation baseline with the SCNN-style adapter."""

    def __init__(self,
                 ckpt_path: Optional[str] = None,
                 input_h: int = 800,
                 input_w: int = 320,
                 prob_threshold: float = 0.3,
                 exist_row_frac: float = 0.10,
                 min_lane_rows: int = 5,
                 device: Optional[str] = None):
        self.ckpt_path = Path(ckpt_path or DEFAULT_CKPT)
        self.input_h = int(input_h)
        self.input_w = int(input_w)
        self.prob_threshold = float(prob_threshold)
        self.exist_row_frac = float(exist_row_frac)
        self.min_lane_rows = int(min_lane_rows)
        self.device = device or (
            'cuda' if torch.cuda.is_available() else 'cpu')
        self._model: Optional[UNet] = None

    def _lazy_load(self) -> None:
        if self._model is not None:
            return
        if not self.ckpt_path.exists():
            raise FileNotFoundError(
                f'unet_seg checkpoint not found: {self.ckpt_path}')
        ckpt = torch.load(self.ckpt_path, map_location=self.device,
                          weights_only=False)
        cfg = ckpt.get('config', {})
        self.input_h = int(cfg.get('input_h', self.input_h))
        self.input_w = int(cfg.get('input_w', self.input_w))
        m = UNet(in_channels=3, num_classes=3,
                 base_ch=int(cfg.get('base_ch', 64)))
        m.load_state_dict(ckpt['model'])
        m.to(self.device).eval()
        self._model = m

    @torch.no_grad()
    def predict_image(self, image_gray: np.ndarray) -> dict:
        t0 = time.perf_counter()
        self._lazy_load()
        out = predict_batch_with_model(
            self._model, [image_gray],
            input_h=self.input_h, input_w=self.input_w,
            prob_threshold=self.prob_threshold,
            exist_row_frac=self.exist_row_frac,
            min_lane_rows=self.min_lane_rows)[0]
        out['inference_ms'] = (time.perf_counter() - t0) * 1000.0
        return out
