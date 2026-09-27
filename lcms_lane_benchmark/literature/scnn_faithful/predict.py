"""Inference adapter for faithful SCNN output -> project per-row format.

Registers as the literature method `scnn_faithful`.

Inference mirrors upstream `utils/prob2lines/getLane.py` where possible:
  1. Forward -> segmentation logits and lane-existence probabilities.
  2. Softmax the segmentation logits. For each lane whose existence
     score is above 0.5, smooth that lane probability map with a 9x9
     blur and retain row-wise argmax x positions only when the row max
     probability exceeds 0.3.
  3. Fit a degree-2 polynomial through the retained points and evaluate
     it at every original-image row. This polynomial fit is the project
     adapter from SCNN's probability map to our per-row metric format.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch

from lcms_lane_benchmark.literature import (
    BaseLiteratureMethod, register_method,
)
from lcms_lane_benchmark.literature.scnn_faithful.model import SCNN


_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
DEFAULT_CKPT = 'lcms_lane_benchmark/runs_reselect_train/scnn_faithful/best_gra.pth'


@torch.no_grad()
def predict_batch_with_model(model, image_grays, input_h: int = 800,
                               input_w: int = 320, num_lanes: int = 2,
                               exist_threshold: float = 0.5,
                               prob_threshold: float = 0.3,
                               min_lane_rows: int = 5) -> list:
    """Batched inference: single model.forward over a list of images,
    then per-image post-processing (softmax + smoothing + polyfit)."""
    device = next(model.parameters()).device
    B = len(image_grays)
    tensors = []
    sizes = []
    for img in image_grays:
        H, W = img.shape[:2]
        sizes.append((H, W))
        img_r = cv2.resize(img, (input_w, input_h),
                           interpolation=cv2.INTER_AREA)
        img_3 = np.stack([img_r, img_r, img_r], axis=-1).astype(np.float32) / 255.0
        img_3 = (img_3 - _IMAGENET_MEAN) / _IMAGENET_STD
        tensors.append(torch.from_numpy(
            np.ascontiguousarray(img_3.transpose(2, 0, 1))).float())
    x = torch.stack(tensors, dim=0).to(device)

    seg_pred, exist_pred, _, _, _ = model(x, None, None)
    seg_prob_all = torch.softmax(seg_pred, dim=1).cpu().numpy()  # (B, C, h, w)
    exists_all = exist_pred.cpu().numpy()                          # (B, num_lanes)

    results = []
    for i in range(B):
        H, W = sizes[i]
        seg_prob = seg_prob_all[i]
        exists = exists_all[i]
        eL = bool(exists[0] > exist_threshold)
        eR = bool(exists[1] > exist_threshold) if num_lanes >= 2 else False
        # Geometry must survive a later validation-selected presence threshold.
        x_L = _extract_per_row_x_helper(seg_prob, 1, H, W, prob_threshold, min_lane_rows)
        x_R = (_extract_per_row_x_helper(seg_prob, 2, H, W, prob_threshold, min_lane_rows)
               if num_lanes >= 2 else np.full(H, np.nan, dtype=np.float64))
        if eL and not np.isfinite(x_L).any():
            eL = False
        if eR and not np.isfinite(x_R).any():
            eR = False
        results.append({
            'pred_x_L': x_L, 'pred_x_R': x_R,
            'pred_exists_L': eL, 'pred_exists_R': eR,
            'pred_conf_L': float(exists[0]),
            'pred_conf_R': float(exists[1]) if num_lanes >= 2 else 0.0,
        })
    return results


@torch.no_grad()
def predict_with_model(model, image_gray, input_h: int = 800, input_w: int = 320, num_lanes: int = 2, exist_threshold: float = 0.5, prob_threshold: float = 0.3, min_lane_rows: int = 5) -> dict:
    """Use the same decoder for individual and batched predictions."""
    t0 = time.perf_counter()
    out = predict_batch_with_model(model, [image_gray], input_h=input_h, input_w=input_w, num_lanes=num_lanes, exist_threshold=exist_threshold, prob_threshold=prob_threshold, min_lane_rows=min_lane_rows)[0]
    out['inference_ms'] = (time.perf_counter() - t0) * 1000.0
    return out


def _extract_per_row_x_helper(seg_prob: np.ndarray, class_idx: int,
                               H: int, W: int, prob_threshold: float,
                               min_lane_rows: int) -> np.ndarray:
    """Mirror of `SCNNFaithfulMethod._extract_per_row_x`. Module-level so
    `predict_with_model` doesn't need a class instance.
    """
    prob_map = seg_prob[class_idx]
    prob_map = cv2.blur(prob_map, (9, 9), borderType=cv2.BORDER_REPLICATE)
    ih, iw = prob_map.shape
    row_x = prob_map.argmax(axis=1).astype(np.float64)
    row_prob = prob_map[np.arange(ih), row_x.astype(np.int64)]
    row_idx = np.where(row_prob > prob_threshold)[0]
    if len(row_idx) < min_lane_rows:
        return np.full(H, np.nan, dtype=np.float64)
    y_norm = row_idx.astype(np.float64) / max(float(ih - 1), 1.0)
    x_norm = row_x[row_idx] / max(float(iw - 1), 1.0)
    deg = 2 if len(row_idx) >= 5 else 1
    coefs = np.polyfit(y_norm, x_norm, deg)
    y_full = np.arange(H, dtype=np.float64) / max(float(H - 1), 1.0)
    xn = np.polyval(coefs, y_full)
    x_pix = xn * float(W - 1)
    off = (x_pix < 0.0) | (x_pix > float(W - 1))
    return np.where(off, np.nan, x_pix)


@register_method('scnn_faithful')
class SCNNFaithfulMethod(BaseLiteratureMethod):
    """Paper-faithful SCNN inference with project output adaptation."""

    def __init__(self,
                 ckpt_path: Optional[str] = None,
                 input_h: int = 800,
                 input_w: int = 320,
                 num_lanes: int = 2,
                 exist_threshold: float = 0.5,
                 prob_threshold: float = 0.3,
                 min_lane_rows: int = 5,
                 device: Optional[str] = None):
        self.ckpt_path = Path(ckpt_path or DEFAULT_CKPT)
        self.input_h = int(input_h)
        self.input_w = int(input_w)
        self.num_lanes = int(num_lanes)
        self.exist_threshold = float(exist_threshold)
        self.prob_threshold = float(prob_threshold)
        self.min_lane_rows = int(min_lane_rows)
        self.device = device or (
            'cuda' if torch.cuda.is_available() else 'cpu')
        self._model: Optional[SCNN] = None

    def _lazy_load(self) -> None:
        if self._model is not None:
            return
        if not self.ckpt_path.exists():
            raise FileNotFoundError(
                f'scnn_faithful checkpoint not found: {self.ckpt_path}')
        ckpt = torch.load(self.ckpt_path, map_location=self.device,
                          weights_only=False)
        cfg = ckpt.get('config', {})
        self.input_h = int(cfg.get('input_h', self.input_h))
        self.input_w = int(cfg.get('input_w', self.input_w))
        self.num_lanes = int(cfg.get('num_lanes', self.num_lanes))
        m = SCNN(input_size=(self.input_w, self.input_h),
                 num_lanes=self.num_lanes, pretrained=False)
        m.load_state_dict(ckpt['model'])
        m.to(self.device).eval()
        self._model = m

    @torch.no_grad()
    def predict_image(self, image_gray: np.ndarray) -> dict:
        self._lazy_load()
        return predict_with_model(self._model, image_gray, input_h=self.input_h, input_w=self.input_w, num_lanes=self.num_lanes, exist_threshold=self.exist_threshold, prob_threshold=self.prob_threshold, min_lane_rows=self.min_lane_rows)

    def _extract_per_row_x(self, seg_prob, class_idx, H, W):
        return _extract_per_row_x_helper(seg_prob, class_idx, H, W,
                                         self.prob_threshold, self.min_lane_rows)
