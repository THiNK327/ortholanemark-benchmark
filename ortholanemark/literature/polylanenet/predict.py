"""Inference adapter: vendored PolyLaneNet output → our (L, R, exists)
format.

Inference path mirrors upstream's `test.py`:
  1. Forward + decode → (outputs, extra)
     outputs shape (1, max_lanes, 7) where 7 =
       (sigmoid_conf, lower_y, upper_y, a3, a2, a1, a0)
     for x_norm = a3·y_norm³ + a2·y_norm² + a1·y_norm + a0.
  2. exists per lane: sigmoid(conf) > 0.5  (upstream conf_threshold=0.5
     in `PolyRegression.decode`).
  3. Evaluate each polynomial over the full original-image height.
     Predicted vertical extents do not truncate benchmark boundaries.
     Keep geometry independently of confidence for later threshold selection;
     off-image coordinates remain undefined under the common convention.

Lane 0 = LEFT, lane 1 = RIGHT (positional, by the dataset's convention).
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
from ortholanemark.literature.polylanenet.models import (
    PolyRegression,
)


_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
DEFAULT_CKPT = default_checkpoint_path('polylanenet')


@torch.no_grad()
def predict_batch_with_model(model, image_grays, img_h: int = 360,
                               img_w: int = 640, max_lanes: int = 2,
                               conf_threshold: float = 0.5) -> list:
    """Batched inference: single model.forward over the batch, then
    per-image polynomial-eval post-processing."""
    device = next(model.parameters()).device
    B = len(image_grays)
    tensors = []
    sizes = []
    for img in image_grays:
        H, W = img.shape[:2]
        sizes.append((H, W))
        img_r = cv2.resize(img, (img_w, img_h),
                           interpolation=cv2.INTER_AREA)
        img_3 = np.stack([img_r, img_r, img_r], axis=-1).astype(np.float32) / 255.0
        img_3 = (img_3 - _IMAGENET_MEAN) / _IMAGENET_STD
        tensors.append(torch.from_numpy(
            np.ascontiguousarray(img_3.transpose(2, 0, 1))).float())
    x = torch.stack(tensors, dim=0).to(device)
    output, _extra = model(x)
    output = output.reshape(B, max_lanes, 7)
    confs_all = torch.sigmoid(output[:, :, 0]).cpu().numpy()    # (B, max_lanes)
    polys_all = output[:, :, 3:7].cpu().numpy()                  # (B, max_lanes, 4)

    results = []
    for i in range(B):
        H, W = sizes[i]
        confs = confs_all[i]
        polys = polys_all[i]

        def _one(idx: int):
            a3, a2, a1, a0 = polys[idx]
            y_full = np.arange(H, dtype=np.float64) / max(float(H - 1), 1.0)
            x_norm = (a3 * y_full ** 3 + a2 * y_full ** 2
                      + a1 * y_full + a0)
            x_pix = x_norm * float(W - 1)
            off = (x_pix < 0.0) | (x_pix > float(W - 1))
            return np.where(off, np.nan, x_pix), bool(confs[idx] >= conf_threshold)

        x_L, eL = _one(0)
        x_R, eR = _one(1) if max_lanes > 1 \
            else (np.full(H, np.nan, dtype=np.float64), False)
        results.append({
            'pred_x_L': x_L, 'pred_x_R': x_R,
            'pred_exists_L': eL, 'pred_exists_R': eR,
            'pred_conf_L': float(confs[0]),
            'pred_conf_R': float(confs[1]) if max_lanes > 1 else 0.0,
        })
    return results


@torch.no_grad()
def predict_with_model(model, image_gray, img_h: int = 360, img_w: int = 640, max_lanes: int = 2, conf_threshold: float = 0.5) -> dict:
    """Use the same decoder for individual and batched predictions."""
    t0 = time.perf_counter()
    out = predict_batch_with_model(model, [image_gray], img_h=img_h, img_w=img_w, max_lanes=max_lanes, conf_threshold=conf_threshold)[0]
    out['inference_ms'] = (time.perf_counter() - t0) * 1000.0
    return out


@register_method('polylanenet')
class PolyLaneNetMethod(BaseLiteratureMethod):

    def __init__(self,
                 ckpt_path: Optional[str] = None,
                 img_h: int = 360,
                 img_w: int = 640,
                 max_lanes: int = 2,
                 backbone: str = 'efficientnet-b0',
                 conf_threshold: float = 0.5,
                 device: Optional[str] = None):
        self.ckpt_path = Path(ckpt_path or DEFAULT_CKPT)
        self.img_h = int(img_h)
        self.img_w = int(img_w)
        self.max_lanes = int(max_lanes)
        self.backbone = backbone
        self.conf_threshold = float(conf_threshold)
        self.device = device or (
            'cuda' if torch.cuda.is_available() else 'cpu')
        self._model: Optional[PolyRegression] = None

    def _lazy_load(self) -> None:
        if self._model is not None:
            return
        if not self.ckpt_path.exists():
            raise FileNotFoundError(
                f'polylanenet checkpoint not found: {self.ckpt_path}')
        ckpt = torch.load(self.ckpt_path, map_location=self.device,
                          weights_only=False)
        cfg = ckpt.get('config', {})
        self.img_h = int(cfg.get('img_h', self.img_h))
        self.img_w = int(cfg.get('img_w', self.img_w))
        self.max_lanes = int(cfg.get('max_lanes', self.max_lanes))
        self.backbone = cfg.get('backbone', self.backbone)
        num_outputs = int(cfg.get('num_outputs', self.max_lanes * 7))
        m = PolyRegression(num_outputs=num_outputs, backbone=self.backbone,
                           pretrained=False, curriculum_steps=None,
                           extra_outputs=0, share_top_y=True,
                           pred_category=False)
        m.load_state_dict(ckpt['model'])
        m.to(self.device).eval()
        self._model = m

    @torch.no_grad()
    def predict_image(self, image_gray: np.ndarray) -> dict:
        self._lazy_load()
        return predict_with_model(self._model, image_gray, img_h=self.img_h, img_w=self.img_w, max_lanes=self.max_lanes, conf_threshold=self.conf_threshold)
