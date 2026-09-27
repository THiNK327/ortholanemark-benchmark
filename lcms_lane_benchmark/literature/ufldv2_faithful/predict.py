"""Inference adapter for faithful UFLDv2 output -> project per-row format.

Native UFLDv2 output is sparse row/column-anchor ordinal classification.
For this project's metrics we convert the row-anchor head into one
full-height x curve per lane.

The sparse point extraction follows upstream `evaluation/eval_wrapper.py`
for TuSimple `mode='4row'`:
  1. A row anchor is valid when `exist_row.argmax(0)` selects "exist".
  2. For each valid row anchor, take the argmax bin in `loc_row`, then
     compute a local soft-argmax over +/-14 bins and add 0.5.
  3. Normalize the coordinate by `(num_cell_row - 1)`.
  4. Fit a degree-2 polynomial through the sparse row-anchor points and
     evaluate it at every original-image row. This final fit is the
     project adapter from UFLDv2's sparse output to our per-row metric.

The col-anchor head is still trained by the faithful loss. It is not
consumed here because upstream TuSimple evaluation also emits lanes from
the row head only in `mode='4row'`.
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
from lcms_lane_benchmark.literature.ufldv2_faithful.model.model_culane import (
    parsingNet,
)


_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
DEFAULT_CKPT = 'lcms_lane_benchmark/runs_reselect_train/ufldv2_faithful/best_gra.pth'


def _softmax(x: np.ndarray, axis: int) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    ex = np.exp(x)
    return ex / ex.sum(axis=axis, keepdims=True)


def _local_softargmax_row(logits: np.ndarray, num_cell_row: int,
                            num_cls_row: int, local_width_row: int) -> np.ndarray:
    """Upstream's local soft-argmax for `loc_row`. logits shape
    (num_cell_row, num_cls_row).
    """
    max_indices = logits.argmax(axis=0)
    out = np.zeros(num_cls_row, dtype=np.float64)
    for row_cls_idx, max_idx in enumerate(max_indices):
        lo = max(0, int(max_idx) - local_width_row)
        hi = min(num_cell_row - 1, int(max_idx) + local_width_row)
        all_ind = np.arange(lo, hi + 1, dtype=np.float64)
        local_logits = logits[lo:hi + 1, row_cls_idx]
        probs = _softmax(local_logits, axis=0)
        coord = float((probs * all_ind).sum() + 0.5)
        out[row_cls_idx] = coord / max(float(num_cell_row - 1), 1.0)
    return out


@torch.no_grad()
def predict_batch_with_model(model, image_grays, input_h: int = 800,
                               input_w: int = 320, num_lanes: int = 2,
                               num_cls_row: int = 56, num_cell_row: int = 100,
                               min_lane_rows: int = 3,
                               local_width_row: int = 14) -> list:
    """Batched inference: single model.forward, per-image local
    soft-argmax + polyfit post-processing."""
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

    pred = model(x)
    loc_row_all = pred['loc_row'].cpu().numpy()       # (B, n_cell, n_cls, n_lanes)
    exist_row_all = pred['exist_row'].cpu().numpy()   # (B, 2, n_cls, n_lanes)
    row_anchors_norm = np.linspace(0.0, 1.0, num_cls_row)

    results = []
    for i in range(B):
        H, W = sizes[i]
        loc_row = loc_row_all[i]
        exist_row = exist_row_all[i]
        confidences = _softmax(exist_row, axis=0)[1].mean(axis=0)

        per_lane = []
        for lane in range(num_lanes):
            exists_per_anchor = (exist_row.argmax(axis=0)[:, lane] == 1)
            if int(exists_per_anchor.sum()) <= min_lane_rows:
                per_lane.append((np.full(H, np.nan, dtype=np.float64), False))
                continue
            x_norm_anchor = _local_softargmax_row(
                loc_row[:, :, lane], num_cell_row, num_cls_row, local_width_row)
            ys = row_anchors_norm[exists_per_anchor]
            xs = x_norm_anchor[exists_per_anchor]
            deg = 2 if len(xs) >= 5 else 1
            coefs = np.polyfit(ys, xs, deg)
            y_full = np.arange(H, dtype=np.float64) / max(float(H - 1), 1.0)
            xn = np.polyval(coefs, y_full)
            x_pix = xn * float(W - 1)
            off = (x_pix < 0.0) | (x_pix > float(W - 1))
            curve = np.where(off, np.nan, x_pix)
            per_lane.append((curve, bool(np.isfinite(curve).any())))

        x_L, eL = per_lane[0]
        x_R, eR = (per_lane[1] if len(per_lane) > 1
                   else (np.full(H, np.nan, dtype=np.float64), False))
        results.append({
            'pred_x_L': x_L, 'pred_x_R': x_R,
            'pred_exists_L': eL, 'pred_exists_R': eR,
            'pred_conf_L': float(confidences[0]),
            'pred_conf_R': float(confidences[1]) if num_lanes > 1 else 0.0,
        })
    return results


@torch.no_grad()
def predict_with_model(model, image_gray, input_h: int = 800, input_w: int = 320, num_lanes: int = 2, num_cls_row: int = 56, num_cell_row: int = 100, min_lane_rows: int = 3, local_width_row: int = 14) -> dict:
    """Use the same decoder for individual and batched predictions."""
    t0 = time.perf_counter()
    out = predict_batch_with_model(model, [image_gray], input_h=input_h, input_w=input_w, num_lanes=num_lanes, num_cls_row=num_cls_row, num_cell_row=num_cell_row, min_lane_rows=min_lane_rows, local_width_row=local_width_row)[0]
    out['inference_ms'] = (time.perf_counter() - t0) * 1000.0
    return out


@register_method('ufldv2_faithful')
class UFLDv2FaithfulMethod(BaseLiteratureMethod):

    def __init__(self,
                 ckpt_path: Optional[str] = None,
                 input_h: int = 800,
                 input_w: int = 320,
                 num_lanes: int = 2,
                 num_cls_row: int = 56,
                 num_cls_col: int = 41,
                 num_cell_row: int = 100,
                 num_cell_col: int = 100,
                 backbone: str = '18',
                 min_lane_rows: int = 3,
                 local_width_row: int = 14,
                 device: Optional[str] = None):
        self.ckpt_path = Path(ckpt_path or DEFAULT_CKPT)
        self.input_h = int(input_h)
        self.input_w = int(input_w)
        self.num_lanes = int(num_lanes)
        self.num_cls_row = int(num_cls_row)
        self.num_cls_col = int(num_cls_col)
        self.num_cell_row = int(num_cell_row)
        self.num_cell_col = int(num_cell_col)
        self.backbone = backbone
        self.min_lane_rows = int(min_lane_rows)
        self.local_width_row = int(local_width_row)
        self.device = device or (
            'cuda' if torch.cuda.is_available() else 'cpu')
        self._model: Optional[parsingNet] = None

    def _lazy_load(self) -> None:
        if self._model is not None:
            return
        if not self.ckpt_path.exists():
            raise FileNotFoundError(
                f'ufldv2_faithful checkpoint not found: {self.ckpt_path}')
        ckpt = torch.load(self.ckpt_path, map_location=self.device,
                          weights_only=False)
        cfg = ckpt.get('config', {})
        for k in ('input_h', 'input_w', 'num_lanes',
                  'num_cls_row', 'num_cls_col',
                  'num_cell_row', 'num_cell_col', 'backbone'):
            if k in cfg:
                setattr(self, k, cfg[k])
        # Must match the training-time `use_aux` so SegHead weights align.
        use_aux_ckpt = bool(cfg.get('use_aux', False))
        m = parsingNet(
            pretrained=False, backbone=str(self.backbone),
            num_grid_row=self.num_cell_row, num_cls_row=self.num_cls_row,
            num_grid_col=self.num_cell_col, num_cls_col=self.num_cls_col,
            num_lane_on_row=self.num_lanes, num_lane_on_col=self.num_lanes,
            use_aux=use_aux_ckpt,
            input_height=self.input_h, input_width=self.input_w,
            fc_norm=False)
        m.load_state_dict(ckpt['model'])
        m.to(self.device).eval()
        self._model = m

    @torch.no_grad()
    def predict_image(self, image_gray: np.ndarray) -> dict:
        self._lazy_load()
        return predict_with_model(self._model, image_gray, input_h=self.input_h, input_w=self.input_w, num_lanes=self.num_lanes, num_cls_row=self.num_cls_row, num_cell_row=self.num_cell_row, min_lane_rows=self.min_lane_rows, local_width_row=self.local_width_row)

    def _local_softargmax_row(self, logits):
        return _local_softargmax_row(logits, self.num_cell_row, self.num_cls_row,
                                     self.local_width_row)


def _softmax(x: np.ndarray, axis: int) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    ex = np.exp(x)
    return ex / ex.sum(axis=axis, keepdims=True)


# ---- Sibling: UFLDv2-TuSimple recipe (use_aux=False) ----
# `ufldv2_faithful`          → CULane recipe (use_aux=True)
# `ufldv2_faithful_tusimple` → TuSimple recipe (use_aux=False)
# Both faithful to *an* upstream config; identical adapter code.

UFLDV2_TUSIMPLE_DEFAULT_CKPT = (
    'lcms_lane_benchmark/runs_reselect_train/ufldv2_faithful_tusimple/best_gra.pth')


@register_method('ufldv2_faithful_tusimple')
class UFLDv2FaithfulTusimpleMethod(UFLDv2FaithfulMethod):
    """UFLDv2 faithful, mirrors upstream **TuSimple** config
    (`use_aux=False`). Identical adapter and inference path; only the
    default ckpt directory differs.
    """

    def __init__(self,
                 ckpt_path: Optional[str] = None,
                 **kwargs):
        super().__init__(
            ckpt_path=ckpt_path or UFLDV2_TUSIMPLE_DEFAULT_CKPT,
            **kwargs,
        )
