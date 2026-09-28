"""Inference adapter: vendored LaneATT proposals → our (L, R, exists) format.

Inference path mirrors upstream's test code (`lib/runner.py:eval` →
`model.decode(..., as_lanes=False)`):
  1. Forward with conf_threshold=0.2, nms_thres=45 (upstream TuSimple
     test defaults).
     This returns a per-image list of (proposals, anchors, attn, inds).
     Proposals format per row: [cls0, cls1, start_y, start_x, length, x[S]].
  2. Per image, sort surviving proposals by softmax(cls1) descending and
     keep at most 2 (matching our max_lanes).
  3. Of those, assign LEFT/RIGHT by midline x (the lane whose mean
     x at the visible y rows is smaller → LEFT, larger → RIGHT).
  4. For each assigned lane, convert the (x_offset, anchor_y) pairs to
     a per-row x array in the original image coords by polynomial
     fitting across all H rows.
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
from ortholanemark.literature.laneatt.model.laneatt import (
    LaneATT,
)


DEFAULT_CKPT = default_checkpoint_path('laneatt')


def _resolve_anchor_frequency_path(value):
    """Resolve the package-relative anchor resource, preserving custom paths."""
    if value is None:
        return None
    normalized = str(value).replace("\\", "/").removeprefix("./")
    if normalized == "ortholanemark/literature/laneatt/anchor_frequencies.pt":
        return str(Path(__file__).resolve().with_name("anchor_frequencies.pt"))
    return value


def _proposal_to_full_height_curve(prop: np.ndarray, S: int,
                                    img_h: int, img_w: int,
                                    H: int, W: int) -> 'np.ndarray | None':
    """Convert one LaneATT proposal (5+S vector) to a per-row x curve over
    the original image height H.

    Upstream's predict-only-within-(start, length) truncates the curve to
    a small y range, leaving the rest as NaN. OrthoLaneMark annotations span the
    full image, so we instead fit a degree-2 polynomial through the
    visible-and-in-frame strips and evaluate it at every row. Off-image
    rows after evaluation become NaN.
    """
    start_y_norm, _start_x_pix, length_f = prop[2], prop[3], prop[4]
    length = int(round(length_f))
    start_strip = min(max(0, int(round(start_y_norm * (S - 1)))), S)
    end_strip = min(start_strip + length, S)
    xs_resized = prop[5:5 + S]
    valid_idx = np.arange(start_strip, end_strip)
    valid_idx = valid_idx[(xs_resized[valid_idx] >= 0)
                          & (xs_resized[valid_idx] < img_w)]
    if len(valid_idx) < 2:
        return None
    anchor_ys_resized = np.linspace(1.0, 0.0, S) * (img_h - 1)
    ys_orig = anchor_ys_resized * (H - 1) / max(float(img_h - 1), 1.0)
    ys_in = ys_orig[valid_idx]
    xs_in_orig = xs_resized[valid_idx] * float(W - 1) / max(float(img_w - 1), 1.0)
    deg = 2 if len(ys_in) >= 5 else 1
    coefs = np.polyfit(ys_in, xs_in_orig, deg)
    y_full = np.arange(H, dtype=np.float64)
    x_full = np.polyval(coefs, y_full)
    off = (x_full < 0.0) | (x_full > float(W - 1))
    curve = np.where(off, np.nan, x_full)
    return curve if np.isfinite(curve).any() else None


@torch.no_grad()
def predict_batch_with_model(model, image_grays, img_h: int = 360,
                               img_w: int = 640, S: int = 72,
                               max_lanes: int = 2,
                               conf_threshold: float = 0.2,
                               nms_thres: float = 45.0,
                               nms_topk: int = 2) -> list:
    """Batched inference for the val_strict_iou hook. Preprocesses every
    image in the list, runs ONE batched model.forward, then post-processes
    per-image (selection / interp / L/R assignment).

    Returns a list of result dicts, one per image, in the same order.
    """
    import time as _time
    device = next(model.parameters()).device
    B = len(image_grays)

    # Preprocess on CPU then stack to GPU.
    tensors = []
    sizes = []
    for img in image_grays:
        H, W = img.shape[:2]
        sizes.append((H, W))
        img_r = cv2.resize(img, (img_w, img_h),
                           interpolation=cv2.INTER_AREA)
        img_3 = np.stack([img_r, img_r, img_r], axis=-1).astype(np.float32) / 255.0
        tensors.append(torch.from_numpy(
            np.ascontiguousarray(img_3.transpose(2, 0, 1))).float())
    x = torch.stack(tensors, dim=0).to(device)

    # Single batched forward. The model's nms handles per-image internally.
    proposals_list = model(x, conf_threshold=conf_threshold,
                            nms_thres=nms_thres, nms_topk=nms_topk)
    # proposals_list is a list of length B: (proposals, anchors, attn, inds)

    results = []
    for i in range(B):
        H, W = sizes[i]
        proposals, _anchors, _attn, _inds = proposals_list[i]
        if proposals.numel() == 0 or proposals.shape[0] == 0:
            nan_arr = np.full(H, np.nan, dtype=np.float64)
            results.append({
                'pred_x_L': nan_arr, 'pred_x_R': nan_arr,
                'pred_exists_L': False, 'pred_exists_R': False,
                'pred_conf_L': 0.0, 'pred_conf_R': 0.0,
            })
            continue

        probs = torch.softmax(proposals[:, :2], dim=1)[:, 1]
        order = torch.argsort(probs, descending=True)
        kept = proposals[order[:max_lanes]].cpu().numpy()
        kept_probs = probs[order[:max_lanes]].cpu().numpy()

        curves, midlines, cconf = [], [], []
        for j, prop in enumerate(kept):
            x_full = _proposal_to_full_height_curve(prop, S, img_h, img_w, H, W)
            if x_full is None:
                continue
            curves.append(x_full)
            midlines.append(float(np.nanmean(x_full)))
            cconf.append(float(kept_probs[j]))

        confL = confR = 0.0
        if len(curves) == 0:
            x_L = np.full(H, np.nan, dtype=np.float64)
            x_R = np.full(H, np.nan, dtype=np.float64)
            eL = eR = False
        elif len(curves) == 1:
            if midlines[0] < W / 2:
                x_L, eL, confL = curves[0], True, cconf[0]
                x_R, eR = np.full(H, np.nan, dtype=np.float64), False
            else:
                x_L, eL = np.full(H, np.nan, dtype=np.float64), False
                x_R, eR, confR = curves[0], True, cconf[0]
        else:
            mid_order = np.argsort(midlines)
            x_L, eL, confL = curves[mid_order[0]], True, cconf[mid_order[0]]
            x_R, eR, confR = curves[mid_order[-1]], True, cconf[mid_order[-1]]

        results.append({
            'pred_x_L': x_L, 'pred_x_R': x_R,
            'pred_exists_L': eL, 'pred_exists_R': eR,
            'pred_conf_L': confL, 'pred_conf_R': confR,
        })
    return results


@torch.no_grad()
def predict_with_model(model, image_gray, img_h: int = 360, img_w: int = 640, S: int = 72, max_lanes: int = 2, conf_threshold: float = 0.2, nms_thres: float = 45.0, nms_topk: int = 2) -> dict:
    """Use the same decoder for individual and batched predictions."""
    t0 = time.perf_counter()
    out = predict_batch_with_model(model, [image_gray], img_h=img_h, img_w=img_w, S=S, max_lanes=max_lanes, conf_threshold=conf_threshold, nms_thres=nms_thres, nms_topk=nms_topk)[0]
    out['inference_ms'] = (time.perf_counter() - t0) * 1000.0
    return out


@register_method('laneatt')
class LaneATTMethod(BaseLiteratureMethod):

    def __init__(self,
                 ckpt_path: Optional[str] = None,
                 img_h: int = 360,
                 img_w: int = 640,
                 S: int = 72,
                 max_lanes: int = 2,
                 backbone: str = 'resnet34',
                 conf_threshold: float = 0.2,
                 nms_thres: float = 45.0,
                 nms_topk: Optional[int] = None,
                 device: Optional[str] = None):
        self.ckpt_path = Path(ckpt_path or DEFAULT_CKPT)
        self.img_h = int(img_h)
        self.img_w = int(img_w)
        self.S = int(S)
        self.max_lanes = int(max_lanes)
        self.backbone = backbone
        self.conf_threshold = float(conf_threshold)
        self.nms_thres = float(nms_thres)
        self.nms_topk = int(nms_topk if nms_topk is not None else max_lanes)
        self.device = device or (
            'cuda' if torch.cuda.is_available() else 'cpu')
        self._model: Optional[LaneATT] = None

    def _lazy_load(self) -> None:
        if self._model is not None:
            return
        if not self.ckpt_path.exists():
            raise FileNotFoundError(
                f'laneatt checkpoint not found: {self.ckpt_path}')
        ckpt = torch.load(self.ckpt_path, map_location=self.device,
                          weights_only=False)
        cfg = ckpt.get('config', {})
        for k in ('img_h', 'img_w', 'S', 'max_lanes', 'backbone'):
            if k in cfg:
                setattr(self, k, cfg[k])
        # Reconstruct anchor filtering exactly as during training so the
        # weights shape match.
        freq_path = _resolve_anchor_frequency_path(cfg.get('anchors_freq_path'))
        topk = cfg.get('topk_anchors')
        m = LaneATT(backbone=self.backbone, pretrained_backbone=False,
                    S=self.S, img_w=self.img_w, img_h=self.img_h,
                    anchors_freq_path=freq_path, topk_anchors=topk,
                    anchor_feat_channels=64)
        m.load_state_dict(ckpt['model'])
        m.to(self.device).eval()
        self._model = m

    @torch.no_grad()
    def predict_image(self, image_gray: np.ndarray) -> dict:
        self._lazy_load()
        return predict_with_model(self._model, image_gray, img_h=self.img_h, img_w=self.img_w, S=self.S, max_lanes=self.max_lanes, conf_threshold=self.conf_threshold, nms_thres=self.nms_thres, nms_topk=self.nms_topk)
