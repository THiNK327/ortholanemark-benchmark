"""Inference adapter: vendored CLRNet proposals -> our (L, R, exists) format.

Registers as the literature method `clrnet_faithful`.

Inference mirrors upstream's test path (`CLRHead.get_lanes`):
  1. Forward -> (num_priors, 78) proposals; softmax confidence filter at
     conf_threshold=0.4, line-NMS at nms_thres=50 (upstream CULane test
     defaults), top_k = max_lanes = 2.
  2. Per surviving proposal, take the x offsets on the valid strip range
     [start, start+length) that lie inside the frame (upstream's
     predictions_to_pred masking), map (prior_y, x) to original-image
     coords, fit a degree-2 polynomial, and evaluate at every row —
     the same full-height adapter used for LaneATT.
  3. Keep at most 2 lanes (score-ordered); assign LEFT/RIGHT by mean x.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch

from ortholanemark.literature import (
    BaseLiteratureMethod, register_method,
)
from ortholanemark.literature.clrnet_faithful.model.detector import (
    CLRNet, CLRNetConfig,
)


DEFAULT_CKPT = ('ortholanemark/runs_reselect_train/'
                'clrnet_faithful/best_gra.pth')


def _proposal_to_full_height_curve(prop: np.ndarray, prior_ys: np.ndarray,
                                   img_w: int, n_strips: int,
                                   H: int, W: int):
    """One post-NMS CLRNet proposal -> per-row x curve over the original
    image height H, or None if too few valid points.

    prop: [cls0, cls1, start_y, start_x, theta, length(strips), xs_norm...]
    """
    xs_norm = prop[6:]
    start = min(max(0, int(round(prop[2] * n_strips))), n_strips)
    length = int(round(prop[5]))
    end = min(start + length - 1, len(xs_norm) - 1)
    if end < start:
        return None
    idx = np.arange(start, end + 1)
    idx = idx[(xs_norm[idx] >= 0.0) & (xs_norm[idx] <= 1.0)]
    if len(idx) < 2:
        return None
    ys_orig = prior_ys[idx] * (H - 1)
    xs_orig = xs_norm[idx] * float(W - 1)
    deg = 2 if len(idx) >= 5 else 1
    coefs = np.polyfit(ys_orig, xs_orig, deg)
    y_full = np.arange(H, dtype=np.float64)
    x_full = np.polyval(coefs, y_full)
    off = (x_full < 0.0) | (x_full > float(W - 1))
    x_full = np.where(off, np.nan, x_full)
    if not np.isfinite(x_full).any():
        return None
    return x_full


def _preprocess(image_gray: np.ndarray, img_w: int,
                img_h: int) -> torch.Tensor:
    img_r = cv2.resize(image_gray, (img_w, img_h),
                       interpolation=cv2.INTER_AREA)
    img_3 = np.stack([img_r, img_r, img_r],
                     axis=-1).astype(np.float32) / 255.0
    return torch.from_numpy(
        np.ascontiguousarray(img_3.transpose(2, 0, 1))).float()


@torch.no_grad()
def predict_batch_with_model(model, image_grays,
                             conf_threshold: float = 0.4,
                             nms_thres: float = 50.0,
                             nms_topk: int = 2) -> list:
    """Batched inference; same contract as the other faithful adapters."""
    device = next(model.parameters()).device
    cfg = model.cfg
    sizes = [img.shape[:2] for img in image_grays]
    x = torch.stack([_preprocess(img, cfg.img_w, cfg.img_h)
                     for img in image_grays], dim=0).to(device)
    output = model(x)                       # (B, num_priors, 78)
    decoded = model.get_lanes(output, conf_threshold=conf_threshold,
                              nms_thres=nms_thres, nms_topk=nms_topk)

    prior_ys = model.heads.prior_ys.cpu().numpy()
    n_strips = model.heads.n_strips

    results = []
    for (H, W), props in zip(sizes, decoded):
        props = props.cpu().numpy()
        # score-ordered curves (props emerge from NMS score-sorted). Carry
        # each proposal's softmax(cls) confidence to become the per-side
        # existence score (post-NMS → PR sweeps the operating range).
        curves, cconf = [], []
        for p in props[:nms_topk]:
            c = _proposal_to_full_height_curve(p, prior_ys, cfg.img_w,
                                               n_strips, H, W)
            if c is not None:
                z0, z1 = float(p[0]), float(p[1])
                m = max(z0, z1)
                sc = np.exp(z1 - m) / (np.exp(z0 - m) + np.exp(z1 - m))
                curves.append(c)
                cconf.append(float(sc))

        nanH = np.full(H, np.nan, dtype=np.float64)
        x_L, x_R = nanH, nanH.copy()
        eL = eR = False
        confL = confR = 0.0
        if len(curves) >= 2:
            if np.nanmean(curves[0]) <= np.nanmean(curves[1]):
                x_L, x_R, confL, confR = (curves[0], curves[1],
                                          cconf[0], cconf[1])
            else:
                x_L, x_R, confL, confR = (curves[1], curves[0],
                                          cconf[1], cconf[0])
            eL = eR = True
        elif len(curves) == 1:
            if np.nanmean(curves[0]) < W / 2.0:
                x_L, eL, confL = curves[0], True, cconf[0]
            else:
                x_R, eR, confR = curves[0], True, cconf[0]

        results.append({
            'pred_x_L': x_L, 'pred_x_R': x_R,
            'pred_exists_L': eL, 'pred_exists_R': eR,
            'pred_conf_L': confL, 'pred_conf_R': confR,
        })
    return results


@torch.no_grad()
def predict_with_model(model, image_gray, **kw) -> dict:
    t0 = time.perf_counter()
    out = predict_batch_with_model(model, [image_gray], **kw)[0]
    out['inference_ms'] = (time.perf_counter() - t0) * 1000.0
    return out


@register_method('clrnet_faithful')
class CLRNetFaithfulMethod(BaseLiteratureMethod):
    """Paper-faithful CLRNet inference with project output adaptation."""

    def __init__(self,
                 ckpt_path: Optional[str] = None,
                 conf_threshold: float = 0.4,
                 nms_thres: float = 50.0,
                 nms_topk: int = 2,
                 device: Optional[str] = None):
        self.ckpt_path = Path(ckpt_path or DEFAULT_CKPT)
        self.conf_threshold = float(conf_threshold)
        self.nms_thres = float(nms_thres)
        self.nms_topk = int(nms_topk)
        self.device = device or (
            'cuda' if torch.cuda.is_available() else 'cpu')
        self._model: Optional[CLRNet] = None

    def _lazy_load(self) -> None:
        if self._model is not None:
            return
        if not self.ckpt_path.exists():
            raise FileNotFoundError(
                f'clrnet_faithful checkpoint not found: {self.ckpt_path}')
        ckpt = torch.load(self.ckpt_path, map_location=self.device,
                          weights_only=False)
        cfg = CLRNetConfig(**ckpt.get('config_model', {}),
                           pretrained=False)
        m = CLRNet(cfg)
        m.load_state_dict(ckpt['model'])
        m.to(self.device).eval()
        self._model = m

    @torch.no_grad()
    def predict_image(self, image_gray: np.ndarray) -> dict:
        t0 = time.perf_counter()
        self._lazy_load()
        out = predict_batch_with_model(
            self._model, [image_gray],
            conf_threshold=self.conf_threshold,
            nms_thres=self.nms_thres, nms_topk=self.nms_topk)[0]
        out['inference_ms'] = (time.perf_counter() - t0) * 1000.0
        return out
