"""Shared utilities for Tier-1 classical literature baselines.

The three classical methods (LSD, Canny+Hough, Steger ridge) each detect
line/curve evidence in different ways, but share the same post-processing:

  raw segments / ridge points
    → filter to near-vertical orientation (lane boundaries are roughly
      column-aligned in top-down LCMS imagery)
    → cluster into left/right by centroid x position
    → fit degree-1 polynomial per group
    → emit per-row x array (NaN where off-frame)

This module exposes that pipeline as `segments_to_curves(...)`.
"""

from __future__ import annotations

import math
from typing import Optional

import cv2
import numpy as np


def _segment_angle_deg(x1: float, y1: float, x2: float, y2: float) -> float:
    """Angle of the segment from vertical (y-axis), in degrees.
    A perfectly vertical segment returns 0.
    A perfectly horizontal segment returns 90.
    """
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    if dy < 1e-9:
        return 90.0
    return math.degrees(math.atan(dx / dy))


def filter_near_vertical(segments: np.ndarray,
                          max_angle_from_vertical_deg: float = 25.0,
                          min_length_px: float = 30.0
                          ) -> np.ndarray:
    """Keep only segments roughly aligned with the image's vertical axis
    AND longer than `min_length_px`.

    Args:
      segments : (N, 4) array of (x1, y1, x2, y2).
      max_angle_from_vertical_deg : reject segments tilted more than this.
      min_length_px : reject segments shorter than this.

    Returns:
      (M, 4) subset that passed both filters.
    """
    if segments is None or len(segments) == 0:
        return np.zeros((0, 4), dtype=np.float64)
    segs = np.asarray(segments, dtype=np.float64).reshape(-1, 4)
    kept = []
    for x1, y1, x2, y2 in segs:
        length = math.hypot(x2 - x1, y2 - y1)
        if length < min_length_px:
            continue
        if _segment_angle_deg(x1, y1, x2, y2) > max_angle_from_vertical_deg:
            continue
        kept.append([x1, y1, x2, y2])
    return (np.array(kept, dtype=np.float64) if kept
            else np.zeros((0, 4), dtype=np.float64))


def group_left_right(segments: np.ndarray, W: int,
                     left_zone: float = 0.45,
                     right_zone: float = 0.55) -> tuple:
    """Group segments by centroid x position into left / right buckets.

    A segment with mean x in [0, left_zone * W] joins the left group.
    A segment with mean x in [right_zone * W, W] joins the right group.
    Segments in the middle zone (between left_zone*W and right_zone*W)
    are split: assigned to whichever side their centroid is closer to.

    Returns:
      (left_segments, right_segments) — each (N, 4) arrays of (x1,y1,x2,y2).
    """
    if len(segments) == 0:
        return (np.zeros((0, 4), dtype=np.float64),
                np.zeros((0, 4), dtype=np.float64))
    cx = 0.5 * (segments[:, 0] + segments[:, 2])
    left_mask = cx <= left_zone * W
    right_mask = cx >= right_zone * W
    middle_mask = ~(left_mask | right_mask)
    # Middle-zone segments go to whichever boundary their centroid is closer.
    middle_to_left = middle_mask & (cx < 0.5 * W)
    middle_to_right = middle_mask & (cx >= 0.5 * W)
    return (segments[left_mask | middle_to_left],
            segments[right_mask | middle_to_right])


def fit_degree1_curve(segments: np.ndarray, H: int, W: int,
                       min_total_length_px: float = 80.0
                       ) -> tuple:
    """Fit a degree-1 polynomial through the endpoints of all segments
    in `segments`. Used per left/right group.

    Args:
      segments : (N, 4) (x1, y1, x2, y2).
      H, W     : image dimensions.
      min_total_length_px : minimum cumulative segment length required
                            to declare an "existing" side. If the total
                            length of segments is below this, returns
                            (NaN-array, False).

    Returns:
      (pred_x : (H,) float64 NaN where off-frame, exists : bool).
    """
    if len(segments) == 0:
        return (np.full(H, np.nan, dtype=np.float64), False)
    total_length = 0.0
    for x1, y1, x2, y2 in segments:
        total_length += math.hypot(x2 - x1, y2 - y1)
    if total_length < min_total_length_px:
        return (np.full(H, np.nan, dtype=np.float64), False)

    # Stack endpoints and fit x = a1*y + a0.
    pts = []
    for x1, y1, x2, y2 in segments:
        pts.append((y1, x1))
        pts.append((y2, x2))
    pts = np.asarray(pts, dtype=np.float64)
    ys = pts[:, 0]
    xs = pts[:, 1]
    if ys.max() - ys.min() < 5:    # endpoints too clustered to fit slope
        coefs = np.array([0.0, float(np.mean(xs))], dtype=np.float64)
    else:
        coefs = np.polyfit(ys, xs, 1)
    # Eval at every row.
    rows = np.arange(H, dtype=np.float64)
    x_per_row = np.polyval(coefs, rows)
    # Mark off-frame rows as NaN.
    in_frame = (x_per_row >= 0) & (x_per_row <= W - 1)
    x_per_row = np.where(in_frame, x_per_row, np.nan)
    return (x_per_row, True)


def segments_to_curves(segments: np.ndarray, H: int, W: int,
                       *,
                       max_angle_from_vertical_deg: float = 25.0,
                       min_segment_length_px: float = 30.0,
                       left_zone: float = 0.45,
                       right_zone: float = 0.55,
                       min_total_length_px: float = 80.0
                       ) -> tuple:
    """Full classical pipeline: filter → group → fit. Returns
    (pred_x_L, pred_x_R, exists_L, exists_R)."""
    filtered = filter_near_vertical(
        segments,
        max_angle_from_vertical_deg=max_angle_from_vertical_deg,
        min_length_px=min_segment_length_px,
    )
    left_segs, right_segs = group_left_right(
        filtered, W, left_zone=left_zone, right_zone=right_zone)
    pred_x_L, exists_L = fit_degree1_curve(
        left_segs, H, W, min_total_length_px=min_total_length_px)
    pred_x_R, exists_R = fit_degree1_curve(
        right_segs, H, W, min_total_length_px=min_total_length_px)
    return pred_x_L, pred_x_R, exists_L, exists_R


# ----------------------------------------------------------------------
# Robust inner-boundary conversion (shared by all three classical methods)
#
# The three detectors emit raw evidence (edge/line segments, or ridge
# pixels) that is (a) polluted by non-marking responses on cracks/texture
# and (b) centred on the marking rather than on the lane-facing inner edge
# the annotation defines. This converts raw evidence to an inner-boundary
# prediction in three named steps:
#   1. INTENSITY GATING   — keep only evidence coincident with bright
#      (marking) pixels, using a per-image Otsu threshold.
#   2. ROBUST LINE FIT     — fit the dominant near-vertical boundary per
#      side with RANSAC, rejecting the remaining spatial outliers.
#   3. INNER-EDGE SNAP     — shift the fitted boundary onto the marking's
#      lane-facing intensity edge (the annotated boundary).
# ----------------------------------------------------------------------
def otsu_bright_threshold(image: np.ndarray) -> float:
    """Per-image intensity threshold separating bright markings from the
    darker pavement (Otsu). Used for intensity gating and the inner snap."""
    return float(cv2.threshold(image, 0, 255,
                               cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0])


def segments_to_points(segments: np.ndarray, step_px: float = 15.0
                       ) -> np.ndarray:
    """Sample (y, x) points every `step_px` along each segment. Gives the
    RANSAC fit a denser, evenly-weighted cloud than raw endpoints."""
    if segments is None or len(segments) == 0:
        return np.zeros((0, 2), dtype=np.float64)
    out = []
    for x1, y1, x2, y2 in np.asarray(segments, dtype=np.float64).reshape(-1, 4):
        n = max(2, int(math.hypot(x2 - x1, y2 - y1) / step_px))
        t = np.linspace(0.0, 1.0, n)
        out.append(np.stack([y1 + t * (y2 - y1), x1 + t * (x2 - x1)], axis=1))
    return np.concatenate(out) if out else np.zeros((0, 2), dtype=np.float64)


def ransac_near_vertical(ys: np.ndarray, xs: np.ndarray,
                         tol_px: float = 12.0, iters: int = 250,
                         min_inliers: int = 30, max_slope: float = 0.5,
                         min_y_span: float = 200.0,
                         rng: Optional[np.random.RandomState] = None
                         ) -> Optional[np.ndarray]:
    """RANSAC fit of a near-vertical line x = a*y + b to a (y, x) cloud.

    Rejects models tilted more than `max_slope` (dx/dy) from vertical, then
    refits degree-1 on the largest inlier set (points within `tol_px`).
    Returns the polynomial coefficients, or None if no line is supported
    (too few inliers, or inliers span < `min_y_span` rows).
    """
    n = len(ys)
    if n < 2:
        return None
    if rng is None:
        rng = np.random.RandomState(0)
    best = None
    for _ in range(iters):
        i, j = int(rng.randint(0, n)), int(rng.randint(0, n))
        if ys[i] == ys[j]:
            continue
        a = (xs[j] - xs[i]) / (ys[j] - ys[i])
        if abs(a) > max_slope:
            continue
        b = xs[i] - a * ys[i]
        inl = np.abs(xs - (a * ys + b)) < tol_px
        if best is None or inl.sum() > best.sum():
            best = inl
    if best is None or best.sum() < min_inliers:
        return None
    if ys[best].max() - ys[best].min() < min_y_span:
        return None
    return np.polyfit(ys[best], xs[best], 1)


def inner_edge_snap(pred_x: np.ndarray, side: str, image: np.ndarray,
                    bright_threshold: float,
                    max_marking_width_px: float = 90.0,
                    min_points: int = 40) -> np.ndarray:
    """Shift a per-row boundary onto the marking's lane-facing inner edge.

    For each row where the seed sits on a bright marking pixel, grow the
    contiguous bright run through the seed and take its lane-facing end
    (right end for the left marking, left end for the right marking), then
    refit degree-1. Runs wider than `max_marking_width_px` are ignored (not
    a marking). Falls back to the input curve if too few edge points."""
    H, W = image.shape[:2]
    ys, xs = [], []
    for y in range(H):
        xc = pred_x[y]
        if not np.isfinite(xc):
            continue
        xci = int(round(xc))
        if xci < 0 or xci >= W or image[y, xci] < bright_threshold:
            continue
        l = xci
        while l - 1 >= 0 and image[y, l - 1] >= bright_threshold:
            l -= 1
        r = xci
        while r + 1 < W and image[y, r + 1] >= bright_threshold:
            r += 1
        if r - l > max_marking_width_px:
            continue
        ys.append(y)
        xs.append(r if side == 'L' else l)
    if len(ys) < min_points:
        return pred_x
    coefs = np.polyfit(np.asarray(ys, float), np.asarray(xs, float), 1)
    xr = np.polyval(coefs, np.arange(H, dtype=np.float64))
    return np.where((xr >= 0) & (xr <= W - 1), xr, np.nan)


def robust_inner_curves(points: np.ndarray, image: np.ndarray,
                        *,
                        left_zone: float = 0.45, right_zone: float = 0.55,
                        min_group_points: int = 30,
                        max_group_points: int = 3000) -> tuple:
    """Intensity gating → per-side RANSAC line → inner-edge snap.

    `points` is an (N, 2) array of (y, x) raw-detection coordinates
    (segment samples or ridge pixels). Returns
    (pred_x_L, pred_x_R, exists_L, exists_R), each per-row x (NaN off-frame).
    A side "exists" only when a dominant bright near-vertical line is found.
    """
    H, W = image.shape[:2]
    thr = otsu_bright_threshold(image)
    rng = np.random.RandomState(0)
    if len(points):
        pts = np.asarray(points, dtype=np.float64)
        yi = pts[:, 0].astype(int).clip(0, H - 1)
        xi = pts[:, 1].astype(int).clip(0, W - 1)
        pts = pts[image[yi, xi] >= thr]                     # (1) intensity gating
    else:
        pts = np.zeros((0, 2), dtype=np.float64)

    def side_curve(side: str):
        if len(pts) == 0:
            return (np.full(H, np.nan, dtype=np.float64), False)
        cx = pts[:, 1]
        if side == 'L':
            grp = pts[cx <= left_zone * W]
            mid = pts[(cx > left_zone * W) & (cx < right_zone * W)]
            grp = np.concatenate([grp, mid[mid[:, 1] < 0.5 * W]]) if len(mid) else grp
        else:
            grp = pts[cx >= right_zone * W]
            mid = pts[(cx > left_zone * W) & (cx < right_zone * W)]
            grp = np.concatenate([grp, mid[mid[:, 1] >= 0.5 * W]]) if len(mid) else grp
        if len(grp) < min_group_points:
            return (np.full(H, np.nan, dtype=np.float64), False)
        if len(grp) > max_group_points:
            grp = grp[rng.choice(len(grp), max_group_points, replace=False)]
        fit = ransac_near_vertical(grp[:, 0], grp[:, 1], rng=rng)  # (2) robust fit
        if fit is None:
            return (np.full(H, np.nan, dtype=np.float64), False)
        xr = np.polyval(fit, np.arange(H, dtype=np.float64))
        xr = np.where((xr >= 0) & (xr <= W - 1), xr, np.nan)
        snapped = inner_edge_snap(xr, side, image, thr)            # (3) inner-edge snap
        return (snapped, bool(np.isfinite(snapped).any()))

    pred_x_L, exists_L = side_curve('L')
    pred_x_R, exists_R = side_curve('R')
    return pred_x_L, pred_x_R, exists_L, exists_R


def _group_side_points(pts: np.ndarray, side: str, W: int,
                       left_zone: float, right_zone: float) -> np.ndarray:
    """Assign a (y, x) cloud to the left or right side by x zone; middle-zone
    points go to whichever half they fall in."""
    cx = pts[:, 1]
    mid = pts[(cx > left_zone * W) & (cx < right_zone * W)]
    if side == 'L':
        grp = pts[cx <= left_zone * W]
        extra = mid[mid[:, 1] < 0.5 * W]
    else:
        grp = pts[cx >= right_zone * W]
        extra = mid[mid[:, 1] >= 0.5 * W]
    return np.concatenate([grp, extra]) if len(extra) else grp


def edge_inner_curves(segments: np.ndarray, image: np.ndarray,
                      *,
                      left_zone: float = 0.45, right_zone: float = 0.55,
                      min_total_length_px: float = 80.0,
                      near_bright_win: int = 8, marking_band_px: float = 55.0,
                      min_inner_points: int = 30, max_inner_points: int = 3000,
                      n_y_bins: int = 40, inner_quantile: float = 0.85) -> tuple:
    """Inner-edge SELECTION for edge/segment detectors (Canny-Hough, LSD).

    Existence is decided the ordinary way — a side exists when its grouped
    near-vertical segments exceed `min_total_length_px` — so detection recall is
    unchanged. Only the BOUNDARY is refined to the marking's lane-facing inner
    edge, taken from the detector's OWN edges:
      1. discard edges not adjacent to a bright marking (a marking edge has a
         bright pixel within `near_bright_win` px; crack/texture edges do not);
      2. locate the marking with a RANSAC near-vertical line and keep the edge
         points within `marking_band_px` of it (both marking edges);
      3. take the lane-facing inner envelope (high quantile toward the lane) and
         fit it with a RANSAC line.
    If the inner-edge fit cannot lock on, the boundary falls back to the ordinary
    degree-1 fit through the grouped segments, so a detected side always yields a
    curve. Returns (pred_x_L, pred_x_R, exists_L, exists_R).
    """
    H, W = image.shape[:2]
    left_segs, right_segs = group_left_right(
        segments, W, left_zone=left_zone, right_zone=right_zone)
    thr = otsu_bright_threshold(image)
    kern = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * near_bright_win + 1, 1))
    bright = cv2.dilate(image, kern) >= thr
    rng = np.random.RandomState(0)

    def inner_edge(segs, side):
        pts = segments_to_points(segs)
        if len(pts) == 0:
            return None
        yi = pts[:, 0].astype(int).clip(0, H - 1)
        xi = pts[:, 1].astype(int).clip(0, W - 1)
        pts = pts[bright[yi, xi]]                               # (1) drop non-marking edges
        if len(pts) < min_inner_points:
            return None
        if len(pts) > max_inner_points:
            pts = pts[rng.choice(len(pts), max_inner_points, replace=False)]
        fit = ransac_near_vertical(pts[:, 0], pts[:, 1], rng=rng)   # (2) locate marking
        if fit is None:
            return None
        near = np.abs(pts[:, 1] - np.polyval(fit, pts[:, 0])) <= marking_band_px
        gg = pts[near]
        if len(gg) < min_inner_points:
            return None
        bins = np.clip((gg[:, 0] / (H / n_y_bins)).astype(int), 0, n_y_bins - 1)
        ys_e, xs_e = [], []
        for b in np.unique(bins):
            sel = gg[bins == b][:, 1]
            ys_e.append(gg[bins == b][:, 0].mean())
            xs_e.append(np.quantile(sel, inner_quantile if side == 'L'
                                     else 1.0 - inner_quantile))   # (3) inner envelope
        if len(ys_e) < 5 or (max(ys_e) - min(ys_e)) < 200:
            return None
        coefs = ransac_near_vertical(np.asarray(ys_e), np.asarray(xs_e),
                                     tol_px=8.0, iters=200, min_inliers=5,
                                     min_y_span=200.0, rng=rng)
        if coefs is None:
            return None
        xr = np.polyval(coefs, np.arange(H, dtype=np.float64))
        xr = np.where((xr >= 0) & (xr <= W - 1), xr, np.nan)
        return xr if np.isfinite(xr).any() else None

    def side_curve(segs, side):
        # existence: ordinary segment-length rule (recall unchanged) — its
        # degree-1 fit is also the fallback boundary.
        px_fallback, exists = fit_degree1_curve(
            segs, H, W, min_total_length_px=min_total_length_px)
        if not exists:
            return px_fallback, False
        inner = inner_edge(segs, side)
        return (inner if inner is not None else px_fallback), True

    pred_x_L, exists_L = side_curve(left_segs, 'L')
    pred_x_R, exists_R = side_curve(right_segs, 'R')
    return pred_x_L, pred_x_R, exists_L, exists_R
