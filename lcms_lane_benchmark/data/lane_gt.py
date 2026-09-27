"""Canonical per-side geometric ground truth for the LCMS lane-boundary benchmark.

Hard contract enforced by this module:

  The polynomial extends continuously across the full image height.
  `clicked_y_range` is annotator bookkeeping (literally min/max y of
  `clicked_points`); it is NOT a visibility window, NOT a polynomial
  bound, and NOT "where GT exists". No public API in this module
  exposes `clicked_y_range` as a polynomial bound. The original
  contamination bug (see plan §0 Context) used `y_range` as a row
  mask in 18 modules — this module's contract is the regression
  fence that prevents it from creeping back in.

See plan §2.2 for the full dataclass spec and §6.0 for the Phase A0
clean-GT sidecar schema this dataclass also reads.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class LaneGeometricTruth:
    """Geometric ground truth for one lane side.

    Attributes
    ----------
    has_marking : bool
        Whether a marking was annotated for this side at all. When False,
        `poly_coefs` is a length-1 zero array and the other geometric
        fields are placeholders — downstream code must check
        `has_marking` before treating the polynomial as meaningful.
    poly_coefs : np.ndarray
        Polynomial coefficients in NumPy polyfit order (highest degree
        first), in full-image pixel space: x_px = sum_k a_k * y_px^k.
    poly_degree : int
        1 (line) or 2 (quadratic). 0 when `has_marking=False`.
    clicked_y_range : tuple[int, int] | None
        DIAGNOSTIC ONLY. The y-span of `clicked_points` (literally
        (min y, max y) of the clicked dots). Storing it lets us
        stratify metrics later ("MAE within click span vs outside")
        without using it as a polynomial bound.
    clicked_points : list[tuple[int, int]]
        QA only. The 2-4 (x, y) dots the annotator clicked.
    """

    has_marking: bool
    poly_coefs: np.ndarray
    poly_degree: int
    clicked_y_range: Optional[tuple] = None
    clicked_points: list = field(default_factory=list)

    # ------------------------------------------------------------------
    # Evaluation helpers
    # ------------------------------------------------------------------
    def eval_x_at(self, ys: np.ndarray, W: int,
                  clamp_to_frame: bool = True) -> np.ndarray:
        """Polynomial value at given y positions.

        Parameters
        ----------
        ys : np.ndarray
            Y coordinates (same coord frame as `poly_coefs`).
        W : int
            Image width (in the same coord frame). Used only to decide
            the in-frame condition.
        clamp_to_frame : bool, default True
            True  → x clipped to [0, W-1]. Operational eval semantics
                    (the inference rasterizer clamps to image edges
                    when a curve goes off-frame, so eval must match).
            False → x is NaN where the polynomial extrapolates off-frame
                    (x < 0 or x >= W). Training-target semantics — we
                    must NOT supervise the model toward image edges
                    where the lane has actually exited the frame.
        """
        if not self.has_marking:
            return np.full_like(np.asarray(ys, dtype=np.float64), np.nan
                                if not clamp_to_frame
                                else 0.0)
        ys = np.asarray(ys, dtype=np.float64)
        x = np.polyval(self.poly_coefs, ys)
        if clamp_to_frame:
            return np.clip(x, 0.0, float(W - 1))
        off = (x < 0) | (x > float(W - 1))
        x = np.where(off, np.nan, x)
        return x

    def in_frame_mask(self, H: int, W: int) -> np.ndarray:
        """(H,) bool: True iff polynomial evaluates to x ∈ [0, W) at
        every image row in [0, H). All-False when has_marking=False."""
        if not self.has_marking:
            return np.zeros(int(H), dtype=bool)
        ys = np.arange(int(H), dtype=np.float64)
        x = np.polyval(self.poly_coefs, ys)
        return (x >= 0.0) & (x <= float(W - 1))

    # ------------------------------------------------------------------
    # Crop-local view (closed-form coordinate substitution)
    # ------------------------------------------------------------------
    def to_crop_view(self, crop_y0: int, crop_x0_padded: int,
                     pad_left: int, crop_h: int, crop_w: int
                     ) -> 'LaneGeometricTruth':
        """Return a NEW LaneGeometricTruth with poly_coefs rewritten in
        crop-local pixel coords (y ∈ [0, crop_h), x ∈ [0, crop_w)).

        The relationship between full-image and crop-local coords:
            y_full  = y_crop + crop_y0
            x_full  = polyval(coefs, y_full)
            x_pad   = x_full + pad_left
            x_crop  = x_pad - crop_x0_padded
                    = x_full + (pad_left - crop_x0_padded)

        Substituting y_full = y_crop + crop_y0 into x_full gives a new
        polynomial in y_crop. Closed-form for degree 1 and 2 — the
        only degrees in this dataset (~96% deg 1, ~4% deg 2).
        """
        delta = float(pad_left) - float(crop_x0_padded)
        c0 = float(crop_y0)

        if not self.has_marking:
            return LaneGeometricTruth(
                has_marking=False,
                poly_coefs=np.zeros(1, dtype=np.float64),
                poly_degree=0,
                clicked_y_range=None,
                clicked_points=[],
            )

        coefs = np.asarray(self.poly_coefs, dtype=np.float64)
        deg = int(self.poly_degree)

        if deg == 1:
            a1, a0 = float(coefs[0]), float(coefs[1])
            new_coefs = np.array(
                [a1, a1 * c0 + a0 + delta], dtype=np.float64)
        elif deg == 2:
            a2, a1, a0 = float(coefs[0]), float(coefs[1]), float(coefs[2])
            new_coefs = np.array([
                a2,
                2.0 * a2 * c0 + a1,
                a2 * c0 * c0 + a1 * c0 + a0 + delta,
            ], dtype=np.float64)
        else:
            raise ValueError(
                f"to_crop_view supports poly_degree ∈ {{1, 2}}, got {deg}")

        # Translate clicked_y_range and clicked_points; drop points outside
        # the crop (diagnostic-only data, OK to lose precision here).
        new_click_range = None
        if self.clicked_y_range is not None:
            y_min, y_max = self.clicked_y_range
            if y_min is not None and y_max is not None:
                y_min_c = int(y_min) - int(crop_y0)
                y_max_c = int(y_max) - int(crop_y0)
                y_min_c = max(0, min(int(crop_h) - 1, y_min_c))
                y_max_c = max(0, min(int(crop_h) - 1, y_max_c))
                if y_min_c <= y_max_c:
                    new_click_range = (y_min_c, y_max_c)

        new_points = []
        for px, py in self.clicked_points:
            px_c = int(px) + int(pad_left) - int(crop_x0_padded)
            py_c = int(py) - int(crop_y0)
            if 0 <= px_c < int(crop_w) and 0 <= py_c < int(crop_h):
                new_points.append((px_c, py_c))

        return LaneGeometricTruth(
            has_marking=True,
            poly_coefs=new_coefs,
            poly_degree=deg,
            clicked_y_range=new_click_range,
            clicked_points=new_points,
        )

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------
    @classmethod
    def from_lane_dict(cls, lane: dict) -> Optional['LaneGeometricTruth']:
        """Construct from a raw annotation JSON entry (per-side block).

        Returns None only when the dict is structurally malformed
        (missing `has_marking`). When `has_marking=False`, returns a
        valid LaneGeometricTruth with has_marking=False and zero
        polynomial (the dataset uses it for the existence label only).
        """
        if 'has_marking' not in lane:
            return None
        has_marking = bool(lane['has_marking'])
        if not has_marking:
            return cls(
                has_marking=False,
                poly_coefs=np.zeros(1, dtype=np.float64),
                poly_degree=0,
                clicked_y_range=None,
                clicked_points=[],
            )

        coefs = lane.get('polynomial_coefficients')
        degree = lane.get('polynomial_degree')
        if coefs is None or degree is None:
            return None
        coefs = np.asarray(coefs, dtype=np.float64)

        click_range = None
        yr = lane.get('y_range')
        if yr is not None:
            y_min = yr.get('min')
            y_max = yr.get('max')
            if y_min is not None and y_max is not None:
                click_range = (int(y_min), int(y_max))

        points = []
        for px, py in lane.get('clicked_points', []) or []:
            points.append((int(px), int(py)))

        return cls(
            has_marking=True,
            poly_coefs=coefs,
            poly_degree=int(degree),
            clicked_y_range=click_range,
            clicked_points=points,
        )

    @classmethod
    def from_clean_gt_dict(cls, lane: dict) -> Optional['LaneGeometricTruth']:
        """Construct from a Phase-A0 sidecar JSON entry (per-side block).

        Sidecar schema (see plan §6.0) differs from raw in:
          - `y_range` renamed to `clicked_y_range` (the rename itself
            is the regression fence).
          - `x_full` and `in_frame_mask` stored explicitly. Ignored at
            load time (recomputed from the polynomial); their presence
            is just a redundancy check that lives in the file.

        Otherwise behaves identically to `from_lane_dict`.
        """
        if 'has_marking' not in lane:
            return None
        has_marking = bool(lane['has_marking'])
        if not has_marking:
            return cls(
                has_marking=False,
                poly_coefs=np.zeros(1, dtype=np.float64),
                poly_degree=0,
                clicked_y_range=None,
                clicked_points=[],
            )

        coefs = lane.get('polynomial_coefficients')
        degree = lane.get('polynomial_degree')
        if coefs is None or degree is None:
            return None
        coefs = np.asarray(coefs, dtype=np.float64)

        click_range = None
        yr = lane.get('clicked_y_range')
        if yr is not None:
            y_min = yr.get('min')
            y_max = yr.get('max')
            if y_min is not None and y_max is not None:
                click_range = (int(y_min), int(y_max))

        points = []
        for px, py in lane.get('clicked_points', []) or []:
            points.append((int(px), int(py)))

        return cls(
            has_marking=True,
            poly_coefs=coefs,
            poly_degree=int(degree),
            clicked_y_range=click_range,
            clicked_points=points,
        )
