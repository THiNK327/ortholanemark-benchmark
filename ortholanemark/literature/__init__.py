"""Literature-method baselines for the OrthoLaneMark benchmark.

Each method emits per-row `(pred_x_L, pred_x_R, exists_L, exists_R)` in
a common per-row boundary representation. All methods
are evaluated under the same clean closure rule via
`ortholanemark.evaluation.metrics.per_image_boundary_errors_full`, so
the comparison is honest — only the prediction source varies.

Tier 1 — classical (no training):
  lsd          — Line Segment Detector (Grompone von Gioi, TPAMI 2010)
  canny_hough  — Canny + Probabilistic Hough (Canny 1986; Matas 2000)
  steger_ridge — Steger-style ridge detection (Steger, TPAMI 1998)

Tier 2 — deep learning (retrained on manifest_paper.train):
  polylanenet, scnn, laneatt, ufldv2 — to be vendored.

Orchestrator: `ortholanemark.scripts.run_literature_benchmark`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class BaseLiteratureMethod(ABC):
    """Single-image lane-boundary predictor with a stable output shape.

    Subclasses implement `predict_image(image_gray)` and set `name`.
    The orchestrator calls `predict_image` per image, then funnels the
    results through `per_image_boundary_errors_full` for the metric
    computation.
    """

    name: str = 'unnamed'

    @abstractmethod
    def predict_image(self, image_gray: np.ndarray) -> dict:
        """Predict left/right boundary curves on one (H, W) grayscale image.

        Returns a dict with:
          - 'pred_x_L'      : np.ndarray (H,) float64.
                              NaN at rows where the predicted left curve is
                              off-frame or missing.
          - 'pred_x_R'      : np.ndarray (H,) float64. NaN similarly.
          - 'pred_exists_L' : bool. True if the method emitted a left curve.
          - 'pred_exists_R' : bool. True if the method emitted a right curve.
          - 'inference_ms'  : float. Wall-clock inference time per image.
        """
        raise NotImplementedError


# Registry. Subclasses register themselves here so the orchestrator can
# discover available methods by name. Filled in at module import time of
# each method file.
METHOD_REGISTRY: dict = {}


def register_method(name: str):
    """Decorator that adds a method class to METHOD_REGISTRY under `name`."""
    def deco(cls):
        cls.name = name
        METHOD_REGISTRY[name] = cls
        return cls
    return deco


def list_methods() -> list:
    _load_all_methods()
    return sorted(METHOD_REGISTRY.keys())


def build_method(name: str, **kwargs) -> BaseLiteratureMethod:
    if name not in METHOD_REGISTRY:
        _load_all_methods()
    if name not in METHOD_REGISTRY:
        raise KeyError(
            f"Unknown literature method '{name}'. "
            f"Available: {sorted(METHOD_REGISTRY.keys())}")
    return METHOD_REGISTRY[name](**kwargs)


def _load_all_methods() -> None:
    """Import every method module so they register themselves.

    Deep methods are guarded so a missing torch dependency (unlikely
    here, but defensive) doesn't take down the classical-only path.
    """
    from ortholanemark.literature import (  # noqa: F401
        lsd, canny_hough, steger_ridge, trivial_none,
    )
    # Faithful (paper-aligned) baselines vendored from each method's
    # official repo under `_vendor/`. The model + loss code is from
    # upstream; only num_lanes / dataset / inference adapter are ours.
    try:
        from ortholanemark.literature.scnn_faithful import (  # noqa: F401
            predict as _scnn_faithful_predict,
        )
    except ImportError:
        pass
    try:
        from ortholanemark.literature.ufldv2_faithful import (  # noqa: F401
            predict as _ufldv2_faithful_predict,
        )
    except ImportError:
        pass
    try:
        from ortholanemark.literature.polylanenet_faithful import (  # noqa: F401
            predict as _polylanenet_faithful_predict,
        )
    except ImportError:
        pass
    try:
        from ortholanemark.literature.laneatt_faithful import (  # noqa: F401
            predict as _laneatt_faithful_predict,
        )
    except ImportError:
        pass
    try:
        from ortholanemark.literature.unet_seg import (  # noqa: F401
            predict as _unet_seg_predict,
        )
    except ImportError:
        pass
    try:
        from ortholanemark.literature.clrnet_faithful import (  # noqa: F401
            predict as _clrnet_faithful_predict,
        )
    except ImportError:
        pass


# Back-compat alias.
_load_classical_methods = _load_all_methods
