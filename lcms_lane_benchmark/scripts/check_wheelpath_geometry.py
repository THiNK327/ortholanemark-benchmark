"""Independent raster checks for the midpoint-derived wheelpath metric."""
import numpy as np
from lcms_lane_benchmark.scripts.wheelpath_diagnostic import disagreement


def raster(left, right, width):
    center = (left + right) / 2
    x = np.arange(width)[None, :] * 0.004
    c = center[:, None] * 0.004
    mask = ((x >= c - 1.375) & (x <= c - 0.375)) | ((x >= c + 0.375) & (x <= c + 1.375))
    return mask & (left <= right)[:, None]


def main():
    rng = np.random.default_rng(20260922)
    for _ in range(100):
        bounds = [rng.uniform(0, 1039, size=30) for _ in range(4)]
        p, g = raster(*bounds[:2], 1040), raster(*bounds[2:], 1040)
        union = (p | g).sum()
        expected = 1 - (p & g).sum() / union if union else 0
        assert abs(disagreement(*bounds, 1040) - expected) < 1e-12
    l, r = np.full(10, 70.), np.full(10, 970.)
    assert disagreement(l, r, l, r, 1040) == 0
    assert disagreement(l - 10, r + 10, l, r, 1040) == 0
    assert abs(disagreement(l + 10, r + 10, l, r, 1040) - 20 / 260) < 1e-12
    assert disagreement(r, l, l, r, 1040) == 1
    print('Passed: 100 independent raster comparisons, identical boundaries, cancelling errors, common translation, and crossed boundaries.')


if __name__ == '__main__':
    main()
