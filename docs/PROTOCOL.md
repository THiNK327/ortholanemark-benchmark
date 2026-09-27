# OrthoLaneMark benchmark protocol

The benchmark uses grayscale pavement-intensity images from six surveyed road sections. The 1,391 annotated images comprise 835 training, 192 validation, 294 test, and 70 buffer-excluded images. Exact membership and order are defined by the companion dataset manifests. The splits contain spatially separated blocks within the same sections; they do not represent an unseen-route evaluation.

## Model and threshold selection

Each of the six learning methods has three 100-epoch runs, using seeds 0, 1, and 2. The first epoch attaining the maximum validation GRA supplies each run's `best_gra.pth` checkpoint. Checkpoint metadata agrees with all 18 training histories. Method-specific training-time side-presence rules are recorded in [run_manifest.json](../provenance/run_manifest.json).

For each selected run, a final confidence threshold shared by the left and right sides was selected on the 192 validation images from 0 to 1 in steps of 0.02. The test decision is `confidence >= threshold`, so zero confidence passes at threshold zero. Each run's selected threshold is recorded in the run manifest.

The learning results correspond to the 2026-09-23 decoder audit, which corrected full-height reconstruction while preserving checkpoint hashes, selected thresholds, and presence decisions. The corrections changed localization values. Traditional results use the 2026-09-21 validation selection. For these methods, use `configs/traditional/*_effective.json`, which combines selected overrides with constructor defaults.

The Steger method is a multiscale Hessian/ridge adaptation with shared boundary extraction. Its constructor fields `left_zone`, `right_zone`, `min_pixels`, and `min_y_span` are inactive in the evaluated call path. The active helper uses zones 0.45/0.55, a minimum of 30 and maximum of 3,000 group points, and a minimum RANSAC y span of 200 pixels. These settings are documented in the effective configuration.

## Output and evaluation conventions

Adapters return left/right x coordinates for every image row in original pixel coordinates, together with confidence and native presence decisions. Prediction reconstruction is separate from the annotation polynomials. The controlling implementations are [evaluation/metrics.py](../lcms_lane_benchmark/evaluation/metrics.py) and [scripts/wheelpath_diagnostic.py](../lcms_lane_benchmark/scripts/wheelpath_diagnostic.py).

GRA equals lane-region intersection-over-union only when both side-presence decisions match annotations; otherwise it is zero. Absence substitutes the corresponding image edge for evaluation, without claiming a physical lane boundary. For predictions, nonfinite rows also use the corresponding edge; finite coordinates are clipped to 0 through W-1. Raster regions include integer columns between ceil(left) and floor(right); reversed rows are empty. Zero-union region overlap is one. Equal integer-valued boundaries include one pixel, whereas equal noninteger coordinates can include none. The width used by legacy area/width diagnostics must not be conflated with inclusive raster pixel counts.

Localization is conditional on correctly detected present sides and uses eligible rows from the unmodified evaluator; nonfinite predictions are excluded. MAE combines per-side row means with equal weight within each image, followed by image-level averaging. Within-tolerance fractions instead pool eligible rows across the two sides within each image, followed by image-level averaging; images with no eligible rows are excluded. Millimeter conversion uses 4 mm per transverse pixel. The within-tolerance comparisons are strict `<`, rather than `<=`. Aggregate presence precision/recall/F1 pools the two side decisions; the older report's side-macro F1 is not the paper's pooled F1. The offline table script reconstructs pooled F1 directly from per-image presence flags. Learning means and standard deviations use three seeds and population standard deviation (`ddof=0`).

Wheelpath zones are centered on the boundary midpoint; each 1 m strip extends outward from an inner edge 0.375 m from center. Per-image disagreement is 1 minus IoU of the two-strip unions and is averaged without presence gating. Reversed rows are invalid, and zero union yields zero disagreement. The diagnostic compares predicted and annotation-derived zones; it does not validate observed vehicle tracks or measured distress changes.

The synthetic checks exercise absence, clipping, nonfinite values, crossings, empty/equal regions, localization eligibility, strict tolerances, confidence-zero behavior, and full-height linear/quadratic reconstruction. See [reproduction instructions](REPRODUCTION.md) for the commands.
