# Implementation and provenance

These records describe the evaluated OrthoLaneMark pipelines. Internal `_faithful` identifiers preserve run associations; they do not claim exact reproduction of upstream experiments. Architecture, preprocessing, optimization, and output conversion all affect the comparison.

## Method adaptations

Input sizes are height × width. Learning adapters directly resize with OpenCV `INTER_AREA`, repeat grayscale across three channels, and divide by 255. SCNN, UFLDv2, and PolyLaneNet additionally apply ImageNet channel normalization. Resizing changes aspect ratio; no padding preserves it. Predictions use original coordinates over the full image height; reconstructed off-image positions are undefined.

| Method | Input | Evaluated components and boundary conversion |
|---|---|---|
| Canny–Hough | Original grayscale | OpenCV Canny/Hough; Otsu intensity filtering, side grouping, inner-edge refinement, and linear RANSAC fitting. |
| LSD | Original grayscale | OpenCV line segments; orientation, length, and support filters; shared Canny–Hough boundary fitting. |
| Steger ridge adaptation | Original grayscale | Gaussian/Sobel-Hessian responses at 2/4/6/8-pixel scales, 97th-percentile filtering; ridge RANSAC followed by connected-bright-run inner-edge selection and linear refit or ridge fallback. Not the original subpixel algorithm. |
| U-Net | 800 × 320 | Local three-class encoder-decoder, batch normalization, 64 base channels; smoothed row maxima, quadratic fit, row-support presence confidence. |
| SCNN | 800 × 320 | Dilated VGG-16-BN, spatial message passing, two-side segmentation/existence heads; smoothed row maxima and quadratic fit. |
| UFLDv2 | 800 × 320 | ResNet-18; row/column branches and auxiliary head train; evaluation uses row coordinates/existence only. Four accepted anchors give a linear fit; five or more give a quadratic fit. |
| LaneATT | 360 × 640 | ResNet-34, attention, 1,000 anchors selected by the included frequency tensor; Python NMS. Proposal positions determine sides; 2–4 usable coordinates give linear fits, ≥5 quadratic fits. |
| CLRNet | 320 × 800 | ResNet-18, FPN, iterative prior refinement; shared replacement NMS and LaneATT fit-degree/side-assignment rules, with its own proposal decoder. |
| PolyLaneNet | 360 × 640 | EfficientNet-B0; direct cubic regression with two positional slots and side confidence; predicted vertical extents do not truncate boundaries. |

Exact rules are in [method adapters](../ortholanemark/literature/) and the [traditional helper](../ortholanemark/literature/_common.py). Steger's constructor fields `left_zone`, `right_zone`, `min_pixels`, and `min_y_span` are inactive: the helper uses zones 0.45/0.55, 30–3,000 points, and a 200-pixel minimum RANSAC y span. Use [effective traditional settings](../configs/traditional/).

## Selection and evaluation

Learning runs use seeds 0/1/2, 100 epochs, and training-only rotations up to 2°. Backbones, batch sizes, optimizers, and schedules differ; equal epochs do not imply equal computational budgets. The first maximum-validation-GRA epoch supplies each checkpoint, using training-time presence rules. Final validation thresholds are selected separately over 0–1 in steps of 0.02; `confidence >= threshold` allows zero confidence at threshold zero. [Run records](run_manifest.json) and [configurations](../configs/learning/) control reproduction.

Learning results use the September 23, 2026 full-height decoder corrections; checkpoints and thresholds were not reselected. Traditional selection is dated September 21. Localization averages eligible present-side errors per image; within-tolerance fractions pool eligible rows within each image and use strict `<`. Conversion is 4 mm/pixel. Presence F1 pools both sides; learning standard deviations use `ddof=0`.

The [evaluator](../ortholanemark/evaluation/metrics.py) controls absence/nonfinite substitution, clipping, and inclusive raster bounds: columns `ceil(left)` through `floor(right)`, with reversed rows empty and zero-union IoU equal to one. [Wheelpath evaluation](../ortholanemark/scripts/wheelpath_diagnostic.py) uses ungated union disagreement; zero union gives zero disagreement. Synthetic checks cover these edge conventions.

## Environment and timing

Training metadata records Python 3.12.4, PyTorch 2.5.1, CUDA 12.1, cuDNN 90100, and RTX 4070. Corrected-decoder analysis records PyTorch 2.5.1, CUDA 11.8, and RTX 4080. [Offline verification](../verification/environment.json) has its own environment. UFLDv2's 59,219,993 parameters include its auxiliary head.

GPU latency uses RTX 4070 medians over 100 synchronized single-image calls, including preprocessing/forward/decoding/fitting but excluding loading. The harness defaults to 20 warm-ups; the actual count is unverified because execution logs are unavailable. Traditional timings are single-thread CPU means over the test set after three warm-ups. These protocols differ; replacement Python NMS timings do not represent upstream CUDA performance.

## Source records

[Source hashes](source_manifest.json) identify current distributed files and retain historical evaluated hashes as references. `python tools/verify_sources.py`, run from the repository root, checks current hashes, coverage, and comment-only [attribution prefixes](source_notices.json); it does not reconstruct historical source. Original source/edit records remain in the authors' archive. [Checkpoint metadata updates](checkpoint_metadata_updates.json) record path-only changes separately from unchanged tensor storage. Saved-result checks do not constitute fresh training or model-inference validation.
