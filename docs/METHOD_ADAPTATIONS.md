# Method adaptations and comparison scope

This benchmark evaluates nine specified pipelines for left and right lane-marking inner boundaries in pavement-survey intensity images. It does not reproduce the complete experimental setups of the original papers. The learning implementations retain their principal model components while adapting input geometry, supervision, training, and output conversion to this task. Performance therefore reflects both architecture and implementation choices.

Internal identifiers ending in `_faithful` are stable run and provenance identifiers. The suffix does not imply that preprocessing, training schedules, decoding, or reported performance exactly reproduce an upstream experiment.

## Inputs and experimental protocol

All methods use the same intensity-image collection and fixed training, validation, and test memberships. The paired range-image exports are supplied for future use and were not inputs to the reported experiments. Traditional methods operate on the original grayscale image. Learning methods directly resize the image using OpenCV `INTER_AREA`, replicate it across three channels, and divide intensities by 255. SCNN, UFLDv2, and PolyLaneNet additionally apply ImageNet channel normalization; U-Net, LaneATT, and CLRNet do not. Direct resizing changes aspect ratio; there is no aspect-preserving padding in these inference adapters.

All six learning methods have three retained runs (seeds 0, 1, and 2), trained for 100 epochs with training-only rotations of up to 2 degrees. Batch sizes, backbones, optimizers, and learning-rate schedules differ. Equal epochs do not imply equal optimization steps or computational budgets. U-Net starts without pretrained weights; the other learning methods use ImageNet-pretrained backbones. These are documented benchmark configurations, not an exhaustive search for each method's best performance.

Checkpoint selection used validation GRA and method-specific training-time presence decisions. Final presence thresholds were selected separately for each run on validation data. Checkpoints were not reselected after the recorded full-height decoder corrections. See [Protocol](PROTOCOL.md), the [18 learning configurations](../configs/learning/), and [traditional effective configurations](../configs/traditional/).

## Method-specific adaptations

Input sizes below are **height x width**. Learning-method geometry is expressed in original-image coordinates and evaluated over the full original image height; off-image reconstructed positions are undefined. The table describes the evaluated adapters, not every optional constructor setting.

| Method | Input | Components and task adaptations | Boundary conversion |
|---|---|---|---|
| Canny-Hough | Original grayscale | OpenCV Canny and probabilistic Hough, with Otsu-derived intensity thresholds and validation-selected segment filters. No learned weights. | Near-vertical segments are grouped by side; shared intensity-guided inner-edge refinement and first-degree RANSAC fitting produce boundaries. |
| LSD | Original grayscale | OpenCV's default line-segment detector, with validation-selected orientation, segment-length, and cumulative-support filters. No learned weights. | Uses the same side grouping and inner-edge fitting helper as Canny-Hough. |
| Steger ridge adaptation | Original grayscale | Multiscale Gaussian/Sobel-Hessian bright-ridge response at scales 2, 4, 6, and 8 pixels, thresholded at the 97th percentile. This is not the original Steger subpixel localization algorithm. | Intensity filtering, side grouping, and RANSAC locate a ridge; connected bright-run tracing selects the lane-facing edge before a first-degree refit or ridge fallback. |
| U-Net | 800 x 320 | Local three-class encoder-decoder with skip connections, batch normalization, and 64 base channels; background and two positional boundary classes. Adam with constant learning rate. | Smoothed side-probability maps supply row maxima for quadratic fitting. Side confidence is the fraction of rows with sufficient segmentation support. |
| SCNN | 800 x 320 | Referenced PyTorch SCNN core with dilated VGG-16-BN, spatial message passing, and segmentation/existence heads adapted to two sides. SGD with Nesterov momentum and polynomial decay. | Smoothed side maps supply row maxima for quadratic fitting; the existence head supplies side confidence. |
| UFLDv2 | 800 x 320 | ResNet-18 with row/column localization and existence branches; two sides per branch, full-height row anchors, and an auxiliary segmentation head. Both branches train. SGD with warm-up and stepwise decay. | Both evaluated boundaries use row-branch coordinates and existence scores. Four accepted anchors give a linear fit; five or more give a quadratic fit. Column outputs do not supply evaluated boundaries. |
| LaneATT | 360 x 640 | ResNet-34 and attention-based anchor proposals; two-boundary targets and a retained training-set anchor-frequency tensor selecting 1,000 anchors. Tensor/Python NMS replaces the compiled CUDA operator. Adam with cosine decay. | Retained proposals are assigned to left/right by fitted horizontal position. Two to four usable strip coordinates give a linear fit; five or more give a quadratic fit. |
| CLRNet | 320 x 800 | ResNet-18, FPN, and iterative lane-prior refinement; two-boundary targets and adapted data/configuration integration. Uses the same replacement NMS module as LaneATT. AdamW with cosine decay. | Retained proposals undergo the same fit-degree and side-assignment rules described for LaneATT, using CLRNet's own proposal decoding and thresholds. |
| PolyLaneNet | 360 x 640 | EfficientNet-B0 with direct cubic regression and two positional output slots. Adam with cosine decay. | Predicted cubics are evaluated directly. Predicted vertical extents do not truncate the benchmark's full-height boundaries; each slot supplies its own confidence. |

The exact geometry, confidence, and fallback rules are implemented in each method's `predict.py` under [literature/](../lcms_lane_benchmark/literature/) and in the traditional [shared helper](../lcms_lane_benchmark/literature/_common.py). Output adaptation can affect localization and presence results; it is part of each evaluated pipeline. Replacement NMS is an implementation adaptation, not a basis for claiming original CUDA runtime performance.

## Recorded upstream provenance

These pins identify the reference sources recorded in local provenance; they do not imply that entire repositories were copied unchanged or that upstream experiments were rerun. U-Net and the three traditional pipelines are local implementations using the components described above, without a recorded upstream repository pin.

| Integration | Recorded reference repository | Commit |
|---|---|---|
| SCNN | `harryhan618/SCNN_Pytorch` | `bbe0acff4681bcda7e984f867dd31fdaa9a7bf81` |
| UFLDv2 | `cfzd/Ultra-Fast-Lane-Detection-V2` | `c903880678454dfd9b55a63022368db05c00bc6d` |
| LaneATT | `lucastabelini/LaneATT` | `2f8583ba14eccba05e6779668bc3a38bc751984a` |
| CLRNet | `Turoad/CLRNet` | `7269e9d1c1c650343b6c7febb8e764be538b1aed` |
| PolyLaneNet | `lucastabelini/PolyLaneNet` | `6155ce2e7d3841c46a0035a25acc9d2e304d9856` |

See [third-party notices](../THIRD_PARTY_NOTICES.md) for retained licenses, [environments and timing limits](ENVIRONMENTS.md), and [reproduction instructions](REPRODUCTION.md). The reported ranking applies to these configurations, splits, and adapters; it does not isolate architectural effects or establish a universal ranking of the original methods.
