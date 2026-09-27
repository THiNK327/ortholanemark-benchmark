# OrthoLaneMark third-party notices and provenance

This package contains adapted source from the projects below. Their license and copyright notices remain applicable to their components and are not replaced by the license chosen for benchmark-authored code. Dataset images, annotations, and checkpoint weights have separate release terms; the source-code licenses below do not establish permission to redistribute those materials.

## Method implementations

The five upstream model licenses are retained verbatim in the indicated directories. The pinned commits identify the local upstream snapshots used to prepare the benchmark; upstream executable code was not upgraded during release preparation.

| Method | Recorded upstream and revision | License and retained notice |
|---|---|---|
| SCNN | [harryhan618/SCNN_Pytorch](https://github.com/harryhan618/SCNN_Pytorch/tree/bbe0acff4681bcda7e984f867dd31fdaa9a7bf81) | [MIT](lcms_lane_benchmark/literature/scnn_faithful/LICENSE_upstream), Copyright (c) 2019 HarryHan |
| UFLDv2 | [cfzd/Ultra-Fast-Lane-Detection-V2](https://github.com/cfzd/Ultra-Fast-Lane-Detection-V2/tree/c903880678454dfd9b55a63022368db05c00bc6d) | [MIT](lcms_lane_benchmark/literature/ufldv2_faithful/LICENSE_upstream), Copyright (c) 2022 zequn qin |
| PolyLaneNet | [lucastabelini/PolyLaneNet](https://github.com/lucastabelini/PolyLaneNet/tree/6155ce2e7d3841c46a0035a25acc9d2e304d9856) | [MIT](lcms_lane_benchmark/literature/polylanenet_faithful/LICENSE_upstream), Copyright (c) 2020 Lucas Tabelini Torres |
| LaneATT | [lucastabelini/LaneATT](https://github.com/lucastabelini/LaneATT/tree/2f8583ba14eccba05e6779668bc3a38bc751984a) | [MIT](lcms_lane_benchmark/literature/laneatt_faithful/LICENSE_upstream), Copyright (c) 2021 Lucas Tabelini; additional components listed below |
| CLRNet | [Turoad/CLRNet](https://github.com/Turoad/CLRNet/tree/7269e9d1c1c650343b6c7febb8e764be538b1aed) | [Apache-2.0](lcms_lane_benchmark/literature/clrnet_faithful/LICENSE_upstream); additional components listed below |

Canny-Hough and LSD are benchmark wrappers calling separately installed OpenCV functions. The Steger-inspired ridge detector and U-Net architecture are implemented in the benchmark source; no separate original-author source distribution for either method is bundled. Their research citations describe the methods and do not replace the applicable software licenses.

Task-specific adapters, grayscale preprocessing, two-side targets, fitted boundary output, and the shared Python lane-NMS differ from full upstream training and evaluation systems. Source headers describe these adaptations. This package does not claim exact reproduction of the original upstream experiments.

## Components included within the method source

Paths in the following table are relative to `lcms_lane_benchmark/literature/`. Licenses are included even for supporting code that the selected benchmark configuration does not execute.

| Included component | Packaged files and attribution | License documents |
|---|---|---|
| Kornia focal-loss and one-hot helpers | `laneatt_faithful/model/focal_loss.py` and `clrnet_faithful/model/focal_loss.py`; derived from [Kornia commit f4f70fefb63287f72bc80cd96df9c061b1cb60dd](https://github.com/kornia/kornia/tree/f4f70fefb63287f72bc80cd96df9c061b1cb60dd). Historical copyright notices identify Arraiy, Inc. (2017-2019), Open Source Vision Foundation (2019-), and Kornia authors (2019-). | [Apache-2.0](LICENSES/Kornia-Apache-2.0.txt) and the exact historical [COPYRIGHT](LICENSES/Kornia-COPYRIGHT.txt) |
| CIFAR ResNet implementation by Yerlan Idelbayev | `laneatt_faithful/model/resnet.py`; source identifies [akamaster/pytorch_resnet_cifar10](https://github.com/akamaster/pytorch_resnet_cifar10). Copyright (c) 2018, Yerlan Idelbayev. | [BSD-2-Clause](LICENSES/Idelbayev-ResNet-BSD-2-Clause.txt) |
| Lane-NMS port | `laneatt_faithful/model/nms_stub.py`, also imported by `clrnet_faithful/model/clr_head.py`. The Python implementation ports the [LaneATT NMS kernel](https://github.com/lucastabelini/LaneATT/tree/2f8583ba14eccba05e6779668bc3a38bc751984a/lib/nms). The retained notice identifies Grégoire Payen de La Garanderie, Durham University (2018), and incorporated Faster R-CNN material, Microsoft Corporation (2015). | Exact upstream [BSD-3-Clause and MIT notice chain](LICENSES/LaneATT-NMS-BSD-3-Clause-and-MIT.txt) |
| Torchvision ResNet code included through CLRNet | `clrnet_faithful/model/resnet.py`; shared ResNet blocks and helpers match [torchvision source](https://github.com/pytorch/vision/blob/v0.6.0/torchvision/models/resnet.py). Copyright (c) Soumith Chintala 2016. | [BSD-3-Clause](LICENSES/Torchvision-BSD-3-Clause.txt) |
| MMDetection FPN and accuracy code included through CLRNet | `clrnet_faithful/model/fpn.py` and `clrnet_faithful/model/accuracy.py`; source comparison confirms MMDetection ancestry. Copyright (c) OpenMMLab. All rights reserved. Reference source: [FPN](https://github.com/open-mmlab/mmdetection/blob/v2.19.1/mmdet/models/necks/fpn.py), [accuracy](https://github.com/open-mmlab/mmdetection/blob/v2.19.1/mmdet/models/losses/accuracy.py). | [Apache-2.0](LICENSES/MMDetection-Apache-2.0.txt) |

The Kornia source also credits [zhezh/focalloss](https://github.com/zhezh/focalloss/blob/master/focalloss.py) as an algorithmic basis. That acknowledgment is retained here; this package includes the Kornia PyTorch implementation, not the linked TensorFlow source.

Torchvision v0.6.0 and MMDetection v2.19.1 are comparison references used to identify embedded source ancestry. They are not assertions about the exact dependency versions used in benchmark training or the precise revisions originally incorporated by CLRNet. The downloaded license bytes, source URLs, applicable paths, and SHA-256 hashes are recorded in [LICENSES/manifest.json](LICENSES/manifest.json). The existing method-level license files are also covered by the package checksum manifest.

No separate `NOTICE` file was found in the retained CLRNet source tree or the inspected Kornia and MMDetection reference roots. Relevant copyright notices and full license texts are retained above. Redistribution must preserve these notices, include the applicable licenses, and retain modification notices for Apache-licensed files; project and contributor names do not imply endorsement.

## Separately installed dependencies

Runtime dependencies include PyTorch, torchvision, NumPy, SciPy, OpenCV, Pillow, tqdm, and EfficientNet-PyTorch. Their own licenses govern those separately installed packages. This distribution does not bundle framework binaries, compiled CUDA extensions, complete upstream repository trees, or original pretrained ImageNet checkpoint downloads. The source portions expressly identified above are bundled and retain their licenses regardless of whether the complete dependency is installed separately.

License notices were completed on September 27, 2026. See [LICENSE_STATUS.md](LICENSE_STATUS.md) for the current release-license status and [docs/RELEASE_PLAN.md](docs/RELEASE_PLAN.md) for the publication plan.
