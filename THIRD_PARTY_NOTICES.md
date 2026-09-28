# OrthoLaneMark third-party notices and provenance

This package contains adapted source from the projects below. Their license and copyright notices remain applicable to their components and are not replaced by the MIT license for benchmark-authored code. Dataset images, annotations, and checkpoint weights have separate release terms; the source-code licenses below do not establish permission to redistribute those materials.

## License scope

Copyright (c) 2026 Haolin Wang, Shiwei Luo, Zhongyu Yang, and Yi-Chang J. Tsai, for the benchmark authors' contributions.

| Material | License and scope |
|---|---|
| Benchmark-authored code, tools, configurations, and documentation | [MIT](LICENSE) |
| Third-party source and adaptations | Original terms below; exact texts retained alongside code and in `LICENSES/` |
| Authors' saved scores, predictions, and analysis in `artifacts/` and `verification/` | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/legalcode) |
| [README figure](assets/annotation_examples.png) | CC BY 4.0; unchanged manuscript Figure 3 panel pixels, with source and composition hashes in [figure provenance](provenance/readme_figure.json) |
| Companion dataset images, annotations, splits, and documentation | CC BY 4.0, as specified in the [dataset README](../dataset/README.md#license) |
| Authors' contributions to companion checkpoints | CC BY 4.0, as specified in the checkpoint package's `LICENSE`; pretrained components retain their applicable terms |

These grants cover the benchmark authors' contributions. Cite the benchmark and retain license and attribution notices. The CC BY 4.0 grants do not replace third-party licenses.

## Method implementations

The five upstream model licenses are retained verbatim in the indicated directories. The pinned commits identify the upstream snapshots used by the benchmark.

| Method | Recorded upstream and revision | License and retained notice |
|---|---|---|
| SCNN | [harryhan618/SCNN_Pytorch](https://github.com/harryhan618/SCNN_Pytorch/tree/bbe0acff4681bcda7e984f867dd31fdaa9a7bf81) | [MIT](ortholanemark/literature/scnn/LICENSE_upstream), Copyright (c) 2019 HarryHan |
| UFLDv2 | [cfzd/Ultra-Fast-Lane-Detection-V2](https://github.com/cfzd/Ultra-Fast-Lane-Detection-V2/tree/c903880678454dfd9b55a63022368db05c00bc6d) | [MIT](ortholanemark/literature/ufldv2/LICENSE_upstream), Copyright (c) 2022 zequn qin |
| PolyLaneNet | [lucastabelini/PolyLaneNet](https://github.com/lucastabelini/PolyLaneNet/tree/6155ce2e7d3841c46a0035a25acc9d2e304d9856) | [MIT](ortholanemark/literature/polylanenet/LICENSE_upstream), Copyright (c) 2020 Lucas Tabelini Torres |
| LaneATT | [lucastabelini/LaneATT](https://github.com/lucastabelini/LaneATT/tree/2f8583ba14eccba05e6779668bc3a38bc751984a) | [MIT](ortholanemark/literature/laneatt/LICENSE_upstream), Copyright (c) 2021 Lucas Tabelini; additional components listed below |
| CLRNet | [Turoad/CLRNet](https://github.com/Turoad/CLRNet/tree/7269e9d1c1c650343b6c7febb8e764be538b1aed) | [Apache-2.0](ortholanemark/literature/clrnet/LICENSE_upstream); additional components listed below |

Canny-Hough and LSD are benchmark wrappers calling separately installed OpenCV functions. The Steger-inspired ridge detector and U-Net architecture are implemented in the benchmark source; no separate original-author source distribution for either method is bundled. Their research citations describe the methods and do not replace the applicable software licenses.

Task-specific adapters, grayscale preprocessing, two-side targets, fitted boundary output, and the shared Python lane-NMS differ from full upstream training and evaluation systems. Source headers describe these adaptations. This package does not claim exact reproduction of the original upstream experiments.

## Components included within the method source

Paths in the following table are relative to `ortholanemark/literature/`. Licenses are included even for supporting code that the selected benchmark configuration does not execute.

| Included component | Packaged files and attribution | License documents |
|---|---|---|
| Kornia focal-loss and one-hot helpers | `laneatt/model/focal_loss.py` and `clrnet/model/focal_loss.py`; derived from [Kornia commit f4f70fefb63287f72bc80cd96df9c061b1cb60dd](https://github.com/kornia/kornia/tree/f4f70fefb63287f72bc80cd96df9c061b1cb60dd). Historical copyright notices identify Arraiy, Inc. (2017-2019), Open Source Vision Foundation (2019-), and Kornia authors (2019-). | [Apache-2.0](LICENSES/Kornia-Apache-2.0.txt) and the exact historical [COPYRIGHT](LICENSES/Kornia-COPYRIGHT.txt) |
| CIFAR ResNet implementation by Yerlan Idelbayev | `laneatt/model/resnet.py`; source identifies [akamaster/pytorch_resnet_cifar10](https://github.com/akamaster/pytorch_resnet_cifar10). Copyright (c) 2018, Yerlan Idelbayev. | [BSD-2-Clause](LICENSES/Idelbayev-ResNet-BSD-2-Clause.txt) |
| Lane-NMS port | `laneatt/model/nms_stub.py`, also imported by `clrnet/model/clr_head.py`. The Python implementation ports the [LaneATT NMS kernel](https://github.com/lucastabelini/LaneATT/tree/2f8583ba14eccba05e6779668bc3a38bc751984a/lib/nms). The retained notice identifies Grégoire Payen de La Garanderie, Durham University (2018), and incorporated Faster R-CNN material, Microsoft Corporation (2015). | Exact upstream [BSD-3-Clause and MIT notice chain](LICENSES/LaneATT-NMS-BSD-3-Clause-and-MIT.txt) |
| Torchvision ResNet code included through CLRNet | `clrnet/model/resnet.py`; shared ResNet blocks and helpers match [torchvision source](https://github.com/pytorch/vision/blob/v0.6.0/torchvision/models/resnet.py). Copyright (c) Soumith Chintala 2016. | [BSD-3-Clause](LICENSES/Torchvision-BSD-3-Clause.txt) |
| MMDetection FPN and accuracy code included through CLRNet | `clrnet/model/fpn.py` and `clrnet/model/accuracy.py`; source comparison confirms MMDetection ancestry. Copyright (c) OpenMMLab. All rights reserved. Reference source: [FPN](https://github.com/open-mmlab/mmdetection/blob/v2.19.1/mmdet/models/necks/fpn.py), [accuracy](https://github.com/open-mmlab/mmdetection/blob/v2.19.1/mmdet/models/losses/accuracy.py). | [Apache-2.0](LICENSES/MMDetection-Apache-2.0.txt) |

The Kornia source also credits [zhezh/focalloss](https://github.com/zhezh/focalloss/blob/master/focalloss.py) as an algorithmic basis. That acknowledgment is retained here; this package includes the Kornia PyTorch implementation, not the linked TensorFlow source.

Torchvision v0.6.0 and MMDetection v2.19.1 are comparison references used to identify embedded source ancestry. They are not assertions about the exact dependency versions used in benchmark training or the precise revisions originally incorporated by CLRNet. The downloaded license bytes, source URLs, applicable paths, and SHA-256 hashes are recorded in [LICENSES/manifest.json](LICENSES/manifest.json). The existing method-level license files are also covered by the package checksum manifest.

No separate `NOTICE` file was found in the retained CLRNet source tree or the inspected Kornia and MMDetection reference roots. Relevant copyright notices and full license texts are retained above. Redistribution must preserve these notices, include the applicable licenses, and retain modification notices for Apache-licensed files; project and contributor names do not imply endorsement.

## Separately installed dependencies

Runtime dependencies include PyTorch, torchvision, NumPy, SciPy, OpenCV, Pillow, tqdm, and EfficientNet-PyTorch. Their own licenses govern those separately installed packages. This distribution does not bundle framework binaries, compiled CUDA extensions, complete upstream repository trees, or original pretrained ImageNet checkpoint downloads. The source portions expressly identified above are bundled and retain their licenses regardless of whether the complete dependency is installed separately.

## Checkpoint initialization

The companion package contains 18 selected checkpoints: seeds 0, 1, and 2 for each learning method. Three U-Net checkpoints use random initialization; the other 15 use pretrained backbones. Traditional methods have no trained checkpoints. Selected epochs, thresholds, loading paths, and hashes are in the [run manifest](provenance/run_manifest.json) and companion `manifest.json`.

| Method | Initialization | Implementation |
|---|---|---|
| U-Net | Random | [Model](ortholanemark/literature/unet_seg/model.py) |
| SCNN | ImageNet VGG-16-BN, `torchvision.models.vgg16_bn(pretrained=True)` | [Model](ortholanemark/literature/scnn/model.py) |
| UFLDv2 | ImageNet ResNet-18, `torchvision.models.resnet18(pretrained=True)` | [Backbone](ortholanemark/literature/ufldv2/model/backbone.py) |
| LaneATT | ImageNet ResNet-34, `torchvision.models.resnet34(pretrained=True)` | [Model](ortholanemark/literature/laneatt/model/laneatt.py) |
| CLRNet | ImageNet ResNet-18, PyTorch `resnet18-5c106cde.pth` | [Download mapping](ortholanemark/literature/clrnet/model/resnet.py) |
| PolyLaneNet | Standard ImageNet EfficientNet-B0, `EfficientNet.from_pretrained('efficientnet-b0')`, `efficientnet_pytorch==0.6.3` | [Model](ortholanemark/literature/polylanenet/models.py) |

Torchvision software is BSD-3-Clause; its [model documentation](https://docs.pytorch.org/vision/main/models.html) explains that weights may have separate provider or training-data terms. EfficientNet-PyTorch [release 1.0](https://github.com/lukemelas/EfficientNet-PyTorch/releases/tag/1.0) distributes B0 weights under a project retaining an [Apache-2.0 license](https://github.com/lukemelas/EfficientNet-PyTorch/blob/1.0/LICENSE). These software licenses do not establish a replacement CC BY license for pretrained parameters.

Initialization sources are supported by packaged training code and configurations; original download logs and full weight-file hashes were not retained. The historical torchvision version for SCNN, UFLDv2, and LaneATT is unverified; the version in [requirements-models.txt](requirements-models.txt) is a compatibility pin.

CLRNet maps to [PyTorch's ResNet-18 file](https://download.pytorch.org/models/resnet18-5c106cde.pth). EfficientNet-PyTorch 0.6.3 maps standard B0 to [efficientnet-b0-355c32eb.pth](https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b0-355c32eb.pth), not AdvProp weights. Its source-distribution SHA-256 is `6667459336893e9bf6367de3788ba449fed97f65da3b6782bf2204b6273a319f`; this verifies the provider mapping, not the original training download.
