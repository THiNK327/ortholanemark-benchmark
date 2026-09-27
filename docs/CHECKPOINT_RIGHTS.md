# Checkpoint provenance and redistribution status

Reviewed September 27, 2026. This record covers the 18 selected checkpoint files in the sibling `checkpoints/` package: six learning methods, each with seeds 0, 1 and 2. The checkpoint hashes, training configurations and selection records are preserved in [the run manifest](../provenance/run_manifest.json) and `../../checkpoints/manifest.json`. No weights were changed or newly evaluated during this review.

## Release status

The upstream method code can be redistributed under its applicable licenses, as described in [the third-party notices](../THIRD_PARTY_NOTICES.md). Those code licenses do not, by themselves, establish the terms for every pretrained initialization or for our complete trained checkpoint files.

**The checkpoint companion remains a local release candidate. A blanket CC BY 4.0 license for all 18 checkpoint files has not been established.** CC BY 4.0 remains a proposal for rights the confirmed project rights holder controls; it must not be presented as replacing applicable third-party terms. Publication awaits the authors' release agreement and resolution of the weight-specific licensing scope below. No automatic research-only or noncommercial restriction is assigned by this review, and no explicit prohibition on distributing these fine-tuned checkpoints was found in the sources inspected.

## Initialization provenance

The table records the packaged training code and saved configurations. It does not claim that original download logs or complete hashes of the initialization files were retained.

| Method and run IDs | Initialization | Local evidence |
|---|---|---|
| U-Net: `unet_seg_seed{0,1,2}` | Random initialization; no pretrained model | [Training](../lcms_lane_benchmark/literature/unet_seg/train.py), [model](../lcms_lane_benchmark/literature/unet_seg/model.py) |
| SCNN: `scnn_faithful_seed{0,1,2}` | ImageNet VGG-16-BN from `torchvision.models.vgg16_bn(pretrained=True)` | [Training](../lcms_lane_benchmark/literature/scnn_faithful/train.py), [model](../lcms_lane_benchmark/literature/scnn_faithful/model.py) |
| UFLDv2: `ufldv2_faithful_seed{0,1,2}` | ImageNet ResNet-18 from `torchvision.models.resnet18(pretrained=True)` | [Training](../lcms_lane_benchmark/literature/ufldv2_faithful/train.py), [backbone](../lcms_lane_benchmark/literature/ufldv2_faithful/model/backbone.py), [saved configuration](../configs/learning/ufldv2_faithful_seed0.json) |
| LaneATT: `laneatt_faithful_seed{0,1,2}` | ImageNet ResNet-34 from `torchvision.models.resnet34(pretrained=True)` | [Training](../lcms_lane_benchmark/literature/laneatt_faithful/train.py), [model](../lcms_lane_benchmark/literature/laneatt_faithful/model/laneatt.py), [saved configuration](../configs/learning/laneatt_faithful_seed0.json) |
| CLRNet: `clrnet_faithful_seed{0,1,2}` | ImageNet ResNet-18; direct PyTorch download URL ending `resnet18-5c106cde.pth` | [Detector configuration](../lcms_lane_benchmark/literature/clrnet_faithful/model/detector.py), [download mapping](../lcms_lane_benchmark/literature/clrnet_faithful/model/resnet.py) |
| PolyLaneNet: `polylanenet_faithful_seed{0,1,2}` | Standard ImageNet EfficientNet-B0 from `EfficientNet.from_pretrained('efficientnet-b0')`, using `efficientnet_pytorch==0.6.3` | [Training](../lcms_lane_benchmark/literature/polylanenet_faithful/train.py), [model](../lcms_lane_benchmark/literature/polylanenet_faithful/models.py), [environment record](ENVIRONMENTS.md) |

Thus, three U-Net checkpoints have no pretrained-weight dependency, while 15 checkpoints include parameters fine-tuned from pretrained backbones. All still require the project's own rights-holder and training-data release decisions. The three traditional methods have no trained checkpoint files.

For SCNN, UFLDv2 and LaneATT, the historical torchvision version and initialization-file hashes are not independently recovered. The package's `torchvision==0.20.1` entry is a compatibility pin, as explained in [ENVIRONMENTS.md](ENVIRONMENTS.md), rather than proof of the original training installation. CLRNet's direct URL is [the PyTorch ResNet-18 download](https://download.pytorch.org/models/resnet18-5c106cde.pth).

For PolyLaneNet, the public [EfficientNet-PyTorch 0.6.3 source distribution](https://pypi.org/project/efficientnet-pytorch/0.6.3/#files) was retrieved and inspected for this review. Its SHA-256 is `6667459336893e9bf6367de3788ba449fed97f65da3b6782bf2204b6273a319f`, matching PyPI. Its standard B0 mapping is [efficientnet-b0-355c32eb.pth in release 1.0](https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b0-355c32eb.pth). The packaged training call uses the standard initialization, not the optional AdvProp initialization. This verifies the provider's mapping, not the bytes downloaded during the original training run.

## Verified terms and their limits

| Source | What is verified | What this establishes for our release |
|---|---|---|
| [Torchvision license](https://github.com/pytorch/vision/blob/v0.20.1/LICENSE) and [official pretrained-model documentation](https://docs.pytorch.org/vision/main/models) | The software uses BSD-3-Clause. The documentation says pretrained models may have separate terms, including terms derived from training datasets. | Retain applicable software notices. The library's code license alone is insufficient evidence for an unrestricted replacement license on all initialization weights. No separate restrictive license for the VGG/ResNet files used here was identified in this review. |
| [EfficientNet-PyTorch 0.6.3 metadata](https://pypi.org/project/efficientnet-pytorch/0.6.3/), [release 1.0 license](https://github.com/lukemelas/EfficientNet-PyTorch/blob/1.0/LICENSE), and [release page](https://github.com/lukemelas/EfficientNet-PyTorch/releases/tag/1.0) | The project declares Apache licensing; the release-tag license is Apache-2.0, and the release distributes pretrained weights. | Preserve this provenance and applicable Apache notices. No distinct weight-specific license was found on the release page. The project license is evidence supporting reuse, but this review does not infer a separate CC BY 4.0 grant for our complete PolyLaneNet checkpoints. |
| [ImageNet access terms](https://www.image-net.org/download.php) | The database access terms limit database use to noncommercial research and education. | This package does not redistribute ImageNet images. These terms do not expressly resolve the licensing of the fine-tuned weights here; this review neither transfers the database restriction automatically to the checkpoints nor declares it irrelevant to every use. |

## Decisions needed before checkpoint publication

1. Confirm who can authorize release of the project's trained parameters and that the pavement-data permissions cover the intended release. Paper authorship and article copyright do not establish these rights.
2. Record the rights holder's determination of the applicable pretrained-weight terms for the 15 initialized checkpoints. If the available provider statements do not settle the intended redistribution or relicensing, obtain clarification from the provider or institutional licensing support. No such clarification is claimed here.
3. Assign the final checkpoint terms only for rights the licensor can grant, retaining applicable third-party terms and notices. Align `checkpoints/README.md`, the release metadata and the Zenodo description with that decision. Do not choose a single CC BY 4.0 label that implies unresolved underlying rights have been cleared.

These outstanding checkpoint decisions do not change the measured results or the permissive licensing of the method source. Code, saved results and dataset materials can be prepared separately under their own verified rights and licenses. See [RELEASE_PLAN.md](RELEASE_PLAN.md) for the agreed publication condition.
