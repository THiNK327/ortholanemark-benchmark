# Checkpoint provenance and terms

The companion `checkpoints/` package contains 18 selected checkpoints: six learning methods, each with seeds 0, 1, and 2. File hashes, selected epochs, confidence thresholds, and loading paths are recorded in the [run manifest](../provenance/run_manifest.json) and the companion `manifest.json`.

## License scope

The benchmark authors' checkpoint contributions are licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/legalcode), as stated in the companion `LICENSE`. Pretrained components retain their applicable third-party terms; this grant does not replace those terms. Method-source licenses and notices are listed in [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).

Three U-Net checkpoints were trained from random initialization. The other 15 checkpoints were fine-tuned from pretrained backbones. The traditional methods have no trained checkpoints.

## Initialization sources

| Method | Backbone initialization | Implementation |
|---|---|---|
| U-Net | Random initialization | [Model](../lcms_lane_benchmark/literature/unet_seg/model.py) |
| SCNN | ImageNet VGG-16-BN, `torchvision.models.vgg16_bn(pretrained=True)` | [Model](../lcms_lane_benchmark/literature/scnn_faithful/model.py) |
| UFLDv2 | ImageNet ResNet-18, `torchvision.models.resnet18(pretrained=True)` | [Backbone](../lcms_lane_benchmark/literature/ufldv2_faithful/model/backbone.py) |
| LaneATT | ImageNet ResNet-34, `torchvision.models.resnet34(pretrained=True)` | [Model](../lcms_lane_benchmark/literature/laneatt_faithful/model/laneatt.py) |
| CLRNet | ImageNet ResNet-18, PyTorch `resnet18-5c106cde.pth` | [Download mapping](../lcms_lane_benchmark/literature/clrnet_faithful/model/resnet.py) |
| PolyLaneNet | Standard ImageNet EfficientNet-B0, `EfficientNet.from_pretrained('efficientnet-b0')` with `efficientnet_pytorch==0.6.3` | [Model](../lcms_lane_benchmark/literature/polylanenet_faithful/models.py) |

Torchvision's software is BSD-3-Clause, and its [pretrained-model documentation](https://docs.pytorch.org/vision/main/models.html) explains that model weights may carry separate provider or training-data terms. EfficientNet-PyTorch [release 1.0](https://github.com/lukemelas/EfficientNet-PyTorch/releases/tag/1.0) distributes the B0 weights and retains an [Apache-2.0 project license](https://github.com/lukemelas/EfficientNet-PyTorch/blob/1.0/LICENSE). These software licenses do not establish a replacement CC BY license for all underlying pretrained parameters.

## Reproduction details

The table is supported by the packaged training code and selected configurations. Original initialization download logs and complete file hashes were not retained. For SCNN, UFLDv2, and LaneATT, the historical torchvision version is also not independently recovered; the installation pin in [ENVIRONMENTS.md](ENVIRONMENTS.md) is a compatibility choice.

CLRNet uses [PyTorch's ResNet-18 file](https://download.pytorch.org/models/resnet18-5c106cde.pth). EfficientNet-PyTorch 0.6.3 maps standard B0 to [efficientnet-b0-355c32eb.pth](https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b0-355c32eb.pth). Its source distribution has SHA-256 `6667459336893e9bf6367de3788ba449fed97f65da3b6782bf2204b6273a319f`; this verifies the provider mapping, not the bytes downloaded by an original training run. PolyLaneNet uses standard initialization, rather than AdvProp weights.

Use the selected configurations and checkpoint hashes in the run manifest for reproduction. See [REPRODUCTION.md](REPRODUCTION.md) for the model interfaces.
