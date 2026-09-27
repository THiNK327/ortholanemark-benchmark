# OrthoLaneMark environments and timing

All 18 selected checkpoints record the following training environment:

| Component | Recorded value |
|---|---|
| Python | 3.12.4 |
| PyTorch | 2.5.1 |
| CUDA | 12.1 |
| cuDNN | 90100 |
| GPU | NVIDIA GeForce RTX 4070 |

The 2026-09-23 corrected-decoder analysis records PyTorch 2.5.1, CUDA 11.8, and an RTX 4080. Learning latency values come from the RTX 4070 measurements. The offline verification environment is recorded separately in [verification/environment.json](../verification/environment.json).

## Dependencies

[requirements-models.txt](../requirements-models.txt) supplies dependencies for the model interfaces. PyTorch and `efficientnet_pytorch==0.6.3` are pinned from the experiment records. `torchvision==0.20.1` is a compatibility pin for PyTorch 2.5.1; the original torchvision version was not independently recovered. Other dependencies are unpinned because a complete training-environment export is unavailable. Select a PyTorch build compatible with your device.

The lightweight offline checks need only NumPy, OpenCV, and SciPy, listed in [requirements-verification.txt](../requirements-verification.txt). These checks verify saved outputs and evaluation behavior; they do not establish that a newly installed model environment reproduces training or inference.

## Timing provenance

The GPU latency table reports RTX 4070 medians from the shared harness over 100 repetitions. The harness synchronizes CUDA before and after each call, processes one image, and includes preprocessing, forward computation, decoding, and fitting; image loading is excluded. Its default is 20 warm-up calls. The original executed command and raw log are unavailable, so the actual warm-up count cannot be independently verified.

Traditional-method reports contain full-test-set, single-thread CPU means from their validation/evaluation driver, with three warm-up calls. These CPU and GPU timing protocols are distinct.

## Model size and verification scope

The UFLDv2 parameter count is 59,219,993 (59.2 M), including its auxiliary segmentation head (`use_aux=True`). Both row and column training branches remain in the model, while evaluated boundaries use the row decoder. A backbone-only count does not describe this configuration.

The test runner supports ground-truth/GRA tests and synthetic raster checks without PyTorch. Optional decoder-stub tests require PyTorch and EfficientNet dependencies; they use synthetic arrays without loading benchmark checkpoints or images. The package verification covers saved-result reconstruction and evaluation checks, not a fresh training or inference run.
