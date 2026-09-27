# OrthoLaneMark environment records and limits

All 18 selected checkpoint metadata records report Python 3.12.4, PyTorch 2.5.1, CUDA 12.1, cuDNN 90100, and NVIDIA GeForce RTX 4070. These are recorded training metadata, not a complete dependency lock. The prior 2026-09-23 corrected-decoder analysis records PyTorch 2.5.1, CUDA 11.8, and RTX 4080. Original learning latency values remain associated with RTX 4070. Current verification environment versions are recorded separately in `verification/environment.json`.

`requirements-models.txt` is an installation specification derived from retained dependencies and imports, not a historical lockfile. The Torch version and `efficientnet_pytorch==0.6.3` are pinned where supported by records. `torchvision==0.20.1` is a compatibility pin for Torch 2.5.1, rather than independently recovered proof of the original torchvision version. Remaining packages are unpinned because a complete original environment export is not retained. Install a Torch build compatible with your chosen device. The lightweight offline checks need only NumPy/OpenCV/SciPy. No environment was reconstructed through package installation for this preparation.

## Timing provenance

The retained GPU latency table reports RTX 4070 medians from the shared harness and 100 repetitions. The harness synchronizes CUDA before and after each call, uses a single image, and includes preprocessing, forward computation, decoding and fitting, excluding image loading. It defaults to 20 warm-up calls. The original executed command/raw log is unavailable, so the exact historical warm-up count is not independently verified. Traditional selected-result reports record full-test-set single-thread CPU means under their own validation/evaluation driver, with three warm-up calls. These timing protocols are distinct. No timing was repeated.

The 59.2 M UFLDv2 count is supported by retained checkpoint configuration and includes its auxiliary segmentation head (`use_aux=True`); the exact parameter count inspected was 59,219,993. Both row and column training branches remain in the model, while the evaluated boundary output uses the row decoder. Do not substitute a smaller backbone-only count.

The test runner can run original ground-truth/GRA tests and synthetic raster checks without Torch. Optional retained decoder-stub tests need Torch and EfficientNet dependencies; they use invented arrays and do not load benchmark checkpoints/images. Model inference/training reproducibility was not newly exercised here.
