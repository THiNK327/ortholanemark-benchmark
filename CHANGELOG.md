# Changelog

## 1.0.0

Initial OrthoLaneMark benchmark package accompanying *Presence-Aware Benchmarking of Lane-Marking Detection Methods for Pavement Distress Measurement*.

- Nine evaluated methods: Canny-Hough, LSD, Steger ridge, U-Net, SCNN, UFLDv2, PolyLaneNet, LaneATT, and CLRNet.
- Eighteen learning-run configurations, training histories, selected checkpoint records, and validation-selected traditional-method settings.
- Saved predictions and per-image scores, with tools for table reconstruction, saved-prediction evaluation, source verification, and synthetic evaluation checks.
- Stable image and section identifiers shared with the companion dataset of 1,391 intensity images, 1,391 paired range-image exports, annotations, and fixed split memberships.
- PNG imagery preserving decoded pixel values, image modes, and dimensions.
- Method-adaptation documentation, environment records, component license texts, attribution notices, and verifiable evaluated-source hashes.

Learning results correspond to the 2026-09-23 corrected-decoder evaluation. Traditional results correspond to the 2026-09-21 validation selection. Dataset packaging and comment-only source notices preserve the evaluated configurations, numeric predictions, annotation values, split memberships, and checkpoint bytes. Checkpoint weights are provided as a separate companion package.
