# OrthoLaneMark paired range-image exports

The companion dataset includes a range-image export for each of its 1,391 intensity images, with the same stable image IDs and frozen split membership. Files are at `../dataset/range/section_*/olm_######.png`: 1,245 grayscale (`L`) and 146 RGB PNGs, all matching their intensity image's dimensions. PNG conversion preserves decoded pixels and image modes; existing PNGs retain their original bytes.

The dataset's `manifests/image_manifest.jsonl` records availability, paths, encoding, dimensions and SHA-256 hashes. Physical units/scale and invalid-value codes are undocumented; calibrated registration has not been independently verified. See the dataset's `docs/RANGE.md`, `reports/range_pairing_report.json` and `reports/png_conversion_report.json`.

The benchmark readers continue to use intensity imagery. Annotation values, splits, decoded model inputs, checkpoints and reported results are unchanged; no range experiment was run.
