# OrthoLaneMark paired range-image exports

The companion dataset includes a range-image export for each of its 1,391 intensity images, using the same stable image IDs and split membership. Files are stored at `range/section_*/olm_######.png` within the dataset package: 1,245 grayscale (`L`) and 146 RGB PNGs, each matching its paired intensity image's dimensions. PNG conversion preserves decoded pixels and image modes; source files already in PNG retain their original bytes.

The dataset's `manifests/image_manifest.jsonl` records availability, paths, encoding, dimensions and SHA-256 hashes. Physical units/scale and invalid-value codes are undocumented; calibrated registration has not been independently verified. See the dataset's `docs/RANGE.md`, `reports/range_pairing_report.json` and `reports/png_conversion_report.json`.

The reported benchmark uses intensity imagery only. Range exports are provided as an additional dataset resource and were not used for training or evaluation.
