# OrthoLaneMark local candidate change record

## 2026-09-27 pre-upload preparation

- Added exact historical Kornia license/copyright, BSD ResNet/NMS, torchvision and MMDetection notices, with component scopes and hashes.
- Added comment-only attribution/modification prefixes to nine source files. All 84 original evaluated-source hashes remain verifiable; AST equivalence and original-byte reconstruction are checked by `tools/verify_sources.py`.
- Added the nine-method adaptation table, checkpoint initialization/rights review, upload guide and offline release preflight. Saved NPZ prediction files are now included by Git's ignore rules.
- Prepared scoped license proposals while keeping unresolved author/rights decisions explicit. No upload, training, inference or model-result change was performed.

## 2026-09-27 release metadata

- Confirmed manuscript-matching creators/order, affiliations, ORCIDs, corresponding contact, personal GitHub owner THiNK327, and no funding.
- Prepared version 1.0.0 metadata for separate benchmark and dataset Zenodo records. Publication remains pending agreement of all authors.
- Recorded proposed MIT code and CC BY 4.0 dataset/checkpoint licenses separately from actual license grants, plus ASCE Technical Paper/Data Paper access requirements.
- Clarified that the preserved source is the evaluated benchmark source. Models, images, annotations, splits, saved predictions and checkpoint bytes were not changed.

## 2026-09-26 PNG standardization

- Standardized all 1,391 intensity and 1,391 range images to PNG: converted 2,490 JPEGs and retained the 292 existing PNGs byte-for-byte. Decoded pixels, image modes and dimensions are preserved.
- Updated paths, hashes, annotation filename references and package documentation; conversion checks are in the dataset's `reports/png_conversion_report.json`. Original sources remain unmodified and the converted package JPEGs are retained in the sibling archive.
- Annotation values, splits, checkpoints and reported benchmark results are unchanged. No experiments were rerun.

## 2026-09-26 paired range exports

- Added all 1,391 matching range-image exports to the companion dataset with stable image IDs, recorded encoding/dimensions and SHA-256 hashes.
- Updated package documentation and metadata. At this stage, the frozen intensity images, annotations, splits, checkpoints and reported benchmark results were unchanged. No range experiment was run.

## Author-approved naming

- Adopted OrthoLaneMark and the official dataset title *OrthoLaneMark: A Pavement Survey Dataset for Lane-Marking Detection*.
- Set the repository name to `ortholanemark-benchmark` and renamed the code/checkpoint archives.
- Preserved internal module/model/run/image identifiers, source code, numeric artifacts, and checkpoint bytes. Earlier archives remain retained as superseded historical candidates.

## 2026-09-24 preparation

- Froze and hashed original Python source before adding portability utilities.
- Copied current method/evaluator source byte-for-byte; retained upstream license notices and LaneATT anchor-frequency tensor.
- Indexed all 18 existing selected checkpoints, checked SHA-256 against prior audit records, and matched selected epochs to checkpoint metadata and histories.
- Extracted exact stored learning configurations and traditional selected/effective settings without training or selection.
- Packaged retained score/prediction artifacts using stable companion-dataset IDs. Checked numeric arrays for exact preservation after identifier-only changes.
- Added offline score reconstruction, saved-prediction evaluation, integrity checks, synthetic geometry checks, and portable manifest resolution.
- Prepared pending citation/release metadata; omitted incomplete range and retained large weights externally.

Learning result provenance is the existing 2026-09-23 corrected-decoder record. Traditional results use the existing 2026-09-21 selected records. This preparation did not rerun inference, revise results, or reselect thresholds/checkpoints. Earlier generated dist/release metadata are superseded as release instructions.
