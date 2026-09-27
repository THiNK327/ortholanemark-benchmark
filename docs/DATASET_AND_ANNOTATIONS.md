# OrthoLaneMark dataset and annotation interface

The companion `dataset/` package, **OrthoLaneMark: A Pavement Survey Dataset for Lane-Marking Detection**, contains 1,391 intensity images, paired range-image exports, annotations, and fixed split memberships. Its image manifest records stable anonymized image IDs, six section IDs, split or exclusion assignment, file paths, geometry, and checksums. Both image types use PNG, preserving decoded pixels, image modes, and dimensions; see [range-image documentation](RANGE.md).

Sequence IDs follow section and acquisition-order records. They do not encode GPS locations or exact acquisition times. The original acquisition-filename mapping is private and is not needed to use the dataset.

## Annotation geometry

The dataset's annotation report defines label semantics and compatibility fields. Accepted polynomial coefficients use NumPy order (highest power first) and original pixel coordinates, with x a function of y. Curves are sampled over the full image height. The recorded click range is annotation bookkeeping, not a visibility interval. Compact records preserve the accepted coefficients without refitting; compatibility records support the original readers. Absent sides have an explicit absence flag and null geometry in compact records, while compatibility readers use their documented absent-side convention.

A full-height curve does not establish visible paint at every row. Original clicks and annotation history are included where available. Consensus adjudication involved two annotators and a third reviewer; the records do not contain a complete vote history or an independent third annotation. The two-pixel reviewer-agreement tolerance is an annotation-review criterion, not a calibrated measure of absolute uncertainty.

## Portable manifest

`manifests/manifest_paper.json` contains the `splits` dictionary and per-image entries with `project`, `stem`, `image_path`, and `clean_gt_path`. Paths are relative to the dataset root. The compatibility names `project` and `stem` correspond to the public `section_id` and `image_id`. Saved predictions and score records use the same IDs and ordering.

Run [tools/prepare_manifest.py](../tools/prepare_manifest.py) to validate the manifest and resolve its paths for the original dataset readers. The command is provided in [REPRODUCTION.md](REPRODUCTION.md).

## Saved prediction format

`artifacts/learning/<run_id>/raw_val.npz` and `raw_test.npz` contain image/section IDs (`stems`, `projects`), full-height boundary coordinates (`xL`, `xR`), continuous confidence scores (`cL`, `cR`), native presence decisions, and image dimensions. `predictions_test.npz` also contains the final thresholded presence decisions (`existsL`, `existsR`).

Load the packaged NPZ files with `numpy.load(..., allow_pickle=False)`. Every array is numeric, boolean, or Unicode. The source module `evaluation/predictions_io.py` describes an older object-array format; use the packaged NPZ schema above for these artifacts.
