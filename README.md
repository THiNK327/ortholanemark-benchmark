# OrthoLaneMark Benchmark

Planned repository: `THiNK327/ortholanemark-benchmark`, hosted on Haolin Wang's personal GitHub account. The dataset name, repository owner, and release creators are confirmed. Public release awaits agreement of all authors.

Local release candidate for *Presence-Aware Benchmarking of Lane-Marking Detection Methods for Pavement Distress Measurement*. This package contains the frozen implementation, selected configurations, saved scores and predictions, and offline verification tools for the nine-method intensity-only benchmark. It has not been uploaded or published. The release uses the manuscript's four authors and order, with Zhongyu Yang as corresponding contact. Funding: none. Proposed licenses and the ASCE release plan are recorded in [docs/RELEASE_PLAN.md](docs/RELEASE_PLAN.md); actual DOIs, public URLs and the release date remain unset. See `RELEASE_METADATA.json`.

The companion **OrthoLaneMark: A Pavement Survey Dataset for Lane-Marking Detection** (`dataset/`) package contains 1,391 intensity images with paired range-image exports: 835 train, 192 validation, 294 test, and 70 buffer-excluded. Membership is frozen. Splits comprise spatially separated blocks within the same six surveyed sections, rather than unseen-route evaluation. All intensity and range images use PNG with decoded pixels preserved. Range exports share the same stable image IDs; the reported experiments use intensity imagery only. See `docs/RANGE.md`.

## Included records

- `lcms_lane_benchmark/`: the evaluated benchmark implementation, with comment-only redistribution notices on nine files, including nine method implementations, native model cores, data readers, GRA/localization evaluator, wheelpath evaluator, and historical tests.
- `configs/`: all 18 learning-run configurations extracted from selected checkpoint metadata; selected traditional overrides and effective constructor settings.
- `artifacts/`: retained per-image validation/test scores, selected test predictions, learning validation/test confidence arrays, threshold grids, histories, and traditional selection records. Only identifiers and local paths were normalized; every numeric prediction array was checked for exact dtype/value preservation.
- `provenance/`: historical evaluated-source hashes, recoverable comment-only notice changes, run/checkpoint hashes, artifact transformations and package checksums. See [source provenance](docs/SOURCE_PROVENANCE.md).
- `LICENSES/` and `THIRD_PARTY_NOTICES.md`: retained exact component license texts and a file-to-license mapping.
- `tools/`: portable, offline verification and manifest-path utilities. These additions do not modify historical metric or decoder computations.

The 18 selected checkpoint files are copied into the separate sibling `checkpoints/` candidate and excluded from this code candidate. Their hashes, byte sizes, epochs, final thresholds, and expected `weights/<run_id>/best_gra.pth` paths are in `provenance/run_manifest.json`. The checkpoint companion is planned for the benchmark Zenodo record; public release awaits agreement of all authors and the weight-specific rights determination in [CHECKPOINT_RIGHTS.md](docs/CHECKPOINT_RIGHTS.md). A blanket CC BY 4.0 license for all checkpoints has not been established.

## Run the documentation checks

From this directory, with Python and the dependencies in `requirements-verification.txt`:

```sh
python tools/verify_integrity.py
python tools/verify_sources.py
python tools/rebuild_tables.py
python tools/run_synthetic_checks.py
python tools/check_edge_cases.py
```

The first command checks packaged files. The source check verifies all 84 evaluated-source hashes, reconstructing nine files after removal of recorded comment prefixes and confirming identical syntax trees. The table command reconstructs results from saved scores and checks learning mean/population-standard-deviation values against their retained summary. The other commands test synthetic geometries without real images or weights. To also run the retained synthetic Torch decoder stubs, install the model dependencies and run `python tools/run_synthetic_checks.py --decoders`.

Resolve the companion dataset's portable manifest for original readers:

```sh
python tools/prepare_manifest.py --dataset-root ../dataset --manifest ../dataset/manifests/manifest_paper.json --output work/manifest_resolved.json
```

The generated manifest contains local absolute paths for your machine and is ignored by Git. All shared manifests use package-relative paths and anonymized stable IDs. See `docs/REPRODUCTION.md` for saved-prediction evaluation and model interfaces; `docs/PROTOCOL.md` for frozen selection/aggregation; [METHOD_ADAPTATIONS.md](docs/METHOD_ADAPTATIONS.md) for all nine methods' task adaptations; and `docs/ENVIRONMENTS.md` for historical environments and verification limits.

For release preparation, run `python tools/release_preflight.py` with the dataset and checkpoint siblings present. It reports technical checks separately from pending author, rights and identifier decisions. `--publication` returns a nonzero status while those decisions remain unset. Follow [UPLOAD_GUIDE.md](docs/UPLOAD_GUIDE.md) after the final review.

No training, inference on benchmark images, threshold selection, split changes, or timing runs were performed to prepare this candidate. Saved records dated 2026-09-23 describe an earlier decoder correction; those are the learning results associated with this draft. Older release/dist snapshots are superseded and were not copied as current benchmark evidence.
