# GitHub and Zenodo upload guide

This package is prepared locally. Publish after the agreed all-author release review; no command here uploads automatically.

## Final local review

Run from `repository/`, with its dataset and checkpoint siblings present:

```sh
python tools/verify_integrity.py
python tools/verify_sources.py
python tools/rebuild_tables.py --output work/reconstructed_tables.json
python tools/run_synthetic_checks.py
python tools/check_edge_cases.py
python tools/release_preflight.py
```

The final command distinguishes technical preparation from unresolved publication decisions. `python tools/release_preflight.py --publication` returns a nonzero status while those decisions or actual record identifiers remain unset. The check reports recorded status; it cannot make a legal rights determination for the authors.

Review [RELEASE_PLAN.md](RELEASE_PLAN.md) and [CHECKPOINT_RIGHTS.md](CHECKPOINT_RIGHTS.md). The proposed license files are drafts for the release review. They do not grant licenses or establish unresolved ownership. After the final decisions, activate the corresponding license texts, update the metadata and refresh package checksums. Keep historical evaluated-source hashes and checkpoint hashes unchanged.

## GitHub

1. Add `repository/` to GitHub Desktop under **File > Add local repository**. The intended owner/name is `THiNK327/ortholanemark-benchmark`.
2. Review the file list and create the initial commit. Saved prediction NPZ files under `artifacts/` are intentional: the Git ignore exception includes them. The dataset and trained checkpoints remain outside this repository.
3. Publish the repository only after the author release decision. Record the actual URL in `RELEASE_METADATA.json` and the citation/documentation files.
4. After final metadata edits, create the paper's version tag/release (`v1.0.0`). Preserve this exact source version in the benchmark archive.

If Git reports dubious ownership on this Windows drive, use the Git client's explicit trust action for this repository only. No global trust setting was changed during preparation.

## Two Zenodo records

Create two drafts and reserve their DOIs. A reserved unpublished DOI does not establish public availability.

| Record | Files and resource type |
|---|---|
| OrthoLaneMark Benchmark: Code, Results, and Checkpoints | Software; archive containing `repository/` and `checkpoints/` |
| OrthoLaneMark: A Pavement Survey Dataset for Lane-Marking Detection | Dataset; archive containing `dataset/` |

Use the confirmed creator lists, affiliations/ORCIDs and no-funding metadata. Link the companion record, GitHub release and paper when identifiers exist. Identify different licenses by component; do not apply a blanket MIT or CC BY label to material governed by retained third-party terms. The benchmark checkpoint license remains unresolved as documented in the rights review.

Archives should preserve the three sibling folder names. Exclude `.git`, caches, the manuscript, private source crosswalks and preparation archives. Include all saved predictions, dataset manifests/splits and verification documents. Zip with ZIP64 support because the bundles exceed traditional ZIP limits. A new snapshot must be generated after any final metadata edit; older archives are superseded.

## Manuscript and release identifiers

Use actual version-specific Zenodo DOIs and the actual GitHub release URL in the manuscript's Data Availability Statement and references. The release plan distinguishes ASCE Technical Paper and Data Paper timing. Update the statement during peer review, before acceptance. Do not fill a public-access claim while the underlying records remain unpublished.

Official instructions: [GitHub Desktop](https://docs.github.com/en/desktop/adding-and-cloning-repositories/adding-an-existing-project-to-github-using-github-desktop), [Zenodo drafts and DOI reservation](https://help.zenodo.org/docs/deposit/create-new-upload/), [Zenodo file preparation](https://help.zenodo.org/docs/deposit/manage-files/).
