# Evaluated source and release notices

The evaluated model/evaluator source is identified by the original hashes in `provenance/source_manifest.json`. Those file hashes remain unchanged as historical evidence.

Before redistribution, nine Python files received comment prefixes identifying upstream components or documenting modifications. No executable statements, function bodies, docstrings, constants or model configurations were changed. For each file, `provenance/source_notice_changes.json` records its evaluated hash, packaged hash, and the exact added UTF-8 prefix. Removing that prefix reconstructs the evaluated file byte-for-byte.

Run:

```sh
python tools/verify_sources.py
```

The check validates all 84 evaluated files, verifies each recorded prefix contains only comments/blank lines, reconstructs original bytes, and compares Python abstract syntax trees without source-location attributes. New package checksums refer to the packaged files; historical run manifests continue to identify the evaluated source. This distinction preserves the audit trail rather than rewriting historical checkpoint or evaluation provenance.

The retained original source files are also backed up outside the distributable package. No private archive is required to run the public verification command.

See [third-party notices](../THIRD_PARTY_NOTICES.md) for the component license mapping and [method adaptations](METHOD_ADAPTATIONS.md) for differences from upstream experimental setups.
