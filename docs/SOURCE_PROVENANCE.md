# Evaluated-source provenance

The model and evaluator source used for the reported experiments is identified by 84 original file hashes in [provenance/source_manifest.json](../provenance/source_manifest.json).

Nine packaged Python files include added comment prefixes identifying upstream components or documenting modifications. Their executable statements, function bodies, docstrings, constants, and model configurations are unchanged. For each file, [source_notice_changes.json](../provenance/source_notice_changes.json) records the evaluated hash, packaged hash, and exact added UTF-8 prefix. Removing that prefix reconstructs the evaluated file byte-for-byte.

Run:

```sh
python tools/verify_sources.py
```

The check validates all 84 evaluated files, verifies that each recorded prefix contains only comments or blank lines, reconstructs original bytes, and compares Python abstract syntax trees without source-location attributes.

Package checksums identify the files as distributed; run manifests identify the source used during evaluation. Both versions can be verified using the repository alone, without a separate source archive.

See [third-party notices](../THIRD_PARTY_NOTICES.md) for the component license mapping and [method adaptations](METHOD_ADAPTATIONS.md) for differences from upstream experimental setups.
