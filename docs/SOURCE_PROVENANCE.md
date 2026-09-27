# Source provenance

[provenance/source_manifest.json](../provenance/source_manifest.json) identifies the 84 distributed source and resource files by their current paths, byte counts, and SHA-256 hashes. It also retains the corresponding historical evaluated hashes and byte counts as audit references.

The Python package is named `ortholanemark`. Release preparation updated package names, imports, paths, documentation strings, and the LaneATT anchor-resource resolver. Model computations, numeric settings, and evaluation rules were checked for equivalence during preparation. The saved-result checks described in [REPRODUCTION.md](REPRODUCTION.md) provide additional verification without rerunning model training or inference.

Nine Python files contain attribution comment prefixes recorded in [source_notices.json](../provenance/source_notices.json). Run:

```sh
python tools/verify_sources.py
```

This command checks the current hashes and byte counts, confirms that the manifest covers every distributed package file apart from caches, and verifies that all nine attribution prefixes remain present and contain only comments or blank lines. It also checks that removing each prefix leaves the current file's abstract syntax tree unchanged.

Historical hashes identify the source used for the reported evaluations; they are not reconstructed by this command. The original source and the exact release-edit records are retained in a separate author archive and are not included in this distribution. Package checksums verify the files as distributed.

See [third-party notices](../THIRD_PARTY_NOTICES.md) for component licenses and [method adaptations](METHOD_ADAPTATIONS.md) for differences from upstream experimental setups.
