# OrthoLaneMark release plan

Prepared September 27, 2026 for the ASCE **Journal of Computing in Civil Engineering**. This is a local preparation record. No repository or Zenodo record was created, no upload was performed, and no publication date is assigned.

## Confirmed metadata

The benchmark and dataset share the manuscript's creator order: **Haolin Wang, Shiwei Luo, Zhongyu Yang, Yi-Chang J. Tsai**. All four affiliations and ORCIDs are recorded in the citation files. The corresponding contact is **Zhongyu Yang, zyang398@gatech.edu**. Funding: **none**, confirmed by the author. The acknowledgment of Raghu Veerareddy's assistance with data preparation is retained.

The intended repository is `THiNK327/ortholanemark-benchmark` on [Haolin Wang's personal GitHub account](https://github.com/THiNK327). This is a planned repository path, not a verified public release. Actual repository URL and DOI fields remain unset.

## Planned records

| Destination | Title and contents | Initial version |
|---|---|---|
| GitHub | `ortholanemark-benchmark`: repository code, configurations, saved results and documentation | 1.0.0 |
| Zenodo benchmark record | **OrthoLaneMark Benchmark: Code, Results, and Checkpoints**; `repository/` and `checkpoints/` | 1.0.0 |
| Zenodo dataset record | **OrthoLaneMark: A Pavement Survey Dataset for Lane-Marking Detection**; `dataset/` | 1.0.0 |

Extract the archives into sibling `repository/`, `dataset/`, and `checkpoints/` folders. Exclude `.git`, private preparation archives and manuscript files from deposits. Preserve the benchmark source and numeric artifacts when finalizing metadata.

## Proposed licenses for the coauthor review

| Material | Proposed terms | Scope |
|---|---|---|
| Benchmark-authored code | MIT | New code that the confirmed rights holder can license; upstream components retain their existing notices and licenses |
| Dataset images and annotations | CC BY 4.0 | The separately deposited dataset |
| Selected checkpoint files | CC BY 4.0 proposed; scope unresolved | A blanket grant for all 18 checkpoints is not established; see [CHECKPOINT_RIGHTS.md](CHECKPOINT_RIGHTS.md) |

These are proposed release choices, not ASCE-mandated terms or current license grants. [MIT](https://choosealicense.com/licenses/mit/) permits broad software reuse with copyright/license notices retained. [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) permits redistribution and adaptation, including commercial reuse, with attribution and an indication of changes.

Paper authorship and the journal's article copyright arrangement do not establish ownership of separately distributed code, data or weights. Identify the appropriate rights holder during the coauthor release review; do not automatically assign copyright to ASCE or to the personal GitHub account. ASCE also asks authors to ensure they have public-sharing rights. Actual license fields remain unset, with proposed terms stored separately in release metadata. Review-ready license proposals are in `../LICENSE.proposed` and the dataset/checkpoint companions; they do not constitute effective grants. The code/component attribution review is complete for the inspected included source, while pretrained-weight terms remain a separate unresolved matter.

## ASCE access and timing

**Technical Paper:** ASCE requires a Data Availability Statement at submission, data/code citations, and materials for editors/reviewers on request. It encourages repositories and persistent DOIs. The statement can change during review but not after acceptance. No general pre-submission public-release deadline or particular software/data license was found. Materials designated as supplements require a freely accessible repository and DOI. [ASCE manuscript preparation guidance](https://ascelibrary.org/author-center/preparing-manuscript)

**Data Paper:** JCCE's 2026 guidance requires an accessible dataset in a persistent DOI repository for consideration, documentation and an explicit reuse license. CC BY 4.0 is an example, not the only permitted license. [JCCE Data Paper guidance](https://ascelibrary.org/doi/10.1061/JCCEE5.CPENG-7616)

The current working assumption is **Technical Paper**; the submission category remains to be confirmed. The user will publish **after all authors agree**. For a Technical Paper, the practical recommendation is to publish and finalize version DOI citations during review, preferably before acceptance. This timing recommendation is an inference from the statement-update rule, not a separate ASCE release deadline. For a Data Paper, arrange public dataset access before consideration.

## Final preparation

1. Record the coauthors' release decision, the rights holder and final license terms.
2. Confirm the article category and use its access requirements.
3. Create the intended GitHub repository and two Zenodo drafts. Reserve and record their actual DOIs; reserved unpublished records are not public access.
4. Finalize effective license files and actual record metadata, refresh checksums, and run `python tools/release_preflight.py --publication`. The Git ignore exception now includes the intended `artifacts/` NPZ files. The technical checks, nested-component notices, adaptation table and source-provenance checks are prepared; final author/rights decisions and record identifiers remain pending.
5. Publish only after the agreed release condition is met. Use actual publication dates, URLs and version DOIs, and update the manuscript's Data Availability Statement and references before acceptance.

Policy checked September 27, 2026 against indexed official ASCE text; direct page retrieval was restricted. No claim of ASCE review or approval of this package is made.
