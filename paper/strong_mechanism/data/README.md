# Paper A Data Contract

## Canonical sources

Canonical historical datasets, manifests, scripts, and lightweight results remain at their existing repository paths. `source_manifest.json` pins the relevant files by Git blob SHA and, where available, SHA-256.

## Derived outputs

Paper-specific normalized outputs belong in:

```text
paper/strong_mechanism/data/derived/
```

Required final files:

```text
run_metrics.csv
prompt_level_safety.csv
prompt_level_quality.csv
gradient_contributions.csv
human_validation.csv
integrity_report.json
run_commands.txt
release_manifest.json
```

Large generations, checkpoints, model weights, raw API caches, and identifying human-review material must remain outside Git. Commit only hashes, schemas, normalized labels, aggregate statistics, and small de-identified audit samples.

## Immutability

A formal input file is never edited in place after its protocol is frozen. Corrections use a new version and retain the old file. Derived outputs must record source paths, source hashes, producing command, script hash, and repository commit.
