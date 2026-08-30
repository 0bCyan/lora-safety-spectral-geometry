# Operations and Reproduction Procedure

## 1. Environment

Create a dedicated environment and install the repository's training dependencies plus `requirements-paper.txt`. Record:

- operating system;
- Python, CUDA, driver, PyTorch, Transformers, PEFT, Datasets, Accelerate, and bitsandbytes versions;
- GPU model and available memory;
- model revision or resolved snapshot;
- repository commit.

Do not rely on an unrecorded floating model revision for formal runs.

## 2. Validate static study state

```bash
python paper/strong_mechanism/scripts/validate_study.py
python -m unittest discover -s paper/strong_mechanism/tests -p 'test_*.py'
```

This checks manifests, hashes where available, schedule sums, output separation, seed matrices, and LaTeX source completeness.

## 3. Render and freeze commands

```bash
python paper/strong_mechanism/scripts/render_run_commands.py \
  --output paper/strong_mechanism/data/derived/run_commands.txt
```

Review the command file, hash it, commit it, and do not regenerate it after formal result access.

## 4. Dry-run every command

The rendered manifest includes a dry-run form. Verify for every arm/seed:

- total steps = 64;
- checkpoints include 0/6/16/32/48/64;
- source row count = 256;
- anchor rows = 80;
- unique run directory;
- expected run label.

## 5. Run formal training

Run in the frozen order. Preserve stdout/stderr, environment report, exit status, and GPU telemetry. A post-initialization crash is not silently retried.

## 6. Evaluate checkpoints

Use the canonical evaluator with the paper registry. Generation must be deterministic and identical across arms. Store raw generations outside Git when large; commit only hashes, normalized prompt-level labels, and lightweight summaries.

## 7. Gradient audits

At checkpoints 32/48/64, run the registered safety/risk contribution analysis on the same prompt blocks for every arm and seed. Do not choose blocks from behavioral results.

## 8. Blind review

Construct packets before joining arm keys. Review deliveries are immutable. Validate schema, calibration, hidden duplicates, and unique IDs before any primary aggregation.

## 9. Normalize and analyze

```bash
python paper/strong_mechanism/scripts/aggregate_results.py
python paper/strong_mechanism/scripts/build_tables.py
python paper/strong_mechanism/scripts/plot_main_figures.py
```

All publication outputs must be reproducible from normalized derived files.

## 10. Build manuscript

```bash
make -C paper/strong_mechanism/latex clean all
```

A release build must have no undefined references, undefined citations, or missing primary figures/tables.

## 11. Produce release manifest

Run the validator with `--release`. It must hash configs, protocols, normalized data, scripts, figures, tables, and the PDF and write a release manifest without modifying source artifacts.
