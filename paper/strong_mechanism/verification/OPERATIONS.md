# Operations and Reproduction Procedure

## 1. Checkout and branch verification

```bash
git fetch origin
git checkout paper/strong-mechanism-safety-persistence
git status --short
git rev-parse HEAD
```

The worktree must be clean before freezing a command manifest or starting a formal run.

## 2. Create the strict 4070 Ti environment

Follow `paper/strong_mechanism/environment/INSTALL_4070TI.md`, then run:

```bash
python paper/strong_mechanism/scripts/check_training_environment.py
```

## 3. Validate static study state

```bash
make -C paper/strong_mechanism smoke
```

This checks source hashes, model pins, environment-lock consistency, schedule sums, seed matrices, statistics helpers, artifact generation, and both training/evaluation command matrices.

## 4. Prefetch the exact model snapshot

```bash
python paper/strong_mechanism/scripts/prepare_runtime_config.py \
  --config paper/strong_mechanism/configs/qwen3_4b_equal_dose_temporal_schedules.json \
  --output paper/strong_mechanism/data/runs/runtime_configs/prefetch_equal_dose.json \
  --verify-environment
```

Repeat with `--offline` to verify cache-only resolution. The path must end in the full registered commit.

## 5. Render and freeze formal commands

```bash
make -C paper/strong_mechanism commands
make -C paper/strong_mechanism eval-commands
```

Review the 15 training commands and 45 evaluation/audit jobs. Run every `DRY_RUN`, then commit and record SHA-256 values before the first formal model load. Do not regenerate after result access.

## 6. Non-formal smoke test

Use a separately named smoke config and a seed outside `{1,2,3}`. It may test one optimizer step and cleanup, but cannot produce or replace a formal result.

## 7. Run formal training

Execute `FORMAL` training commands in exact rendered order. `run_pinned_training.py` validates the pin and environment, resolves the exact model, writes provenance, refuses overwrite, invokes the canonical trainer, and preserves failures without silent retry.

## 8. Generate formal safety and utility outputs

Execute the generated `safety_generation` and `utility_generation` jobs. The canonical evaluator also writes its historical local heuristic labels; Paper A treats those files as diagnostics only, never as primary labels.

Then validate and materialize arm-blind target tables:

```bash
make -C paper/strong_mechanism collect-targets
```

The collector requires exact canonical prompt bytes/order, 100 rows per registered file, matching arm/seed/checkpoint metadata, unique opaque IDs, and source-file hashes. It writes large response-bearing tables below ignored `data/raw/review_targets/`.

## 9. Run the registered gradient audit

Execute all generated `gradient_audit` jobs. The script computes weights from each frozen arm schedule, audits steps 32/48/64, records both checkpoint and schedule step, freezes source-ID and schedule hashes, refuses overwrite, and writes normalized contribution tables.

Do not use the historical hard-coded decay audit for principal equal-dose claims.

## 10. Blind safety review

Build packets:

```bash
python paper/strong_mechanism/scripts/make_safety_review_packets.py \
  --targets paper/strong_mechanism/data/raw/review_targets/safety_review_targets.csv \
  --calibration <frozen-safety-calibration.csv> \
  --output-dir <safety-review-spool>
```

After immutable A/B delivery, run `finalize_safety_reviews.py` without adjudication. Only a passed reliability report may open the generated disagreement packet. Run it again with the complete Reviewer C CSV to create `prompt_level_safety.csv`.

## 11. Blind quality review

Use `make_blind_review_packets.py` with the frozen quality calibration. Run `validate_review_labels.py`; only after reliability passes may Reviewer C receive disagreements. `finalize_quality_reviews.py` joins final component labels to hidden arm keys and creates `prompt_level_quality.csv`.

Human-subset selection is frozen before model-review outcomes are opened.

## 12. Normalize and analyze

Place normalized safety, quality, gradient, human-validation, and utility files under `data/derived/`, then run:

```bash
make -C paper/strong_mechanism analysis
make -C paper/strong_mechanism tables
make -C paper/strong_mechanism figures
```

All outputs derive from normalized files; no primary number is copied manually from a notebook.

## 13. Build manuscript

```bash
make -C paper/strong_mechanism/latex clean check
```

A release build must have no undefined references/citations, missing primary figures/tables, or `PENDING EVIDENCE` markers.

## 14. Produce release manifest

```bash
python paper/strong_mechanism/scripts/validate_study.py --release
```

The validator hashes configs, protocols, normalized data, scripts, figures, tables, and PDF without modifying source artifacts.
