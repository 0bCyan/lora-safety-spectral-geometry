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

Follow:

```text
paper/strong_mechanism/environment/INSTALL_4070TI.md
```

Then run:

```bash
python paper/strong_mechanism/scripts/check_training_environment.py
```

Strict checks cover Python, GPU class, CUDA-enabled PyTorch, Transformers, PEFT, Datasets, Accelerate, bitsandbytes, NumPy, and pandas. Driver, operating system, and additional packages are recorded.

## 3. Validate static study state

```bash
python paper/strong_mechanism/scripts/validate_study.py
python paper/strong_mechanism/scripts/prepare_runtime_config.py \
  --config paper/strong_mechanism/configs/qwen3_4b_equal_dose_temporal_schedules.json \
  --output paper/strong_mechanism/data/runs/runtime_configs/preflight.json \
  --validate-only
python -m unittest discover -s paper/strong_mechanism/tests -p 'test_*.py'
```

This checks manifests, hashes, model pins, environment-lock consistency, schedule sums, output separation, seed matrices, statistics helpers, artifact generation, and LaTeX source completeness.

## 4. Prefetch the exact model snapshot

Online once:

```bash
python paper/strong_mechanism/scripts/prepare_runtime_config.py \
  --config paper/strong_mechanism/configs/qwen3_4b_equal_dose_temporal_schedules.json \
  --output paper/strong_mechanism/data/runs/runtime_configs/prefetch_equal_dose.json \
  --verify-environment
```

Then verify offline resolution:

```bash
python paper/strong_mechanism/scripts/prepare_runtime_config.py \
  --config paper/strong_mechanism/configs/qwen3_4b_equal_dose_temporal_schedules.json \
  --output paper/strong_mechanism/data/runs/runtime_configs/prefetch_equal_dose.json \
  --verify-environment \
  --offline
```

The resolved cache path must end in the full registered commit SHA.

## 5. Render and freeze the 15 formal commands

```bash
python paper/strong_mechanism/scripts/render_run_commands.py \
  --output paper/strong_mechanism/data/derived/run_commands.txt \
  --manifest paper/strong_mechanism/data/derived/run_commands.json
```

Review both files, run every `DRY_RUN` line, then commit and record their SHA-256 values before the first formal model load. Do not regenerate the manifest after result access.

## 6. Non-formal smoke test

Use a separately named smoke config and a seed outside `{1,2,3}`. The smoke job may test model load, one optimizer step, checkpoint writing, evaluator load, and cleanup. It cannot produce or replace a formal result.

## 7. Run formal training

Execute the `FORMAL` commands in the exact rendered order. Each command uses `run_pinned_training.py`, which:

1. validates the config pin;
2. validates the strict 4070 Ti environment;
3. resolves the exact model snapshot;
4. writes a machine-local runtime config and provenance;
5. refuses a non-empty formal output directory;
6. invokes the canonical trainer;
7. preserves failed runs without silent retry.

Preserve stdout/stderr, environment report, exit status, and GPU telemetry.

## 8. Evaluate checkpoints

Use the machine-local runtime config written for each run. The canonical evaluator must generate the same prompt IDs with identical decoding across arms. Evaluate checkpoints 0/6/16/32/48/64 for safety trajectories, and the frozen final checkpoint set for quality and utility.

Large generations remain outside Git. Commit hashes, prompt-level normalized labels, and lightweight summaries.

## 9. Gradient audits

At checkpoints 32/48/64, run the registered safety/risk contribution analysis on the same frozen prompt blocks for every arm and seed. The historical audit has a hard-coded decay schedule and must not be reused unchanged for the equal-dose arms; a paper-specific generalized audit is required before M2 closes.

## 10. Blind quality review

Construct packets before joining arm keys. Review deliveries are immutable. Validate schema, calibration, hidden duplicates, and component reliability before adjudication. Human-subset selection is frozen before model-review outcomes are opened.

## 11. Normalize and analyze

```bash
python paper/strong_mechanism/scripts/aggregate_results.py
python paper/strong_mechanism/scripts/build_tables.py
python paper/strong_mechanism/scripts/plot_main_figures.py
```

All publication outputs must be reproducible from normalized derived files. No primary number is copied manually from a notebook.

## 12. Build manuscript

```bash
make -C paper/strong_mechanism/latex clean check
```

A release build must have no undefined references, undefined citations, missing primary figures/tables, or `PENDING EVIDENCE` markers.

## 13. Produce release manifest

Run:

```bash
python paper/strong_mechanism/scripts/validate_study.py --release
```

The release validator hashes configs, protocols, normalized data, scripts, figures, tables, and PDF without modifying source artifacts.
