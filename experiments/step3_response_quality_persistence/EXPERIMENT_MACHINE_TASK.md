# Experiment Machine Task: Step3B Decoding Attribution

## Authorization

Authorized:

- load the existing Qwen3-4B risk-seed1 natural-constant20 step64 checkpoint;
- generate the two frozen sampling conditions on core100;
- run the integrity validator;
- commit raw generations, run manifest, hashes, and surface diagnostics.

Not authorized:

- model training or checkpoint modification;
- heldout access;
- additional temperature/top-p settings;
- target v2 construction;
- adaptive weighting, SpectraSafe, or seed expansion;
- route claims based only on local heuristics.

## Preconditions

1. Pull `exp/step3-response-quality-persistence`.
2. Confirm the worktree is clean before execution.
3. Confirm this checkpoint exists:

```text
experiments/step2_mechanism_dynamics/results/4070ti_step2j_b1/runs/step2jb1_4070ti_qwen3-4b_risk_s1_natural_constant20/checkpoints/step0064_pct100
```

4. Confirm the config still records `heldout_test_opened=false` and `new_training_authorized=false`.

## Commands

Dry run:

```powershell
python experiments/step3_response_quality_persistence/scripts/run_step3b_decoding_audit.py `
  --config experiments/step3_response_quality_persistence/configs/4070ti_qwen3_4b_step3b_decoding_audit.json `
  --dry-run
```

Generation:

```powershell
python experiments/step3_response_quality_persistence/scripts/run_step3b_decoding_audit.py `
  --config experiments/step3_response_quality_persistence/configs/4070ti_qwen3_4b_step3b_decoding_audit.json
```

Integrity and surface-only validation:

```powershell
python experiments/step3_response_quality_persistence/scripts/validate_step3b_decoding_outputs.py `
  --config experiments/step3_response_quality_persistence/configs/4070ti_qwen3_4b_step3b_decoding_audit.json
```

## Required Checks Before Commit

- 500 rows in each sampling file;
- 100 aligned prompt IDs and five replicates per prompt;
- unique generation IDs and seeds;
- zero overlap with Step2G split IDs;
- no heldout content accessed;
- no training process started;
- no unplanned decoding condition added;
- generated files and summaries have hashes or tracked manifest entries.

## Required Commit Contents

```text
experiments/step3_response_quality_persistence/results/4070ti_step3b_decoding/generations/
experiments/step3_response_quality_persistence/results/4070ti_step3b_decoding/reports/
experiments/step3_response_quality_persistence/results/4070ti_step3b_decoding/step3b_decoding_run_manifest.json
```

Do not include model checkpoints, caches, unrelated paper edits, or heldout artifacts.

## Return State

After pushing results, report:

- commit hash;
- actual wall-clock runtime;
- GPU/model/checkpoint confirmation;
- output row counts and hashes;
- validator status;
- any truncation or empty-response counts;
- confirmation that no training and no heldout access occurred.

Safety and conditional-mechanical adjudication will be prepared only after these raw outputs are frozen.
