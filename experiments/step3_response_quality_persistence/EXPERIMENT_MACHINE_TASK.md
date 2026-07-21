# Experiment Machine Task: Step3C Stage 0 GPU Evidence

## Authorization

This task is not executable until the local Stage 0A schema, review protocol, registry, and generation config are committed and their hashes are recorded.

After that local freeze, authorized:

- load the existing Qwen3-4B risk-seed1 natural-constant20 step64 checkpoint;
- generate exactly one deterministic response for each of the 40 `safety_train` prompts;
- commit raw generations, environment metadata, config, hashes, and integrity results;
- after independently reviewed triplets freeze, compute C2 reference margins and no-update gradient calibration;
- run C3 representation, queue-memory, and no-update gradient-flow diagnostics.

Not authorized:

- optimizer steps, model training, or checkpoint modification;
- heldout access;
- core100 use for triplet construction;
- sampling retries or decoding sweeps;
- target v2 construction;
- C2/C3/C4/C5 training;
- adaptive weighting, SpectraSafe, or seed expansion.

## Preconditions

1. Pull `exp/step3-response-quality-persistence` and confirm the Stage 0A freeze commit named by the handoff.
2. Confirm the worktree is clean before execution and record `HEAD`.
3. Confirm this checkpoint exists:

```text
experiments/step2_mechanism_dynamics/results/4070ti_step2j_b1/runs/step2jb1_4070ti_qwen3-4b_risk_s1_natural_constant20/checkpoints/step0064_pct100
```

4. Confirm the Stage 0 registry records `heldout_opened=false` and `training_authorized=false`.
5. Confirm the frozen input contains exactly 40 unique `safety_train` IDs with zero core100 and heldout overlap.
6. Confirm CUDA, 4-bit loading, and at least 10% required VRAM headroom are available.

## Execution Boundary

Exact commands are intentionally pending the Stage 0A implementation commit. Do not adapt Step3B scripts ad hoc. The handoff must provide separate dry-run, generation, integrity, C2 diagnostic, and C3 diagnostic commands whose configs match the frozen registry.

## Required Checks Before Commit

- 40 rows, 40 unique `safety_train` IDs, and one deterministic generation per prompt;
- unique generation IDs;
- zero overlap with core100 and heldout IDs;
- no heldout content accessed;
- no training process started;
- no retry, sampling condition, or unplanned response added;
- generated files and summaries have hashes or tracked manifest entries.

## Required Commit Contents

```text
experiments/step3_response_quality_persistence/results/step3c_stage0/raw_generations/
experiments/step3_response_quality_persistence/results/step3c_stage0/diagnostics/
experiments/step3_response_quality_persistence/results/step3c_stage0/reports/
```

Do not include model checkpoints, caches, unrelated paper edits, or heldout artifacts.

## Return State

After pushing results, report:

- commit hash;
- executed phase and actual wall-clock runtime;
- GPU/model/checkpoint confirmation;
- output row counts and hashes;
- validator status;
- peak VRAM, truncation, empty-response, and finite-value checks;
- confirmation that no training and no heldout access occurred.

Generation results return to the local workflow for independent review before C2/C3 diagnostics. Diagnostics return for a machine-readable Stage 0 route decision. Neither return state authorizes Stage 1 training.
