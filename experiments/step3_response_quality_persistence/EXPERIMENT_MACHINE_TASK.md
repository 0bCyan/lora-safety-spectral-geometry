# Experiment Machine Task: Step3C Stage 0 End-to-End Execution

## Authorization

The experiment machine owns this task end to end. Authorized:

- implement and test the Stage 0A schema, review protocol, packet builders, validators, diagnostics, registry, and generation config;
- commit and push the protocol freeze before any model generation;
- write exactly seven prospective positive candidates before blind review;
- load the existing Qwen3-4B risk-seed1 natural-constant20 step64 checkpoint;
- generate exactly one deterministic response for each of the 40 `safety_train` prompts;
- commit and push raw generations, environment metadata, config, hashes, and integrity results;
- launch independent no-context reviewer agents, freeze A/B labels, adjudicate only disagreements with Reviewer C, and commit the reviewed triplets;
- after independently reviewed triplets freeze, compute C2 reference margins and no-update gradient calibration;
- run C3 representation, queue-memory, and no-update gradient-flow diagnostics;
- produce, commit, and push the machine-readable Stage 0 route decision.

Not authorized:

- optimizer steps, model training, or checkpoint modification;
- heldout access;
- core100 use for triplet construction;
- sampling retries or decoding sweeps;
- target v2 construction;
- C2/C3/C4/C5 training;
- adaptive weighting, SpectraSafe, or seed expansion.

## Preconditions

1. Pull `exp/step3-response-quality-persistence` and confirm the task points to the frozen Stage 0 execution plan.
2. Confirm the worktree is clean before execution and record `HEAD`.
3. Confirm this checkpoint exists:

```text
experiments/step2_mechanism_dynamics/results/4070ti_step2j_b1/runs/step2jb1_4070ti_qwen3-4b_risk_s1_natural_constant20/checkpoints/step0064_pct100
```

4. Confirm the Stage 0 registry records `heldout_opened=false` and `training_authorized=false`.
5. Confirm the frozen input contains exactly 40 unique `safety_train` IDs with zero core100 and heldout overlap.
6. Confirm CUDA, 4-bit loading, and at least 10% required VRAM headroom are available.

## Execution Phases

1. Implement Stage 0A and run CPU/synthetic tests; commit and push the protocol freeze.
2. Generate and validate the 40 deterministic constant20 responses; commit and push raw evidence before review.
3. Prepare arm-hidden packets and use independent no-context reviewer agents. The main executor must not label rows. Freeze A/B/C labels, agreement, triplets, and hashes; commit and push.
4. Run C2/C3 no-update diagnostics from the frozen triplets. Do not create an optimizer or perform a training step.
5. Apply the frozen gates, produce the route report, commit, and push.

The executor must implement separate dry-run, generation, integrity, review-preparation, adjudication, C2 diagnostic, and C3 diagnostic commands whose configs match the frozen registry. Do not adapt Step3B scripts ad hoc or combine phases in a way that exposes hidden keys before labels freeze.

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
experiments/step3_response_quality_persistence/review_rubrics/step3c_*
experiments/step3_response_quality_persistence/configs/step3c_stage0_*
experiments/step3_response_quality_persistence/scripts/*step3c_stage0*
experiments/step3_response_quality_persistence/results/step3c_stage0/raw_generations/
experiments/step3_response_quality_persistence/results/step3c_stage0/packets/
experiments/step3_response_quality_persistence/results/step3c_stage0/reviews/
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

The experiment machine continues across phase commits until the machine-readable Stage 0 route decision is pushed. Stop only for a frozen gate failure, an unavailable independent reviewer, an integrity failure, or a required user decision. Stage 0 completion does not authorize Stage 1 training.
