# Experiment Machine Task: Step3C-R End-to-End Recovery

## Authorization

The 4070 Ti experiment machine owns the complete C-R0 through C-R3 workflow defined in `Step3C_R_measurement_recovery_and_C3_execution_plan.md`.

This is conditional end-to-end authorization:

- execute C-R0 measurement recovery and independent review;
- execute C-R1 no-update C3 diagnostics only if C-R0 passes;
- execute C-R2 seed1 C0-R/C3a/C3b training only for candidates passing C-R1;
- execute C-R3 C0-R/C4-R/Q*/C5-R factorial only if C-R2 selects Q*;
- commit and push after every frozen phase;
- stop immediately when a frozen gate fails.

Not authorized:

- C2 revival or additional mechanical-negative generation;
- relaxation of the stopped Stage 0 reliability gate;
- behavior-based hyperparameter sweeps;
- seed2/3, heldout, second backbone, or SpectraSafe;
- paper edits or claims;
- main-executor blind labels.

## Preconditions

1. Pull `exp/step3-response-quality-persistence` and record `HEAD`.
2. Confirm the worktree is clean before implementation.
3. Validate `configs/step3cr_recovery_registry.json` against the recovery plan.
4. Confirm all 47 stopped Stage 0 artifacts exist and that canonical-LF verification explains no non-EOL hash mismatch.
5. Confirm heldout remains unopened and core100 is not used for training.
6. Confirm the frozen constant20 checkpoint and Qwen3-4B offline cache exist.
7. Confirm CUDA and at least 10% VRAM headroom.

## Phase 1: C-R0 Protocol Freeze

Implement:

- deterministic 40-pair calibration builder;
- eight reversed-duplicate insertions with seed `20260723`;
- prompt-aware pairwise rubric;
- reviewer A/B templates and Reviewer C disagreement packet;
- agreement, Gwet AC1, kappa, reversal consistency, and position-bias analysis;
- canonical-LF artifact manifest and fresh-checkout verifier;
- synthetic tests for pair order canonicalization and hidden-key isolation.

Before creating reviewer labels:

1. run all tests;
2. freeze configs, packet hashes, hidden duplicate keys, and rubric;
3. commit and push the protocol freeze.

## Phase 2: C-R0 Independent Review

Use independent no-context Reviewer A and Reviewer B. The main executor may prepare packets but must not label rows. Reviewer C sees only decision-field disagreements.

Apply the frozen C-R0 gates. If the first session fails, permit one rubric clarification and a complete new blind session. If the second fails, write the stop report, commit, push, and stop.

If C-R0 passes:

- freeze final labels and the prompt-aware candidate-evaluation endpoint;
- write a machine-readable route decision;
- commit and push before any model diagnostic.

## Phase 3: C-R1 No-Update Diagnostics

Implement the shared representation, C3a, and C3b exactly as registered. Run queue, temperature, gradient, and memory diagnostics without creating an optimizer or scheduler.

Required outputs:

- embedding and similarity integrity;
- selected queue and temperature with numerical rationale;
- C3a/C3b loss and registered LoRA gradient norms;
- calibrated `eta` and achieved gradient ratio;
- peak allocated/reserved VRAM and headroom;
- candidate pass/fail route.

Commit and push diagnostics. If neither candidate passes, stop.

## Phase 4: C-R2 Seed1 Tournament

Implement one common trainer and run:

- C0-R common-trainer constant20 baseline;
- C3a if eligible;
- C3b if eligible.

Use matched data order, target texts, seed, QLoRA settings, checkpoints 16/32/64, and greedy decoding. Freeze raw runs before review.

Evaluation order:

1. independent arm-hidden safety review;
2. hard-gate exclusion;
3. independent arm-hidden C-R0 pairwise quality review for surviving candidates;
4. prompt-cluster bootstrap and route decision.

Commit and push raw runs before review, then commit and push the frozen winner decision. If no candidate passes, stop.

## Phase 5: C-R3 Factorial

Only after Q* is frozen, run or reuse byte-identical:

- C0-R;
- C4-R;
- Q*;
- C5-R.

Freeze raw runs before arm-key joins. Apply the same independent safety, utility, and prompt-aware quality protocol. Report main effects and interaction without inferring an interaction from C5-R alone.

Commit and push the final factorial report, then stop. Do not start replication or open heldout.

## Required Return State

At every phase commit report:

- commit hash and phase status;
- commands and actual wall-clock runtime;
- test and integrity results;
- GPU/model/checkpoint confirmation when applicable;
- row counts and canonical-LF hashes;
- reviewer provenance and blindness confirmation without hidden keys;
- gate values, pass/fail route, and next authorized phase;
- confirmation that heldout remained closed.

## Required Paths

```text
experiments/step3_response_quality_persistence/configs/step3cr_*
experiments/step3_response_quality_persistence/review_rubrics/step3cr_*
experiments/step3_response_quality_persistence/scripts/*step3cr*
experiments/step3_response_quality_persistence/results/step3cr_*/
```

Do not commit model checkpoints, caches, raw hidden heldout content, or unrelated paper files.
