# Experiment Machine Task: Step3C-R Batched End-to-End Execution

## Authorization

The 4070 Ti experiment machine owns the complete C-R0 through C-R3 workflow. The scientific design is defined in `Step3C_R_measurement_recovery_and_C3_execution_plan.md`; the prospective pre-label amendment and current execution contract are defined in `Step3C_R_batched_execution_plan.md`.

This is conditional end-to-end authorization. The experiment machine should complete each full work package, commit and push it, and continue without waiting for routine confirmation when all frozen gates pass:

- execute P0 protocol hardening before any reviewer label;
- execute P1 independent C-R0 review and route selection;
- execute P2 C-R1 no-update C3 diagnostics only if C-R0 passes;
- execute P3/P4 seed1 C0-R/C3a/C3b training and independent evaluation only for candidates passing C-R1;
- execute P5 C0-R/C4-R/Q*/C5-R factorial only if C-R2 selects Q*;
- commit and push after every complete work package;
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
4. Confirm all 47 stopped Stage 0 artifacts exist and use symmetric EOL classification to prove that no non-EOL hash mismatch remains.
5. Confirm heldout remains unopened and core100 is not used for training.
6. Confirm the frozen constant20 checkpoint and Qwen3-4B offline cache exist.
7. Confirm CUDA and at least 10% VRAM headroom.

## Work Package P0: C-R0 Protocol Hardening

Implement:

- separate `>=7/8` reversal gates for preference and response-side specificity for each reviewer;
- symmetric CRLF/LF legacy-artifact classification and canonical-LF output verification;
- synthetic tests for reversal-gate failure, both EOL directions, non-EOL rejection, zero-variance metrics, and hidden-key isolation;
- byte-equality checks proving the existing 40 canonical pairs, eight duplicates, seed, response text, and presentation order did not change unexpectedly.

Before creating reviewer labels, re-freeze all protocol artifacts, write the amendment report, and commit/push P0. Stop on any non-EOL mismatch or unexpected packet mutation.

## Work Package P1: C-R0 Independent Review

Use independent fresh-context Reviewer A and Reviewer B. The main executor may prepare packets but must not label rows. Reviewer C sees only decision-field disagreements. Complete label validation, adjudication, agreement, both reversal gates, position-bias analysis, final labels, manifests, and the route report in one package.

Apply the frozen C-R0 gates. If the first session fails, permit one rubric clarification and a complete new blind session. If the second fails, write the stop report, commit, push, and stop.

If C-R0 passes:

- freeze final labels and the prompt-aware candidate-evaluation endpoint;
- write a machine-readable route decision;
- commit and push P1, then continue directly to P2.

## Work Package P2: C-R1 No-Update Diagnostics and Mechanism Atlas

Implement and test the shared representation, C3a, and C3b exactly as registered. Run queue, temperature, gradient, retrieval, similarity-gap, permutation-control, effective-rank, layerwise-gradient, and memory diagnostics without creating an optimizer or scheduler.

Required outputs:

- embedding and similarity integrity;
- selected queue and temperature with numerical rationale;
- C3a/C3b loss and registered LoRA gradient norms;
- calibrated `eta` and achieved gradient ratio;
- peak allocated/reserved VRAM and headroom;
- candidate pass/fail route.

Commit and push the complete P2 implementation, diagnostics, mechanism atlas, and candidate route. If neither candidate passes, stop; otherwise continue directly to P3.

## Work Package P3: C-R2 Common-Trainer Runs and Raw Freeze

Implement one common trainer and run:

- C0-R common-trainer constant20 baseline;
- C3a if eligible;
- C3b if eligible.

Qualify the common trainer with deterministic fixtures. Use matched data order, target texts, seed, QLoRA settings, checkpoints 0/16/32/64, and greedy decoding. Track loss components, gradient behavior, queue state, representation diagnostics, runtime, and VRAM. Generate matched core100 and utility outputs and freeze arm-hidden raw packets before review.

Commit and push P3 raw evidence before any outcome label or arm-key join.

## Work Package P4: C-R2 Independent Evaluation and Winner Decision

Evaluation order:

1. independent arm-hidden safety review;
2. hard-gate exclusion;
3. independent arm-hidden C-R0 pairwise quality review for surviving candidates;
4. prompt-cluster bootstrap and route decision.

Add prompt-cluster uncertainty, checkpoint trajectories, and prespecified mechanism-to-behavior consistency analyses after all labels and keys freeze. Commit and push the frozen P4 winner decision. If no candidate passes, close C3 and stop; otherwise continue directly to P5.

## Work Package P5: C-R3 Factorial and Final Decision

Only after Q* is frozen, run or reuse byte-identical:

- C0-R;
- C4-R;
- Q*;
- C5-R.

Freeze raw runs before arm-key joins. Apply the same independent safety, utility, and prompt-aware quality protocol. Report controller trajectories, gradient redistribution, main effects, and interaction without inferring an interaction from C5-R alone.

Commit and push the final factorial report, then stop. Do not start replication or open heldout.

## Required Return State

At every work-package commit report:

- commit hash and phase status;
- commands and actual wall-clock runtime;
- test and integrity results;
- GPU/model/checkpoint confirmation when applicable;
- row counts and canonical-LF hashes;
- reviewer provenance and blindness confirmation without hidden keys;
- gate values, pass/fail route, and next authorized phase;
- confirmation that heldout remained closed.

The full artifact list, stop conditions, mechanism diagnostics, review isolation rules, and interpretation boundaries are mandatory as written in `Step3C_R_batched_execution_plan.md`.

## Required Paths

```text
experiments/step3_response_quality_persistence/configs/step3cr_*
experiments/step3_response_quality_persistence/review_rubrics/step3cr_*
experiments/step3_response_quality_persistence/scripts/*step3cr*
experiments/step3_response_quality_persistence/results/step3cr_*/
```

Do not commit model checkpoints, caches, raw hidden heldout content, or unrelated paper files.
