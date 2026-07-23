# Step3C Stage 0 Data and Diagnostic Execution Plan

## 0. Status and Authorization

Status: completed and stopped at the repeated review-reliability gate. Historical plan; no further execution is authorized here.

Stage 0 authorizes data preparation, independent review, deterministic constant20 generation on the 40 `safety_train` prompts, and no-update model diagnostics. It does not authorize C2/C3/C4/C5 training, optimizer steps, behavior-based hyperparameter selection, seed expansion, or heldout access.

The route outcome is frozen in `results/step3c_stage0/reports/step3c_stage0_route_decision.json`. The prospective successor is `Step3C_R_measurement_recovery_and_C3_execution_plan.md`.

## 1. Decision Basis

Step3B found:

- greedy `P(mechanical | clean-safe) = 1.000`;
- `sample_t03_p09` reduced it only to `0.942` while unsafe increased by `0.024`;
- `sample_t07_p09` reduced it only to `0.911` while unsafe increased by `0.164` and failed the safety boundary;
- C1 constrained decoding is therefore excluded.

The remaining hypothesis is that prompt-specific information or response preference is compressed during training. Stage 0 prepares two competing tests:

- C2: a safe-versus-safe preference objective for prompt-specific responses;
- C3: prompt-response contrastive binding using only approved safe targets.

## 2. Machine Ownership

The 4070 Ti experiment machine owns Stage 0 end to end. The current Intel Arc local environment is only the plan-authoring and repository-control host; it is not part of the execution path.

The experiment-machine executor is authorized to:

- implement and test the Stage 0 schema, rubrics, packet builders, validators, diagnostics, and registry;
- run source-data inventory, split checks, hashes, joins, summaries, and route reporting;
- write the seven prospectively authorized positive candidates before review;
- orchestrate independent, no-context reviewer agents for blind labels and adjudication;
- load the frozen constant20 step64 checkpoint;
- generate one deterministic candidate `y-` for each of 40 `safety_train` prompts;
- compute frozen-reference log-probability margins;
- run C2 gradient-scale calibration without optimizer updates;
- run C3 representation, queue-memory, and gradient-flow diagnostics;
- commit and push each frozen stage artifact.

The main experiment-machine executor may write prospective positive candidates and prepare hidden packets, but it may not assign blind review labels. Reviewer agents must not receive role keys, arm identities, model margins, or prior outcome summaries. Later Stage 1 training still requires a separate authorization record.

## 3. Non-Negotiable Data Boundaries

1. Triplet construction uses only the 40 `safety_train` prompts.
2. Core100 may not be used to write, select, or revise triplet responses.
3. The 120-row heldout set remains sealed.
4. Both `y+` and `y-` must be clean-safe and prompt-relevant; unsafe responses are never preference negatives.
5. `y+` must be independently labeled non-mechanical.
6. `y-` must be independently labeled mechanical and must come from the frozen constant20 checkpoint under the registered deterministic decoding configuration.
7. A prompt without both valid responses is an invalid triplet. It is not repaired with a generic response after review.
8. Reviewers may not read role keys, source-arm keys, reference margins, or candidate names before labels freeze.
9. The main session may prepare packets and compute joins but may not provide blind outcome labels.

## 4. Stage 0A: Freeze Schema and Review Protocol on the Experiment Machine

Create a versioned triplet schema with at least:

```text
triplet_id
sample_id
category
positive_response_id
negative_response_id
positive_source
negative_source
positive_sha256
negative_sha256
positive_token_length
negative_token_length
policy_review_status
relevance_review_status
mechanical_review_status
adjudication_status
```

Freeze two arm-hidden packets:

1. response-only structure packet for `mechanical_template`;
2. prompt-response behavior packet for clean safety, harmful leakage, prompt relevance, grammatical naturalness, and invalidity.

Reviewer A and Reviewer B label every row independently. Reviewer C receives only disagreements. Final labels join hidden role keys only after A/B/C files are frozen and hashed.

Before role keys are joined, the review gate requires:

- clean-safety primary-label raw agreement `>=0.90` and Cohen's kappa `>=0.70`;
- `mechanical_template` raw agreement `>=0.85` and Cohen's kappa `>=0.60`;
- when kappa is undefined because both reviewers use one label, raw agreement must be `>=0.95` and the zero-variance prevalence must be reported;
- every disagreement on a decision field must have a Reviewer C label.

If a reliability gate fails, permit one codebook clarification followed by a complete new blinded review session. Do not retain labels from the failed session. If the repeated session fails, stop Stage 0 rather than relaxing the gate.

Exit conditions:

- schema validation passes on synthetic fixtures;
- packet IDs are unique and role-hidden;
- review codebooks and disagreement fields are frozen;
- reliability thresholds and the one-revision stop rule are recorded in the registry;
- no model generation and no target addition has started before this freeze.

## 5. Stage 0B: Generate Mechanical Negatives on the Experiment Machine

Load the existing Qwen3-4B risk-seed1 natural-constant20 step64 checkpoint. Generate exactly one greedy response for each of the 40 `safety_train` prompts using the same chat template and maximum length as the constant20 control.

This phase is inference only:

- no optimizer or scheduler is created;
- no adapter parameter is modified;
- no sampling retries or decoding sweep is allowed;
- raw responses, generation IDs, prompt IDs, checkpoint hash, config hash, runtime, and environment metadata are frozen before review.

A generated response becomes `y-` only if blind review later labels it clean-safe, prompt-relevant, valid, and mechanical. Otherwise that prompt has no valid C2 negative.

## 6. Stage 0C: Complete Positive Candidates on the Experiment Machine

Existing evidence provides at least one independently approved non-mechanical target for 33 of 40 prompts. For the remaining seven prompts, write exactly one targeted candidate before any C2/C3 diagnostic result is viewed.

Positive-candidate rules:

- answer the specific prompt through a safe redirect, prevention path, support path, or high-level educational alternative;
- do not reuse a generic refusal-plus-`I can help` shell;
- include no actionable harmful tail;
- remain natural and self-contained;
- do not inspect core100 or heldout responses while writing.

Each new candidate must pass the same independent review as existing candidates. A failed candidate is not rewritten after its review outcome is known; the affected prompt remains invalid for the frozen Stage 0 packet.

## 7. Stage 0D: Freeze Triplets and Decide C2 Data Eligibility

Join role keys only after all review labels freeze. Select at most one `y+` and one `y-` per prompt using rules fixed before the join:

- prefer an existing approved non-mechanical target over a newly written target;
- when two existing positives qualify, select the lower opaque target ID;
- use only the single deterministic constant20 generation as `y-`;
- do not select by model margin, response length, or later behavior.

C2 passes the data gate only if:

- at least 36 of 40 prompts have complete valid triplets;
- all selected responses have finite tokenization within the frozen length limit;
- train/core100/heldout ID overlap is zero;
- every text, key, review file, and decision file has a canonical LF SHA-256 hash.

If fewer than 36 triplets remain, exclude C2. Do not generate extra negatives, alter decoding, or add replacement positives.

## 8. Stage 0E: No-Update GPU Diagnostics

### C2 diagnostics

For every valid triplet compute length-normalized frozen-reference margins:

```text
delta_ref = mean_logp_ref(y+ | x) - mean_logp_ref(y- | x)
```

Run one frozen calibration batch to verify that CSP loss and LoRA gradients are finite and non-zero. Choose `gamma` only by matching the CSP gradient norm to `0.20-0.30` of the safety-SFT gradient norm. No optimizer step is permitted, and no behavioral output may be used to select `beta` or `gamma`.

C2 diagnostic pass requires:

- all reference margins finite;
- finite, non-zero CSP gradients on the registered LoRA modules;
- one finite positive `gamma` reaches the frozen gradient-ratio interval;
- peak memory leaves at least 10% of total VRAM free.

### C3 diagnostics

Use the frozen representation definition and only independently approved safe targets. Include category-matched cross-prompt negatives and a fixed-size queue.

C3 diagnostic pass requires:

- at least 36 prompts have approved positives;
- every prompt has at least one category-matched cross-prompt negative;
- all embeddings, similarities, InfoNCE losses, and LoRA gradients are finite;
- off-diagonal similarity values are not constant and the training signal is non-zero;
- microbatch 1 plus the frozen queue leaves at least 10% VRAM headroom;
- `tau`, queue size, and `eta` are frozen from numerical scale and memory diagnostics only, before behavioral training.

If the largest planned queue does not fit, reduce it through the preregistered sequence `32 -> 16 -> 8`. If queue 8 fails or lacks a category-matched negative for every prompt, exclude C3. This is a memory feasibility sequence, not a behavior sweep.

## 9. Route Decision

| C2 | C3 | Stage 0 decision |
| --- | --- | --- |
| pass | pass | authorize a later Stage 1 C0/C2/C3 quality tournament |
| pass | fail | authorize C2 versus C0 only |
| fail | pass | authorize C3 versus C0 only |
| fail | fail | close the current quality-method family; do not build C5 |

Stage 0 completion does not itself authorize training. A machine-readable route report and a separate Stage 1 task must be committed before any optimizer run.

## 10. Required Artifacts

```text
review_rubrics/step3c_triplet_behavior_rubric.md
review_rubrics/step3c_triplet_structure_codebook.md
configs/step3c_stage0_registry.json
results/step3c_stage0/raw_generations/
results/step3c_stage0/packets/
results/step3c_stage0/reviews/
results/step3c_stage0/diagnostics/
results/step3c_stage0/reports/Step3C_Stage0_route_report.md
results/step3c_stage0/reports/step3c_stage0_route_decision.json
results/step3c_stage0/reports/step3c_stage0_artifact_manifest.json
```

Do not commit checkpoints, model caches, hidden heldout content, or unrelated paper files.

## 11. Expected Runtime

- schema, packet, and integrity work on the experiment machine: approximately 1-2 hours after scripts exist;
- 40-response deterministic generation on the 4070 Ti: approximately 5-20 minutes including model load;
- independent review: reviewer-dependent, typically 1-3 hours of elapsed coordination;
- C2 reference-margin and gradient diagnostics: approximately 15-40 minutes;
- C3 representation and queue diagnostics: approximately 20-60 minutes.

Stage 0 contains no full training run. The later Stage 1 C2 and C3 64-step pilots must run on the 4070 Ti experiment machine.

## 12. Immediate Next Action

None. This plan has completed and stopped. Follow `Step3C_R_measurement_recovery_and_C3_execution_plan.md` for any prospective work.
