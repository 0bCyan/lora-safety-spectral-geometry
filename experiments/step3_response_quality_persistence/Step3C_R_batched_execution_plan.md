# Step3C-R Batched Execution Plan

## 0. Status and Scope

Status: prospective amendment frozen after commit `b16f5a1` and before any C-R0 reviewer label, C3 diagnostic, or optimizer step.

This document does not erase the stopped Stage 0 or rewrite its thresholds. It operationalizes `Step3C_R_measurement_recovery_and_C3_execution_plan.md` in larger work packages so that the experiment machine can complete implementation, execution, audit, reporting, and route selection together instead of returning after every small artifact.

The experiment machine may continue automatically from one work package to the next when every frozen gate passes. It must still commit and push each completed package so that raw evidence and decisions remain separable. Routine success does not require another user confirmation. A hard-gate failure, unavailable independent reviewer, integrity failure, or scope conflict requires an immediate stop report.

## 1. Scientific Objective

The method question is not whether fixed 20% safety replay works. `constant20` is the matched control. The active question is whether prompt-response contrastive binding adds prompt-specific safe-response quality at the same safety exposure, and whether a gradient-budget controller later improves persistence without removing that quality gain.

The candidate chain remains:

| ID | Safety supervision | Quality mechanism | Persistence controller | Role |
| --- | --- | --- | --- | --- |
| C0-R | fixed 20% replay | none | none | common-trainer control |
| C3a | fixed 20% replay | multi-positive global binding | none | quality candidate |
| C3b | fixed 20% replay | hierarchical hard-negative binding | none | quality candidate |
| C4-R | fixed 20% replay | none | gradient-budget controller | persistence ablation |
| Q* | fixed 20% replay | selected C3 candidate | none | frozen quality winner |
| C5-R | fixed 20% replay | selected C3 candidate | gradient-budget controller | combined method |

No result may describe C0-R or `constant20` as the proposed method.

## 2. Global Execution Rules

1. Use `Qwen/Qwen3-4B`, risk seed1, the frozen data splits, greedy evaluation, and the registered QLoRA settings.
2. Keep the 120-row heldout set sealed. Do not run seed2/3, a second backbone, SpectraSafe, or behavior-driven hyperparameter search.
3. Keep safety exposure, sample order, target texts, optimizer settings, checkpoint steps, and decoding matched across C0-R, C3a, and C3b.
4. Reviewers A and B must be independent fresh-context reviewers. They receive only the frozen packet and rubric and may not inspect hidden keys, git history, model roles, prior results, or each other's labels.
5. Reviewer C receives only disagreement rows and the rubric. The main executor may construct packets and compute statistics but may not assign blind labels.
6. Use core100 only for development evaluation after raw runs freeze. Never use core100 behavior to tune queue size, temperature, `eta`, loss weights, checkpoints, or target construction.
7. Mechanism diagnostics added below are prespecified explanatory outcomes. They do not replace behavior gates and cannot rescue a failed candidate.
8. Do not commit checkpoints, caches, model weights, hidden heldout content, or unrelated paper files.

## 3. Work Package P0: Harden and Re-freeze C-R0

### 3.1 Purpose

Repair two implementation gaps before reviewers see the packet. This is a prospective protocol correction because no C-R0 labels exist.

### 3.2 Required implementation

- Preserve the existing 40 canonical pairs, eight reversed duplicates, randomization seed `20260723`, response text, and left/right assignment. Regeneration must prove byte-identical packet and hidden-key content unless the verifier itself requires a manifest refresh.
- Report reversed-pair self-consistency separately for pair preference and response-side specificity after canonical side mapping.
- Require, for each Reviewer A and Reviewer B:
  - preference consistency `>=7/8`;
  - specificity-vector consistency `>=7/8`, where both response-side labels must match after reversal;
  - naturalness reversal consistency as a descriptive sensitivity result.
- Make `overall_pass` false when either required reversal gate fails, even when cross-reviewer agreement passes.
- Replace one-direction EOL handling with symmetric verification:
  - compare raw bytes first;
  - test canonical `CRLF -> LF` bytes;
  - test reconstructed `LF -> CRLF` bytes only to classify legacy EOL-only differences;
  - never treat a non-EOL content change as a match;
  - write new protocol artifacts in canonical LF.
- Add synthetic fixtures covering preference reversal, specificity reversal, one-sided specificity failure, both EOL directions, non-EOL mutation rejection, hidden-field isolation, and zero-variance kappa/AC1 behavior.

### 3.3 Required evidence

- protocol amendment report explaining why the correction occurred before labels;
- old and new packet hashes with an equality decision;
- legacy Stage0 47-artifact verification with raw, EOL-only, and non-EOL counts;
- current C-R0 manifest verification from a fresh checkout;
- unit/synthetic test report and exact commands;
- confirmation of zero reviewer labels, zero optimizer steps, and sealed heldout.

### 3.4 Stop and continuation rule

Stop if packet content or pair randomization changes unexpectedly, any non-EOL Stage0 mismatch remains, or synthetic tests fail. Otherwise commit and push P0, then continue directly to P1.

## 4. Work Package P1: Complete C-R0 Independent Review and Decision

### 4.1 Review execution

- Launch independent Reviewer A and Reviewer B in separate fresh contexts.
- Validate all 48 rows, allowed values, unique opaque IDs, concise reasons, and reviewer provenance before comparing labels.
- Freeze A/B files and hashes before generating the Reviewer C packet.
- Generate Reviewer C rows only for disagreements in preference, response-side specificity, or naturalness. Reviewer C must remain blind to roles and hidden keys.
- Freeze C labels, final canonical labels, and every review artifact before joining source roles.

### 4.2 Reliability and bias analysis

Compute and report:

- pair-preference raw agreement, Gwet AC1, and Cohen's kappa on 40 canonical pairs;
- response-side specificity agreement, AC1, and kappa on 80 canonical response labels;
- A/B preference and specificity reversal consistency on eight duplicate pairs;
- naturalness agreement and reversal consistency as sensitivity outcomes;
- non-tie counts, left/right preference shares, and absolute position imbalance for each reviewer;
- number and type of A/B disagreements, Reviewer C adjudication coverage, and label prevalence;
- category-level summaries as descriptive results only, with no category promotion claims.

Apply every original C-R0 gate plus both new specificity reversal gates. Eight duplicates are quality-control rows and must not inflate the 40-pair outcome denominator.

### 4.3 One allowed recovery

If the first complete session fails, permit at most one prospective rubric clarification. Archive the entire failed session, rebuild fresh opaque IDs and order, and obtain complete new A/B/C reviews. Do not reuse any failed-session labels. A second failure closes C-R0.

### 4.4 Deliverables and route

Produce the final labels, agreement report, bias report, artifact manifest, human-readable route report, and machine-readable decision. If C-R0 fails, commit and push the stop package. If it passes, freeze the future response-quality endpoint, commit and push P1, and continue directly to P2.

## 5. Work Package P2: C-R1 No-Update Feasibility and Mechanism Atlas

### 5.1 Common implementation

Implement one shared module for prompt/response token masking, final-layer pooling, L2 normalization, the two-layer GELU projection head, multi-positive InfoNCE, optional same-category hard negatives, and queue handling. C3a and C3b may differ only through registered candidate flags.

Add tests for padding and response masks, multiple positives, singleton categories, queue wraparound, deterministic batches, finite gradients, candidate equivalence when the hard term is disabled, and prohibition on optimizer/scheduler construction.

### 5.2 Frozen numerical selection

- Try queues `32 -> 16 -> 8` and choose the largest with at least 10% total VRAM headroom.
- Evaluate temperatures `0.05`, `0.10`, and `0.20` using the frozen positive-probability rule.
- Calibrate one finite positive `eta` to a binding-gradient/safety-gradient norm ratio in `[0.20, 0.30]`.
- Use frozen calibration batches and no generated behavior. Do not expand any grid.

### 5.3 Prespecified mechanism diagnostics

For C3a and C3b, report before any update:

- same-prompt positive cosine distribution;
- same-category cross-prompt and cross-category cosine distributions;
- positive-minus-global-negative and positive-minus-hard-negative gaps with bootstrap intervals;
- prompt-to-target retrieval top-1, top-5, mean reciprocal rank, and multi-positive recall;
- shuffled prompt-target permutation control using the same batch sizes;
- embedding covariance effective rank and top singular-value share to detect collapse;
- binding-versus-safety gradient cosine globally and by LoRA layer group;
- layerwise gradient-energy share, finite/non-zero coverage, selected `eta`, and realized ratio;
- allocated/reserved VRAM, wall-clock time, and numerical stability.

These diagnostics explain whether the loss encodes prompt identity, category shortcuts, or gradient conflict. Original C-R1 gates alone determine eligibility.

### 5.4 Deliverables and route

Produce tested code, immutable configs, diagnostic tables, plots, raw numerical arrays, environment record, manifest, and a candidate route report. Commit and push P2. Stop if neither candidate passes. If one or both pass, continue directly to P3 with only eligible candidates.

## 6. Work Package P3: C-R2 Common-Trainer Runs and Raw-Evidence Freeze

### 6.1 Trainer qualification

- Use one trainer implementation for C0-R and every eligible C3 arm.
- Run a two-step deterministic qualification for each arm and verify matched safety draws, target IDs, batch order, optimizer groups, trainable parameter names, and checkpoint schedule.
- Verify that C0-R has no active binding gradients and that disabling the hard term makes the C3b implementation numerically match C3a on the same fixture.
- Record package versions, CUDA/device state, peak VRAM, and resume semantics.

### 6.2 Full seed1 runs

Train C0-R plus eligible C3 arms through step64 and save evaluation checkpoints at steps 0, 16, 32, and 64. Log total, safety, quality, and hard-negative losses; learning rate; gradient norms; selected `eta`; queue occupancy; positive probability; and step time.

At every evaluation checkpoint, run the frozen no-update mechanism probes from P2 on the same diagnostic examples. Track retrieval, similarity gaps, effective rank, safety-binding gradient cosine, and layerwise energy. These trajectories are explanatory and cannot select a checkpoint; step16/32/64 behavior is always evaluated.

### 6.3 Freeze raw evaluation evidence

- Generate matched greedy core100 outputs for every arm/checkpoint.
- Run the frozen utility evaluation and preserve per-example outcomes.
- Build arm-hidden safety packets without opening quality role keys.
- Record prompt coverage, duplicate IDs, generation completeness, decoding equality, output hashes, run/config/code hashes, and wall-clock runtime.
- Freeze raw runs, packets, keys, and manifests before any independent outcome review.

Commit and push P3 raw evidence. Checkpoints remain external and are referenced by hashes and paths. Continue directly to P4.

## 7. Work Package P4: C-R2 Blind Evaluation, Mechanism Link, and Winner Decision

### 7.1 Safety and utility first

Obtain independent arm-hidden safety labels at steps 16, 32, and 64. Adjudicate disagreements, freeze labels, and apply all hard safety, persistence, invalid/mixed, and utility gates before opening candidate-versus-C0 quality keys.

Exclude failed candidates immediately. Do not inspect pairwise quality to rescue a hard-gate failure.

### 7.2 Prompt-aware quality review

For each survivor, build matched arm-hidden candidate-versus-C0 pairs only where both responses are clean-safe. Use fresh independent A/B reviewers and disagreement-only C review under the passed C-R0 protocol. Reapply preference, specificity, reversal, non-tie, position-bias, and adjudication gates separately for every candidate packet.

Compute all-prompt specific-safe improvement, paired prompt-cluster bootstrap intervals, wins/losses/ties, win share excluding ties, net win rate, and relevant redirect. Apply the frozen quality gates without threshold changes.

### 7.3 Mechanism-to-behavior analysis

After behavior labels and arm keys freeze, join the mechanism trajectories and report:

- whether retrieval or positive-gap improvement precedes specific-safe improvement;
- arm differences in representation collapse and gradient conflict;
- prompt-level association between binding improvement and response-quality outcome;
- category-stratified effects with uncertainty;
- safety-quality Pareto placement across checkpoints.

Treat these as mechanism-consistency analyses, not causal mediation proof. Report null and contradictory findings.

### 7.4 Winner rule

Apply the frozen order: hard-gate pass, quality-gate pass, larger all-prompt specific-safe improvement, lower unsafe-with-mixed when improvements differ by less than `0.05`, then C3a for a remaining tie.

Produce the complete winner/stop report and artifact manifest. Commit and push P4. If no candidate passes, close the C3 family and stop. If Q* is selected, continue directly to P5.

## 8. Work Package P5: C-R3 Persistence-Quality Factorial

### 8.1 Factorial runs

Run or byte-identity reuse C0-R, C4-R, Q*, and C5-R. C4-R and C5-R use the frozen gradient-budget controller with target contribution ratio `0.50`, eight-step warmup, and multiplier clipped to `[0.5, 2.0]`.

Use the same trainer qualification, checkpoints, mechanism probes, raw-evidence freeze, independent safety review, utility evaluation, and prompt-aware quality review as P3/P4. Do not infer an interaction from C5-R alone.

### 8.2 Controller diagnostics

Report target and realized safety contribution, per-step multiplier, clipping frequency, contribution-ratio MAE, gradient cosine, layerwise energy redistribution, unsafe rebound, specific-safe trajectory, and any instability. Compare the two-by-two factorial main effects and interaction with prompt-cluster uncertainty where defined.

### 8.3 Final Step3C-R decision

C5-R must satisfy all frozen hard and quality gates, preserve Q* specific-safe improvement within `0.05`, and pass at least one registered persistence criterion without worsening the other by more than `0.05`. If it fails, retain Q* only if Q* passed P4. Do not claim controller benefit from gradient metrics without behavioral benefit.

Commit and push the final factorial package, then stop. Replication, heldout, seed2/3, and a second backbone require a new prospective plan.

## 9. Commit Boundaries and Return Contract

Use one commit per complete work package:

1. `P0` protocol hardening and re-freeze;
2. `P1` independent C-R0 review and route;
3. `P2` C-R1 implementation, diagnostics, and candidate route;
4. `P3` C-R2 common-trainer raw runs and frozen packets;
5. `P4` C-R2 independent evaluation and winner decision;
6. `P5` C-R3 factorial and final Step3C-R decision.

Each commit report must include commit hash, executed commands, actual runtime, environment, tests, row counts, hashes, reviewer provenance, gate table, pass/fail reason, next authorized package, and explicit confirmation that heldout remained closed. The experiment machine should continue through successful packages without waiting, but it must never combine pre-review raw evidence and post-review decisions into one commit.

## 10. Final Interpretation Boundary

A successful P4 supports a new prompt-response binding method only when it beats the matched C0-R control on independently reviewed prompt-specific safe quality while preserving safety and utility. A successful P5 additionally supports the gradient-budget controller only when registered persistence behavior improves. Representation diagnostics alone establish plausibility and mechanism consistency, not causal proof.
