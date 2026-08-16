# Step3 CSPR M1 Disjoint-Batch Replication Plan

## 1. Decision

CSPR M0 established one-batch real-model mechanism feasibility, but its
preference/safety gradient conflict was weak and its wrong-prompt cosine was
high. M1 therefore tests whether the registered projection behavior replicates
across three new, mutually disjoint batches. M1 remains an exploratory
zero-update mechanism experiment. It does not establish behavior improvement,
authorize training, qualify authored responses, or replace the deferred P2U
external-human measurement.

The complete M1 code, data-selection rule, model state, gates, and route table
must be frozen in a clean pushed commit before any M1 real-model gradient is
computed. Core100 and heldout remain closed.

## 2. Question

At the same frozen Qwen3-4B constant20 step-64 LoRA state used by M0, does the
CSPR projection activate and admit finite joint steps across independently
selected, M0-disjoint batches, including at least one repeat of the finite-step
safety rescue observed in M0?

This tests batch robustness of the implementation-level mechanism. It is not a
test of generated response quality, safety generalization, or training benefit.

## 3. Frozen Data Design

M1 excludes every M0 preference `pair_id` and safety `target_id` before
selection. It then freezes three mutually disjoint batches.

Each preference batch contains six same-prompt pairs from `mechanism_dev`, with
six different categories. Across all three batches there are 18 unique pair
IDs. Six categories with at least two eligible pairs contribute two pairs each,
assigned to different batches; six additional categories contribute one pair
each. Within a pair, one mechanically valid authored response is selected by a
stable hash and paired with the unchanged original rejected response. No human
review label is used and candidate correctness is not assumed.

Each safety batch contains six rows from the frozen constant20 natural-target
manifest, with six different categories. Five categories with at least three
remaining rows contribute one unique row to every batch. Three additional
categories contribute one unique row to one batch each. Across all batches
there are 18 unique safety target IDs.

All true, swapped, wrong-prompt, and safety sequences must fit the 256-token
response-only encoding without truncation. Wrong-prompt controls are a
deterministic within-batch derangement and remain descriptive only.

## 4. Frozen Model and Losses

- backbone: local `Qwen/Qwen3-4B` in 4-bit NF4;
- trainable state: the frozen constant20 step-64 LoRA checkpoint;
- trainable coordinates: checkpoint LoRA parameters only;
- reference policy: the exact initial M1 checkpoint state;
- model mode: evaluation mode with gradients and dropout disabled;
- preconditioner: identity;
- preference loss: mean DPO pair loss with `beta=0.1` and summed response-token
  log probabilities;
- safety loss: mean response-token causal-LM loss over each batch's six safety
  rows.

No optimizer or scheduler object is created. Candidate line-search points are
temporary in-memory perturbations. The exact initial trainable state must be
restored after every batch and at the end. No checkpoint or model output is
written.

## 5. Per-Batch Controls

For each of the three batches, M1 records:

- finite, non-zero LoRA gradient-tensor coverage for each objective;
- six pairwise projection diagnostics and one aggregate projection;
- preference/safety norms, cosine, retained preference fraction, and normalized
  directional/KKT residuals;
- an independently recomputed label-swap gradient;
- a deterministic wrong-prompt aggregate gradient and cosine;
- dual-Armijo trials for the projected aggregate direction;
- unconstrained losses at the accepted projected step size, when one exists;
- initial and restored trainable-state hashes.

The label-swap control requires cosine at most `-0.999` and norm ratio in
`[0.99, 1.01]`. Wrong-prompt cosine is reported but is not a gate because no
CSPR theorem specifies its expected value.

## 6. Registered M1 Gates

M1 passes only when all conditions hold:

1. all frozen inputs, token shapes, losses, gradients, directions, and reported
   values are valid and finite;
2. all 18 preference pair IDs and all 18 safety target IDs are unique, mutually
   batch-disjoint, and disjoint from M0;
3. every batch passes the independently recomputed label-swap control;
4. at least 6 of 18 pairwise preference gradients activate the projection;
5. at least 2 of 3 aggregate batch gradients activate the projection;
6. every active aggregate batch retains at least `0.10` of the unconstrained
   first-order preference decrease;
7. normalized directional and KKT residuals are at most `1e-8`;
8. at least 2 active batches accept a projected dual-Armijo step within ten
   trials from `2e-4`, contraction `0.5`, Armijo fraction `0.1`, and absolute
   comparison tolerance `1e-6`;
9. in at least one active batch, at the same accepted step size, the
   unconstrained direction increases safety loss by more than `1e-6` while the
   projected direction passes both registered Armijo inequalities;
10. the exact initial trainable-state hash is restored after every batch and at
    the end;
11. `optimizer_objects=0`, `scheduler_objects=0`, and `optimizer_steps=0`.

Gate 9 is a replication of a specific M0 finite-step observation, not a claim
that every useful CSPR step must rescue an otherwise harmful update.

## 7. Routes

- structural, hash, truncation, non-finite, residual, or restoration failure:
  `STOP_CSPR_M1_INVALID`;
- any label-swap failure: `STOP_CSPR_M1_DPO_CONTROL_FAILED`;
- insufficient pairwise or aggregate activation:
  `STOP_CSPR_M1_PROJECTION_NOT_REPLICATED`;
- any active batch below the retained-preference threshold:
  `STOP_CSPR_M1_PROJECTED_DIRECTION_DEGENERATE`;
- fewer than two active batches with finite joint steps:
  `STOP_CSPR_M1_FINITE_STEP_NOT_REPLICATED`;
- no registered same-step safety rescue:
  `STOP_CSPR_M1_NO_REPLICATED_SAFETY_RESCUE`;
- every gate passes: `PASS_CSPR_M1_BATCH_ROBUST_MECHANISM`.

Any stop route blocks promotion of the CSPR mechanism claim. A pass supports
only proceeding to external target measurement and a separately preregistered
training study.

## 8. Success Estimate and Interpretation Boundary

After M0 and before M1 outputs exist, the subjective probability of a complete
M1 pass is `0.55-0.75`. The main uncertainty is whether weak aggregate conflict
and the same-step safety rescue recur in disjoint batches. This is a design
estimate, not a statistical confidence interval and not a gate.

The estimated probability of eventually obtaining a credible multi-seed
behavioral paper result remains `0.25-0.40`. M1 cannot increase that estimate by
itself because it performs no training and no independent behavioral
measurement.

Throughout M1, `optimizer_steps=0` and `training_authorized=false`.
