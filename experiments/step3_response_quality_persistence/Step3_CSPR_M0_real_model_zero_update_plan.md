# Step3 CSPR M0 Real-Model Zero-Update Mechanism Plan

## 1. Decision

CSPR M0 is an exploratory mechanism-feasibility study that is separate from
P2U. P2U external-human measurement is deferred, not passed or waived. M0 may
use mechanically valid but behaviorally unqualified authored candidates only
to test real-model gradient geometry. It cannot select a training target,
establish response quality, or authorize an optimizer step.

The complete M0 code, data-selection rule, model state, numerical gates, and
route table are frozen and pushed before any real-model M0 gradient is
computed. Core100 and heldout remain closed.

## 2. Question

At the frozen Qwen3-4B constant20 step-64 LoRA state, does the registered CSPR
projection produce a non-degenerate preference direction that satisfies the
empirical safety half-space and admits at least one finite dual-Armijo step on
the exact frozen pilot batches?

This is a necessary implementation and mechanism check. It is not evidence of
behavioral improvement or safety generalization.

## 3. Frozen Data

The preference pilot contains eight category-diverse prompts from the existing
`mechanism_dev` split. For each prompt, one mechanically valid authored
candidate is selected by a stable hash and paired with the unchanged original
rejected response. No P2S or P2T review label is used. Candidate correctness is
not assumed.

The safety pilot contains eight category-diverse rows selected by stable hash
from the frozen constant20 natural safety-target manifest. Every selected true,
swapped, wrong-prompt, and safety sequence must fit the 256-token response-only
encoding without truncation.

Wrong-prompt controls use a deterministic derangement of the eight selected
prompts. They are descriptive attribution controls and are not a promotion
gate because CSPR has no theorem requiring a particular wrong-prompt cosine.

## 4. Frozen Model and Losses

- backbone: local `Qwen/Qwen3-4B` in 4-bit NF4;
- trainable state: the frozen constant20 step-64 LoRA checkpoint;
- trainable coordinates: checkpoint LoRA parameters only;
- reference policy: the exact initial M0 checkpoint state;
- LoRA/model mode: evaluation mode to disable dropout while retaining
  gradients;
- preconditioner: identity;
- preference loss: mean DPO pair loss with `beta=0.1` and summed response-token
  log probabilities;
- safety loss: mean response-token causal-LM loss on the eight frozen safety
  rows.

No optimizer or scheduler object is created. Candidate line-search points are
temporary in-memory parameter perturbations. The exact initial trainable state
must be restored before any result is accepted, and no checkpoint is written.

## 5. Controls and Diagnostics

M0 records:

- finite/non-zero LoRA gradient coverage for both objectives;
- preference/safety norms, cosine, unconstrained safety derivative, projection
  activation, KKT residuals, and retained preference fraction;
- the same projection quantities for each of the eight preference pairs;
- an independently recomputed label-swap gradient;
- a deterministic wrong-prompt aggregate gradient and its cosine to the true
  preference gradient;
- dual-Armijo trials for the projected aggregate direction;
- the unconstrained direction's losses at the accepted projected step size,
  when one exists;
- peak VRAM, runtime, exact input/output hashes, and initial/final trainable
  state hashes.

At the initial reference state, swapping chosen and rejected responses must
reverse the DPO gradient. The frozen numerical gate requires cosine at most
`-0.999` and norm ratio in `[0.99, 1.01]`.

## 6. Success Gates

M0 passes only when all conditions hold:

1. all inputs, losses, gradients, directions, and reported values are finite;
2. every selected response has at least one response token and no sequence is
   truncated;
3. preference and safety gradients are non-zero and every registered LoRA
   parameter receives a finite gradient;
4. at least two of eight pairwise preference gradients activate the safety
   projection;
5. the aggregate preference direction activates the projection;
6. the aggregate retained preference fraction is at least `0.10`;
7. normalized directional and KKT residuals are at most `1e-8`;
8. the independently recomputed label-swap control passes;
9. one projected step passes both registered Armijo inequalities within ten
   trials from `2e-4`, contraction `0.5`, Armijo fraction `0.1`, and absolute
   comparison tolerance `1e-6`;
10. the exact initial trainable-state hash is restored;
11. `optimizer_objects=0`, `scheduler_objects=0`, and `optimizer_steps=0`.

These thresholds are engineering feasibility gates on one frozen pilot, not
effect-size or publication thresholds.

## 7. Routes

- structural, hash, truncation, non-finite, or restoration failure:
  `STOP_CSPR_M0_INVALID`;
- label-swap failure: `STOP_CSPR_M0_DPO_CONTROL_FAILED`;
- fewer than two pairwise active projections or inactive aggregate projection:
  `STOP_CSPR_M0_PROJECTION_NOT_EXERCISED`;
- retained preference fraction below 0.10:
  `STOP_CSPR_M0_PROJECTED_DIRECTION_DEGENERATE`;
- no finite jointly accepted step:
  `STOP_CSPR_M0_NO_FINITE_JOINT_STEP`;
- every gate passes: `PASS_CSPR_M0_REAL_MODEL_MECHANISM_FEASIBLE`.

A pass authorizes only a decision about whether external-human P2U measurement
is worth resuming. It does not authorize target qualification, training,
generation, core100, or heldout access.

## 8. Prior Success Estimate

Before M0 outputs exist, the estimated probability of a complete M0 pass is
`0.40-0.55`. The estimate combines a high probability of implementation and
label-swap correctness with lower probabilities that the aggregate projection
is active, retains at least 10% preference progress, and admits a finite
jointly acceptable step. It is a subjective design prior, not a confidence
interval and not used as a gate.

The estimated probability of eventually obtaining a credible multi-seed
behavioral paper result, including later human measurement, is `0.20-0.35` at
M0 entry.

Throughout M0, `optimizer_steps=0` and `training_authorized=false`.
