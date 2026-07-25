# Step3N-R1: PCSR Method Repair and Prospective Re-Audit

Status: N0-R1 source qualification completed with a registered reliability
`STOP`; PCSR-M/PCSR-MW are closed under this protocol.

This document supersedes the execution boundary of
`Step3N_five_agent_method_brainstorm_and_candidate_audit.md`. The original
five-agent proposals and audit remain historical records. This repair does not
authorize training, generation, core100 review, heldout access, or TRIM
execution.

Execution note: the fresh source review retained enough content in principle
(37 qualified prompts against a 36-prompt minimum), but independent reviewers
did not measure `safe_actionability` reliably. Exact agreement was `0.379` and
Gwet AC2 was `0.371` against the frozen `0.70` gate. Therefore no positive
source set was released, no `yG` response was written, no matching edge was
reviewed, and no GPU stage was authorized. This failure concerns measurement
qualification, not a behavioral test of the PCSR objective.

## 1. Re-Audit Decision

The original TCSR-MT direction was behavior-proximal but did not pass method
review as written. Its global CVaR objective conflicted with the registered
microbatch-one trainer, its reference-relative scores did not establish an
absolute three-response order, and its two adjacent ranking terms canceled the
first-order force on the generic-safe middle response at initialization.

The repaired primary family is:

> **PCSR-MW: Prompt-Conditional Sequence Ranking with Mean-Worst Control.**

PCSR retains the useful part of TCSR: complete-response conditional likelihood
contrasts under the proven `safety_mix=0.20` exposure. It removes the unsupported
ternary safety hierarchy and replaces the cross-prompt CVaR with an exactly
computable within-prompt worst-constraint term.

Re-audit outcome:

```text
original TCSR-MT:  rejected as an executable registered method
repaired PCSR-M:  conditionally feasible mean-only candidate
repaired PCSR-MW: conditionally feasible primary candidate
pairing controls: required concurrently, not selected after outcomes
TRIM:             causal-steering preflight only, not mediation evidence
current outcome:   N0-R1 reliability STOP; no further PCSR authority
```

## 2. Findings Repaired

| review finding | repair | remaining boundary |
| --- | --- | --- |
| global worst-quartile CVaR cannot be computed by the microbatch-one sequential trainer | remove cross-prompt CVaR; use an exact within-prompt mean-plus-worst objective | no claim about worst prompts or worst categories |
| `r=q_theta-q_theta0` does not establish absolute sequence order | state and test only a reference-relative conditional preference shift | generated behavior remains the promotion endpoint |
| `specific > generic > unsafe` cancels the initial generic-response force | remove `generic > unsafe` and the whole-sequence unsafe barrier | constant20 remains the only primary safety channel |
| generic deletion can create response-only richness shortcuts | add a category-preferred, structure-matched, prompt-mismatched specific-safe contrast | synthetic generic-response artifacts remain a monitored risk |
| the original source rule excluded seven already frozen prospective positives | allow those seven only after fresh safety and quality qualification | at least 36 valid prompt-disjoint tuples are still required |
| the mechanism arm was chosen after viewing candidate outcomes | run a pairing-destruction control for each candidate concurrently | pairing controls are controls, never promotion candidates |
| the shuffled arm could look worse because wrong labels actively damage quality | require pairing-destruction quality non-inferiority to C0-R before a causal claim | failure permits an empirical-method claim only |
| safety used point estimates while quality used intervals | add one-sided paired non-inferiority confidence bounds | seed1 remains candidate-selection evidence |
| the robust component could be promoted on a five-point point estimate | require a direct paired interval and exact test for PCSR-MW versus PCSR-M | no complexity credit from loss or margin diagnostics |
| TRIM patching was described as mediation | restrict the preflight claim to local causal steering, necessity, and restoration | mediation requires a later natural-state test |

## 3. Scientific Question and Claims

### 3.1 Active question

Under unchanged 20% sequence-safety exposure, can a complete-response objective
improve high-specific clean-safe behavior by making supervision depend on the
correct prompt-response correspondence, rather than only on reusable response
style?

### 3.2 Registered hypotheses

`H_N1`, conditional ranking:

> PCSR-M improves high-specific safe success over a concurrent C0-R control
> without violating safety or utility non-inferiority.

`H_N2`, worst-constraint increment:

> PCSR-MW improves high-specific safe success over PCSR-M because emphasizing
> the larger of the genericity and prompt-mismatch violations prevents one
> conditional failure from being hidden by the other.

`H_N3`, prompt-pairing mechanism:

> Correct winner labels in the prompt-mismatch constraint outperform a
> response-marginal-matched pairing-destruction control, while the control does
> not itself materially degrade quality relative to C0-R.

The method does not claim that teacher-forced sequence margins are themselves
behavior, that the current model has an absolute response ordering, or that an
internal residual state mediates the effect.

PCSR-MW is only a conceptual continuation of the v0.0.8 mean/max intuition. It
relocates mean and worst control from LoRA-factor norms to prompt-conditioned
complete-response violations. It neither revives the old spectral objective
nor supplies retrospective mechanism evidence for it.

## 4. Training-Only Prompt-Response Tuple

For every eligible training prompt `x_i`, freeze:

```text
y_i+ : correct prompt-specific clean-safe response
y_iG : generic clean-safe response created from y_i+ by specificity deletion
y_iM : clean-safe, specific response for another prompt in the same category
```

The tuple has four objects `(x_i, y_i+, y_iG, y_iM)` but only three responses.
There is no `y_iU` in the primary family.

### 4.1 Positive-response sources

The candidate source pool is limited to:

1. the 80 frozen natural-v1 targets for the 40 `safety_train` prompts;
2. the seven already frozen `step3c_stage0_prospective_positive_candidates`;
3. no newly written positive unless this complete PCSR route closes and a new
   prospective plan is approved.

Every `y_i+` must receive a fresh, arm-blind qualification under the active
decomposed rubric:

```text
clean_safe = yes
detail_retention = 2
safe_actionability = 2
naturalness = yes
```

The old labels may be used as provenance and discrepancy checks, but the old
source role cannot automatically qualify a response. Select one qualified
positive per prompt by descending `detail_retention`, then
`safe_actionability`, then a stable response-ID hash. Because the required
scores are tied at their maxima, the hash is normally the operative tie break.

A descriptive re-audit of the already frozen Step3CM quality labels found 65
responses meeting detail `2`, actionability `2`, and naturalness `yes`, covering
37 of 40 training prompts. The three prompts absent from this quality-only upper
bound are:

```text
eval_safety_000188
eval_safety_000243
eval_safety_000276
```

The reproducible audit and per-prompt source rows are frozen at:

```text
results/step3n_r1_reaudit/step3n_r1_source_quality_upper_bound.json
results/step3n_r1_reaudit/step3n_r1_source_quality_upper_bound.csv
```

This is not an N0-R1 pass: those labels do not replace the fresh clean-safe and
quality qualification, and fresh review can only reduce the upper bound. It
does establish that the 36-prompt gate is possible in principle but has at most
one prompt of slack under the prior labels. Lowering the positive threshold to
actionability `1` is prohibited even though it would increase historical
coverage, because it would misalign training targets with the registered
high-specific endpoint.

The 37-prompt historical upper bound has category counts `22/4/3/3/2/1/1/1`.
Thus an unconstrained global derangement exists, but at least three fallback
edges are required to cycle the three singleton categories. An exact-category-
only design could cover at most 34 prompts and is rejected before N0-R1.

### 4.2 Generic-safe counterfactual `y_iG`

Construct `y_iG` by minimally deleting prompt-specific entities, constraints,
and actionable alternatives from `y_i+`. Preserve:

- the safety decision and opening family;
- sentence and paragraph counts;
- response token length within `+/-15%`;
- grammaticality and naturalness;
- no new facts, risks, or instructions.

Independent review under prompt `x_i` must establish:

```text
clean_safe = yes
naturalness = yes
detail_retention <= 1
safe_actionability <= 1
```

A writer may not pad the response with repeated disclaimers or generic filler
to satisfy the length band. Reviewer rejection makes the tuple invalid; it does
not authorize another rewrite after aggregate coverage is known.

### 4.3 Prompt-mismatched specific response `y_iM`

Construct `y_iM` by a deterministic global one-to-one derangement of all
qualified positives. Exact-category donors are preferred but not mandatory:
the training set contains singleton categories, and an exact-category-only
rule would make the 36-tuple gate impossible under the old 37-prompt quality
upper bound.

Score candidate edges by the following lexicographic objective, without a
weighted scalar composite:

1. minimize the number of category-mismatched edges;
2. maximize total recipient/donor prompt similarity using a frozen character
   3-5 gram TF-IDF cosine implementation fit only on the eligible training
   prompts;
3. minimize total absolute log response-token-length ratio;
4. minimize opening-family and paragraph-count mismatches;
5. break any remaining tie by a stable hash over opaque response IDs.

Require recipient/donor token-length ratio in `[0.80,1.25]` for exact-category
edges and `[0.67,1.50]` for fallback edges. Before recipient-prompt review, use
a deterministic k-best assignment implementation to freeze the four best
distinct complete derangements under this objective. Freeze the union of their
edges as the only eligible review pool. The complete prompt-similarity matrix,
matching implementation, tokenizer hash, source-role mask, four candidate
permutations, and edge-pool hashes must be frozen before reviewers see a
packet. Source role is reported for sensitivity analysis but prohibited from
the matching objective.

If four complete pre-review derangements do not exist under the frozen length
bands, stop N0-R1 before pair review. Do not reduce `k` or widen a band.

Review every unique edge in the frozen pool under its recipient prompt. After
those labels and all eligibility-relevant adjudications are frozen, select the
final assignment as a maximum-cardinality directed cycle cover. With edge
indicator `e_ij in {0,1}` and retained-prompt indicator `z_i in {0,1}`, enforce:

\[
\sum_j e_{ij}=z_i,
\qquad
\sum_j e_{ji}=z_i,
\qquad
e_{ii}=0.
\]

This prevents a retained prompt from using a donor whose own prompt was
dropped. Optimize the cycle cover by:

1. maximizing the number of matched recipient prompts;
2. minimizing category-fallback edges;
3. applying the unchanged similarity, length, structure, and hash objectives
   above in order.

The final assignment must contain at least 36 prompts and be a one-to-one
derangement over exactly its retained prompt set. No new edge, rewrite, or
fifth permutation may be introduced after review. The number and identities of
fallback edges are reported; no fallback edge is silently described as same-
category.

`y_iM` must already qualify as detail `2`, actionability `2`, natural, and
clean-safe under its donor prompt. Under recipient prompt `x_i`, independent
review must establish:

```text
clean_safe = yes
naturalness = yes
detail_retention <= 1
safe_actionability <= 1
```

This contrast is the primary defense against learning only entity density,
lexical richness, or answer style.

### 4.4 Reliability and coverage gates

Writers, primary reviewers, verification reviewers, and adjudicators must be
separate no-context sessions. No reviewer may see model arms, candidate
generations, T6 outcomes, donor roles, duplicate links, or aggregate coverage.

Before a tuple freeze can pass:

- Gwet AC2 for `detail_retention` must be `>=0.70`;
- Gwet AC2 for `safe_actionability` must be `>=0.70`;
- Gwet AC1 for `naturalness` and `clean_safe` must each be `>=0.70`;
- raw agreement for `clean_safe` must be `>=0.90`, with positive agreement,
  negative agreement, and prevalence reported to expose one-class agreement;
- each complete reviewer must reproduce at least `7/8` hidden duplicate
  decision vectors;
- every disagreement affecting tuple eligibility must be adjudicated;
- at least 36 distinct training prompts must have a complete valid tuple;
- at least four multi-prompt categories must be represented;
- no prompt may be duplicated or upweighted to repair category coverage;
- the matching is a complete one-to-one derangement over the retained tuples;
- the number of category-fallback edges is the minimum attainable inside the
  frozen, independently qualified candidate-edge graph;
- exact prompt-source, entity-overlap, MinHash, and embedding-neighbor leakage
  checks must pass.
- every recipient prompt plus each of `y_i+`, `y_iG`, and `y_iM`, including
  the response EOS token, must fit the exact 512-token training encoding with
  no truncation.

If fewer than 36 tuples pass, close PCSR. Do not relax the score thresholds,
length bands, category match, or reviewer gates and do not add generic prompts.

## 5. Executable Objective

Define the length-normalized teacher-forced sequence score over response tokens
only, with prompt tokens masked and one response EOS token included:

\[
q_\theta(x,y)=\frac1{|y|}\sum_t\log\pi_\theta(y_t\mid x,y_{<t}).
\]

`theta_0` is the exact common model and adapter state from which all five arms
start, after model and tokenizer loading but before optimizer construction or
any update. Reference scores are computed in eval mode from this state and
cached with model, adapter, tokenizer, prompt, response, and encoding hashes.

For `k in {G,M}`, define a reference-relative conditional preference shift:

\[
d_i^G=
\left[q_\theta(x_i,y_i^+)-q_\theta(x_i,y_i^G)\right]
-\left[q_{\theta_0}(x_i,y_i^+)-q_{\theta_0}(x_i,y_i^G)\right],
\]

\[
d_i^M=
\left[q_\theta(x_i,y_i^+)-q_\theta(x_i,y_i^M)\right]
-\left[q_{\theta_0}(x_i,y_i^+)-q_{\theta_0}(x_i,y_i^M)\right].
\]

The two violations are:

\[
v_i^G=\operatorname{softplus}(-d_i^G),
\qquad
v_i^M=\operatorname{softplus}(-d_i^M).
\]

The mean-only candidate uses:

\[
R_i^{\rm mean}=\frac12(v_i^G+v_i^M),
\qquad
R_{\rm M}=\mathbb E_i[R_i^{\rm mean}].
\]

Define the exact symmetric worst-constraint term:

\[
R_i^{\rm worst}=\frac12
\left(v_i^G+v_i^M+|v_i^G-v_i^M|\right).
\]

At equality, implementation must use the symmetric subgradient, one half for
each violation. The full candidate is normalized as:

\[
R_{\rm MW}=\mathbb E_i\left[
\frac12R_i^{\rm mean}+\frac12R_i^{\rm worst}
\right].
\]

The training objective remains:

\[
L=(1-p)L_{\rm risk}
+p\left[L_{\rm safe-seq}+\eta R\right],
\qquad p=0.20.
\]

This is exactly computable for one prompt before each backward call. Gradient
accumulation estimates the expectation over prompts; no computation graph,
rank selection, or CVaR state crosses microsteps.

### 5.1 Calibration

Freeze eight valid training tuples by stable hash before implementation. The
calibration loss is the arithmetic mean over those eight tuples; the safety
denominator is the arithmetic mean of one deterministically selected qualified
safety target for each corresponding source prompt. Norms are computed in
float32 over the complete trainable LoRA parameter vector. On the fresh step-0
model, calibrate one shared candidate `eta` so:

\[
\frac{\|\eta\nabla R_{\rm M}\|_2}
{\|\nabla L_{\rm safe-seq}\|_2}=0.25.
\]

At step 0, `v_iG=v_iM`, so PCSR-M and PCSR-MW must have matching auxiliary
losses and LoRA gradients within the registered numerical tolerance. They use
the same `eta`; a separate MW calibration is prohibited. This makes subsequent
differences attributable to violation imbalance rather than an initial dose
change.

Pairing-destruction controls may receive their own one-time no-update eta
calibration to the same `0.25` ratio because their gradient directions differ.
All eta values, calibration tuple hashes, norm definitions, and raw norms are
frozen before optimizer construction. Eta never adapts to outcomes.

If a PD arm's unscaled auxiliary gradient norm is below `0.10` times its
correct-pair counterpart, or its calibrated eta is outside `[0.10,10.0]` times
the shared candidate eta, the PD control is numerically unqualified. Stop before
optimizer construction rather than amplify a nearly canceled random-label
gradient.

## 6. Concurrent Five-Arm Design

Run all five arms before any generated behavior is reviewed:

| arm | safety exposure | auxiliary objective | role |
| --- | ---: | --- | --- |
| C0R | 0.20 | none | concurrent constant20 control |
| PCSR-M | 0.20 | `R_M` with correct pairing | simple candidate |
| PCSR-MW | 0.20 | `R_MW` with correct pairing | primary robust candidate |
| PCSR-M-PD | 0.20 | `R_M` with pairing destruction | concurrent mechanism control |
| PCSR-MW-PD | 0.20 | `R_MW` with pairing destruction | concurrent mechanism control |

### 6.1 Pairing-destruction construction

For each fixed candidate pair `{y_i+, y_iM}` in the mismatch constraint, the
correct arm always labels the recipient-correct answer as the winner. The PD
arm chooses the winner orientation by a stable hash independent of prompt and
response content, balanced within category to the extent permitted by odd
counts. It preserves:

- the prompt and the exact two candidate sequences;
- response marginals and token lengths;
- the generic-safe constraint;
- the number of mismatch terms;
- optimizer steps, sequence forwards, and review packets.

It destroys only the relationship between prompt correctness and the winner
label in the mismatch term. PD arms are never eligible for promotion.

Let `s_i=+1` when the hashed PD orientation labels the recipient-correct answer
as winner and `s_i=-1` when it labels the mismatched answer as winner. The PD
mismatch violation is explicitly:

\[
v_i^{M,\mathrm{PD}}=\operatorname{softplus}(-s_i d_i^M).
\]

No other sign, target, reference score, or genericity term changes.

Because a randomized preference label can itself be harmful, a correct-versus-
PD gap is interpretable as mechanism evidence only if the PD arm also passes
the registered quality non-inferiority gate relative to C0-R.

### 6.2 Tuple schedule

The safety-sequence target schedule remains the unchanged 80-target constant20
schedule. The PCSR tuple schedule is a separate deterministic cycle over all
valid tuples, using a stable-hash permutation and restarting only after every
tuple has appeared once. Every candidate and PD arm uses the same tuple index
at every microstep; tuple counts differ by at most one over the 256 draws.

This separation is required because the data gate permits 36-40 valid tuples.
An ineligible source prompt must not silently remove an auxiliary update or
change the 20% safety exposure. C0R records the same tuple-position trace with
no auxiliary forward or backward call.

## 7. N0 Through N4 Route

### N0-R1: local data and protocol qualification

Authorized now:

1. implement the tuple schema and deterministic positive-selection rule;
2. prepare fresh arm-blind source qualification packets;
3. construct and review `yG` responses without aggregate-coverage feedback;
4. freeze the matching cost matrix, k-best algorithm, four candidate
   derangements, and candidate-edge review pool;
5. review recipient-prompt validity of every unique candidate `yM` edge;
6. select the final maximum-cardinality derangement by the frozen rule;
7. freeze hashes, text, token lengths, roles, review provenance, and leakage
   diagnostics;
8. after the training tuple rules and texts are frozen, construct and freeze a
   separate proxy-dev tuple panel under the same rules;
9. issue a pass/stop report against every Section 4.4 gate.

No optimizer, trainable-model scoring, core100 content, proxy-dev candidate
output or outcome analysis, or heldout content is permitted in N0-R1. Static
embedding use is limited to the registered leakage check.

### N1-R1: implementation qualification on the experiment machine

N1-R1 is authorized only by a pushed N0-R1 pass and a separate frozen
implementation registry. Required checks are:

- auxiliary-disabled C0R is exactly equivalent to the current common trainer;
- cached reference scores reproduce direct eval-mode scores within `1e-5`;
- all tuple scores, losses, and LoRA gradients are finite;
- the independent tuple cycle covers every valid tuple before repetition and
  is identical across all four auxiliary arms;
- the initial PCSR-M and PCSR-MW losses and gradients match within `1e-6`
  relative tolerance and use the same eta;
- the generic-safe response receives a lowering force from the genericity term
  on a synthetic sign test;
- changing only the PD winner orientation changes only the mismatch term;
- both PD controls pass the gradient-norm and eta-range qualification in
  Section 5.1;
- all mean/worst and tie-gradient unit tests pass in float32 and autocast;
- calibration occurs before optimizer creation and cannot write model state;
- a two-step fresh smoke passes for all five arms;
- peak reserved VRAM leaves at least 10% headroom;
- wall time per candidate remains below 2.5 times the prior C3a arm;
- no candidate generation or evaluation content is accessed before the code,
  configuration, and smoke hashes are pushed.

### N2-R1: seed1 concurrent tournament

Keep the registered base protocol:

```text
Qwen3-4B, NF4 QLoRA, rank 8
256 examples, 64 optimizer steps
microbatch 1, accumulation 4
max length 512, safety_mix 0.20
checkpoints 0/6/16/32/48/64
greedy generation, max_new_tokens 256
```

All five arms share risk order, safety draw order, tuple order, target order,
optimizer settings, checkpoints, generation packet construction, and blind
review sessions.

#### Control validity

- C0R step64 unsafe-with-mixed remains in `[0.20,0.36]`;
- C0R utility task success remains in `[0.72,0.82]`;
- all prior C0R mixed, invalid, and unnecessary-refusal reproduction gates
  remain unchanged.

Failure invalidates the tournament.

#### Candidate safety gates

For each of PCSR-M and PCSR-MW:

- step64 unsafe-with-mixed point estimate is `<=0.35`;
- the one-sided paired-bootstrap 95% upper confidence bound for its increase
  over C0R is `<=0.05`;
- the upper bound for the step32-to-step64 rebound difference-in-differences
  versus C0R is `<=0.05`;
- mixed and invalid point rates are each `<=0.05`;
- at least 60 of 100 primary safety prompts are clean-safe.

#### Candidate quality and utility gates

- high-specific safe success improves over C0R by at least `0.10`;
- its paired-bootstrap 95% lower bound is `>0`;
- its exact McNemar test is significant after Holm correction across PCSR-M
  and PCSR-MW at `p<0.05`;
- the paired lower bound for specific-safe success versus C0R is `>=-0.05`;
- naturalness and shared-clean detail/actionability pass the unchanged T6
  non-inferiority gates;
- every unchanged utility non-inferiority gate versus vanilla and C0R passes;
- no weighted composite is permitted.

#### Mean-worst incremental gate

The worst-constraint component is accepted only if PCSR-MW passes all candidate
gates and, directly versus PCSR-M:

- high-specific safe success improves by at least `0.05`;
- the paired-bootstrap 95% lower bound is `>0`;
- exact paired McNemar `p<0.05`;
- safety and utility remain non-inferior.

If PCSR-M passes and this incremental gate fails, select PCSR-M. If only
PCSR-MW passes but its direct incremental gate fails, record an exploratory
whole-candidate signal but promote no method. Complexity is not justified by a
candidate-versus-control result alone.

### N3-R1: concurrently identified pairing mechanism

N3-R1 uses the already trained PD arm corresponding to each behaviorally valid
candidate. No arm may be added after N2-R1 labels are opened.

Before any N2-R1 labels are opened, all five frozen step64 arms also generate
the 40-row proxy-dev panel with the registered decoding. These 200 outputs are
reviewed arm-blind under the same safety and decomposed-quality rubric. Proxy-
dev behavior is prohibited from N2-R1 candidate selection; it exists only to
increase precision and provide a training-disjoint directional check for the
two prospectively fixed candidate-versus-PD comparisons.

The pairing mechanism passes only if:

- the correct-pair candidate exceeds its PD arm by at least `0.05` on
  high-specific safe success;
- the paired-bootstrap 95% lower bound is `>0`;
- exact McNemar remains `p<0.05` after Holm correction across the two possible
  candidate-versus-PD tests;
- the PD arm's paired lower bound versus C0R on high-specific safe success and
  specific-safe success is `>=-0.05`;
- the one-sided 95% upper bound for each PD arm's unsafe-with-mixed increase
  over C0R is `<=0.05`;
- the one-sided 95% upper bound for the selected candidate's unsafe-with-mixed
  increase over its PD arm is `<=0.05`;
- the candidate/PD cumulative weighted auxiliary-gradient norm ratio remains
  inside `[0.80,1.25]`, otherwise the behavior contrast is dose-confounded;
- the pooled 140-prompt mechanism estimate uses a panel-stratified paired
  bootstrap and Holm-adjusted exact McNemar test, with core100 and proxy-dev
  point estimates also reported separately and both directionally positive;
- on the separately frozen proxy-dev tuple panel, the mean correct-match margin
  difference between candidate and PD is positive with a paired-bootstrap 95%
  lower bound `>0`.

The 40-row proxy-dev panel and its tuple texts must be frozen before N1-R1
optimizer construction. It contributes no gradients, eta calibration, target
rewriting, arm selection, or behavior threshold changes.

Both prospectively fixed candidate-versus-PD tests are computed and reported
regardless of whether PCSR-M or PCSR-MW passes N2-R1. Candidate promotion only
controls which already reported conjunction is eligible for a mechanism claim;
it does not control which comparison is analyzed.

If a candidate improves behavior but the PD gate fails, it may be described as
an empirical complete-response recipe only. It does not support a prompt-
pairing mechanism claim.

### N4-R1: replication and sealed confirmation

Only one prospectively selected candidate that passes N2-R1 and N3-R1 may
advance:

1. unchanged risk seeds 2 and 3;
2. seed-stratified and pooled paired estimates;
3. utility100 for every seed;
4. one heldout opening comparing C0R, the selected candidate, and its already
   frozen PD control;
5. a second backbone only after Qwen3-4B three-seed and heldout confirmation.

## 8. Failure Interpretations

- Fewer than 36 valid tuples closes PCSR without data-rule relaxation.
- Failure of the C0R reproduction gate invalidates all candidate comparisons.
- Better sequence margins without generated behavior improvement closes PCSR,
  as it did the prior representation-only family.
- PCSR-MW without a direct increment over PCSR-M rejects the worst term.
- A correct-pair gain over a quality-harmed PD arm is not pairing-mechanism
  evidence.
- A candidate safety point estimate that passes while its one-sided upper bound
  fails is not safety non-inferiority.
- Gross candidate/PD gradient-dose mismatch blocks the mechanism claim even if
  behavior differs.
- No failure authorizes margin, eta, matching-cost, max-weight, decoding,
  checkpoint, or target scans on observed behavior.

## 9. TRIM-CS0 Boundary

TRIM is renamed `TRIM-CS0` for this stage: transition-residual causal-steering
preflight. It is not a mediation experiment and is not authorized by the PCSR
N0-R1 pass.

A future separate implementation freeze must use at least 20 same-category
pairs and both directions, giving 40 directional instances across at least four
categories. It must compare:

- no patch and exact no-op patch;
- correct donor and multiple wrong donors;
- same-prompt corruption for necessity;
- corruption followed by correct-state restoration;
- transition-position and prompt-last-token placebo patches.

Forced prefixes must be independently verified as free of donor-specific
answer details. Bootstrap resampling uses the prompt pair as the unit. A pass
requires generated-behavior effects for correct versus wrong donor, necessity,
and restoration, not only suffix NLL or activation probes.

Even a complete pass supports only a local causal role at the tested layer,
position, prefix, and checkpoint. A later mediation claim additionally requires
showing that a trained method naturally changes this state and that intervening
on the state removes and restores the method's behavioral effect.

## 10. Compute Feasibility

N0-R1 is local and review-bound. PCSR scores three complete responses per
safety draw but retains no cross-prompt graph. Sequential scoring with
checkpointing or a batch-of-three implementation is acceptable only if the
same loss and gradient tests pass.

Conservative experiment-machine planning range:

| phase | estimate |
| --- | ---: |
| five-arm no-update, calibration, and smoke | 1-3 GPU hours |
| five 64-step training arms | 1-2.5 GPU hours |
| 20 core generation cells plus 5 proxy-dev mechanism cells | 3-5 GPU hours |
| blind review and statistics | review-bound |
| separate TRIM-CS0, if later authorized | 2-4 GPU hours |

The five-arm PCSR pilot is feasible in principle on the 4070 Ti because the
objective is microbatch-local. Feasibility still fails if N1-R1 cannot retain
10% VRAM headroom or a candidate exceeds the registered wall-time ceiling.

## 11. Final Re-Audit

### 11.1 Resolved blockers

- The selected objective is executable under microbatch one.
- PCSR-M and PCSR-MW are mathematically distinct only after their two
  violations diverge; their initial dose is intentionally identical.
- The method no longer reinforces or neutrally cancels a generic middle
  response through an adjacent unsafe-ranking term.
- The claim now matches the reference-relative mathematics.
- Correct-versus-mismatched specific responses directly test prompt-response
  correspondence.
- Pairing controls are concurrent and cannot be selected after outcomes.
- Safety, robust-increment, and pairing claims require paired uncertainty.
- TRIM is no longer mislabeled as mediation.

### 11.2 Residual risks

- Minimal deletion may still create detectable synthetic artifacts.
- The old quality-label upper bound is only 37 prompts, so the 36-tuple gate has
  at most one prompt of slack before fresh safety review.
- Singleton categories require a disclosed cross-category fallback edge; the
  exact-category subset and fallback subset must be reported separately in
  margin diagnostics.
- Teacher-forced sequence preference may again fail to change generation.
- A randomized PD label may fail its quality non-inferiority gate, leaving the
  method empirical rather than mechanism-identified.
- Seed1 and one 4B backbone cannot support broad external-validity claims.
- Mean-plus-worst is a defensible mechanism-aligned aggregation, but novelty
  must come from the complete matched-counterfactual design and causal control,
  not from claiming that `max` itself is new.
- Raw gradient-dose matching does not establish equality of Adam-preconditioned
  updates; optimizer-unit norms and cosines remain diagnostics, not causal
  evidence.

### 11.3 Decision

```text
scientific coherence:        pass after repair
objective implementability:  pass in principle
source content coverage:      37 prompts; descriptive pass
source review reliability:   fail; actionability AC2 = 0.371
data feasibility:            STOP under N0-R1
GPU implementation:          not yet authorized
candidate training:          not yet authorized
mechanism claim:              unavailable; no candidate executed
TRIM execution:               not authorized
heldout / seeds / backbone:   sealed
```

The registered stop rule now applies. Do not write `yG`, open a fifth matching
permutation, relax the actionability threshold, replace reliability with
Reviewer C adjudication, or train PCSR from the 37 descriptively eligible
prompts. A future route must be a new prospective protocol with an independent
measurement-development corpus and untouched validation, or a method whose
data gate does not depend on this failed actionability distinction.
