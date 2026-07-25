# Step3N: Five-Agent New-Method Brainstorm, Audit, and Candidate Selection

Status note: this document remains the historical five-agent proposal and
first-pass audit. Its execution boundary is superseded by
`Step3N_R1_pcsr_method_repair_and_reaudit.md`; TCSR-MT is not authorized for
implementation or training.

## 0. Status

Status: **design audit complete; one primary candidate and one conditional backup selected; no new training is authorized by this document alone**.

This document follows the valid Step3C-T6 route:

```text
R1_no_c3_candidate_passes
```

The current C3 representation-binding recipe remains closed. Seeds 2/3, heldout, a second backbone, C4/C5, and any behavior-driven scan remain sealed. `Step3N` denotes a new-method design route and does not overwrite the historical Step3D factorial that required a C3 winner.

Five isolated agents, with no cross-agent communication and no repository edits, each proposed one method from a different mathematical perspective. The main session then applied one common scientific and feasibility audit. Agent proposals are preserved as ideas, not treated as evidence.

## 1. Frozen Evidence Boundary

The design must explain all of the following results without rewriting them.

1. Step1 found reproducible unsafe drift and useful spectral, subspace, and activation diagnostics, but tested parameter and activation interventions did not control behavior.
2. Step2 showed that norm, spectral, scalar-margin, and token-proxy quantities can move without producing safety.
3. Direct sequence-level safety supervision is behaviorally active.
4. The late weighted safety-gradient contribution controls persistence. A 20% constant safety replay exposure reduced step64 unsafe-with-mixed from `0.56` to `0.28` relative to decay10, but did not produce a complete method.
5. `constant20` means `safety_mix=0.20`, not a loss multiplier of `20`.
6. Constant20 still yielded only `0.72` prompt-relevant redirect and severe safe-response concentration.
7. Step3 validated decomposed quality measurement. C0-R reproduced at step64 with unsafe-with-mixed `0.29`, high-specific safe success `0.66`, detail retention `1.98`, safe actionability `1.45`, and utility success `0.77`.
8. C3a and C3b produced strong internal representation-binding signals but no reliable behavioral quality gain. C3a was `+0.03`, CI `[-0.03, 0.10]`; C3b was `-0.01`.
9. Therefore the next method must couple to complete generated-sequence behavior, preserve the proven 20% persistence channel, and contain a prospective intervention that can falsify its mechanism.

## 2. Common Audit Standard

### 2.1 Weighted criteria

| criterion | points |
| --- | ---: |
| behavior-proximal validity | 20 |
| causal and mechanistic falsifiability | 20 |
| novelty beyond a standard wrapper | 15 |
| feasibility on 12 GB RTX 4070 Ti | 15 |
| compatibility with safety persistence | 10 |
| direct path to quality without template collapse | 10 |
| statistical efficiency and preregistration | 5 |
| defensible relation to v0.0.8 dual regularization | 5 |

The score is a decision aid. A hard veto overrides the total.

### 2.2 Hard vetoes

- proxy-only optimization with no complete-sequence or output-functional coupling;
- renamed KL, DPO, PCGrad, replay, or C3 without a substantive new hypothesis;
- a safety-quality weighted outcome score that permits compensation;
- no causal intervention;
- behavior-driven hyperparameter or checkpoint selection;
- reuse of core100 observations to construct training targets;
- heldout leakage;
- an unqualified online judge used as a training reward;
- a large grid or a method whose first pilot cannot fit the current 64-step, 512-token, 12 GB protocol;
- a mathematical gate that is structurally impossible at the proposed sample size.

## 3. Five Independent Proposals

### 3.1 Agent A: BCRD-LoRA, behavior-constrained primal-dual optimization

The proposal treats safety, detail retention, actionability, naturalness, and utility as separate sequence-level risks:

\[
R_i(\theta)=\mathbb E_{x,y\sim\pi_\theta}[c_i(x,y)],
\qquad R_i(\theta)\leq\tau_i.
\]

It estimates each risk gradient with a likelihood-ratio estimator and updates a separate multiplier:

\[
h_t=g_{\rm task,t}+\sum_i(\lambda_{i,t}+\rho[v_{i,t}]_+)g_{i,t},
\]

\[
\lambda_{i,t+1}=\Pi_{[0,20]}
[\lambda_{i,t}+\eta_\lambda v_{i,t}].
\]

A token-KL trust region rejects unstable updates. Its mechanism claim is closed-loop risk allocation: current behavioral violations should trigger the correction that lowers the same risk in the next round. An equal-budget, time-shifted multiplier replay is the causal control.

Design rationale: unlike SpectraSafe or C3, the optimized quantities are labels on complete generated responses. The primal-dual form also prevents quality gains from numerically compensating for a safety violation.

### 3.2 Agent B: DFGU, differential Fisher guardrail updates

The proposal estimates two empirical output-function metrics in the LoRA tangent space:

\[
\widehat F_S=\frac1{M_S}\sum q_Sq_S^\top,
\qquad
\widehat F_T=\frac1{M_T}\sum q_Tq_T^\top,
\]

and solves

\[
\widehat F_Sv_i=\lambda_i(\widehat F_T+\rho I)v_i.
\]

High generalized-eigenvalue directions are sensitive on safety contexts but relatively less useful on content-matched task contexts. Only the task update is shrunk in this low-rank subspace; the direct 20% safety sequence update is retained. A protected suffix is compared with an add-back suffix and an equal-norm low-eigenvalue sham suffix.

Design rationale: this moves geometry from parameter norms to local output-distribution KL. It directly asks whether task updates repeatedly enter safety-sensitive functional directions.

### 3.3 Agent C: TRIM, transition-residual interventional mediation

The proposal identifies the response position immediately before the first constructive safe redirect token and a fixed middle-late layer

\[
\ell^\star=\lceil 2L/3\rceil.
\]

For a prompt-target pair, it patches the two-token residual state from the correct prompt into a content-matched counterfactual prompt and trains the complete safe suffix:

\[
q_i^{\rm patch}=-\frac1{|y_{i,\geq t_i}|}
\sum_{u\geq t_i}\omega_{i,u}
\log p_\theta(y_{i,u}\mid x_{\pi(i)},y_{i,<u},\operatorname{do}(M_i)),
\]

\[
L_{\rm med}=\frac1N\sum_i
[\delta+q_i^{\rm patch}-\operatorname{sg}(q_i^{\rm cf})]_+.
\]

The causal test swaps correct and incorrect donors during suffix generation and compares the transition position with a prompt-last-token placebo.

Design rationale: C3 showed that prompt information can be retrieved internally without being used by decoding. TRIM targets the proposed decision time at which the model must move from a generic refusal to prompt-specific safe action.

### 3.4 Agent D: TCSR, ternary counterfactual safety ranking

The proposal constructs, for the same prompt, a partial order over complete responses:

\[
y_i^+\;\text{(specific safe)}
\succ y_i^R\;\text{(generic safe refusal)}
\succ y_i^U\;\text{(unsafe execution)}.
\]

With length-normalized sequence score

\[
q_\theta(x,y)=\frac1{|y|}\sum_t\log\pi_\theta(y_t\mid x,y_{<t}),
\qquad
r_\theta=q_\theta-q_{\theta_0},
\]

it penalizes violations of `specific > generic`, `generic > unsafe`, and an absolute unsafe barrier. A same-category pairing shuffle preserves answer marginals while breaking only prompt-specific correspondence.

Design rationale: the method acts in the conditional response-distribution simplex rather than in parameter or hidden-state geometry. Style and length matched counterfactuals are intended to remove refusal wording and length as shortcuts.

### 3.5 Agent E: BASC, behavior-anchored dual-loop supervision control

The proposal maintains task, safety-sequence, and quality-sequence gradients separately:

\[
d_t=g_0+\lambda_{s,t}g_s+\lambda_{q,t}g_q.
\]

It defines effective first-order descent

\[
\phi_i(d_t)=\frac{\langle g_i,d_t\rangle}{\|g_i\|^2+\epsilon}
\]

and solves a two-variable QP so both safety and quality channels receive minimum effective descent. A slower loop decodes a frozen control panel and raises the safety or quality descent target when the corresponding behavior leaves its feasible region. An equal-cumulative-budget open-loop replay tests whether feedback timing, rather than total supervision, causes recovery.

Design rationale: constant20 identified a dynamic persistence problem, while Step3 isolated a distinct quality problem. BASC treats them as two controlled states rather than asking one fixed replay weight to solve both.

## 4. Main-Session Audit

### 4.1 Cross-proposal correction

TCSR, TRIM, and DFGU wrote terms such as `20 L_safe` as if `constant20` were a coefficient. The repository defines the control as

\[
L=(1-p)L_{\rm risk}+pL_{\rm safe},\qquad p=0.20.
\]

Implementing a coefficient of 20 would not reproduce constant20 and would invalidate every comparison. All retained designs must use `safety_mix=0.20` and the existing draw schedule.

Several proposals also assumed 1,000-2,000 updates or 192 new prompts. The active protocol has 40 safety prompt clusters, 80 reviewed targets, 256 examples, 64 optimizer steps, and length 512. Runtime and controller frequency must be assessed against that protocol.

### 4.2 Scores and decisions

| candidate | behavior | causal | novelty | 4070 Ti | persistence | quality | stats | v0.0.8 | total | decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| corrected TCSR family | 20 | 17 | 10 | 13 | 10 | 9 | 4 | 4 | **87** | primary, after correction |
| TRIM | 15 | 20 | 15 | 7 | 9 | 8 | 4 | 3 | **81** | conditional diagnostic backup |
| BASC | 18 | 17 | 13 | 3 | 10 | 9 | 3 | 4 | **77** | do not run now |
| BCRD-LoRA | 20 | 17 | 12 | 2 | 8 | 8 | 3 | 5 | **75** | hard veto |
| DFGU | 12 | 18 | 14 | 7 | 8 | 3 | 4 | 4 | **70** | close for current quality objective |

The numerical ranking does not imply evidence for any method. It records prospective design quality only.

### 4.3 Candidate-specific audit

#### BCRD-LoRA: rejected for the first pilot

The strongest feature is the separation of non-compensating behavioral constraints. The fatal issue is finite-sample feasibility. With only `n=32` samples per round, even for a small number of constraints,

\[
\frac{3\log(2I/\alpha)}{n}
\]

is roughly `0.5` before adding the empirical mean and variance term. The proposed empirical-Bernstein upper bound would therefore report many normalized risks as violated almost permanently. No qualified online training judge currently exists, and 50 on-policy rounds would introduce high-variance policy gradients, API dependence, and many new control parameters. This is a future program, not the next falsifiable experiment.

#### DFGU: scientifically interesting but misaligned with the active failure

The add-back/sham intervention is strong. The objective, however, primarily suppresses task-induced safety drift and contains no direct force toward prompt-specific safe content. Step2 already favored insufficient weighted safety magnitude over a single directional-conflict account. Eight score vectors per side cannot stably identify a six-dimensional generalized eigenspace without a stronger conditioning study, and refreshing every 64 steps means only one refresh in the current run. DFGU may be revisited as a safety-persistence mechanism, but it is not a credible first answer to the measured quality deficit.

#### TRIM: retain only behind a zero-training causal gate

TRIM is the most novel and mechanistically specific proposal. Its risks are equally specific: teacher forcing can leak the target suffix after its first tokens; the two-token residual may become a training-only high-bandwidth shortcut; free-generation transition positions are not fixed; exact token-position alignment and gradient-checkpointing hooks may fail; and the selected layer is theoretical rather than empirically established. A causal patch test on the existing C0-R checkpoint can falsify the premise before any new training, so full TRIM training is not yet justified.

#### BASC: reject under the current protocol

With `T=64`, the proposed `H=max(100,0.05T)` gives no slow-loop update at all. The proposal assumes 1,000-2,000 steps, length 1024, and a qualified online blind metric, none of which matches the frozen setup. Its QP constrains a raw gradient direction while the actual trainer uses AdamW momentum and preconditioning, so the stated descent guarantee does not yet apply to the executed update. BASC could become a second-stage controller after a fixed behavior-proximal quality loss has first demonstrated a real effect.

#### TCSR: retain after three repairs

TCSR is closest to the positive evidence because it uses full-sequence supervision and a direct conditional output order. Three repairs are required:

1. use `p=0.20` replay exposure, never `20 L_safe`;
2. use the existing 40 training prompts and 80 reviewed natural-v1 targets rather than inventing an unavailable 192-prompt set;
3. replace arbitrary fixed auxiliary weights with one prospective gradient-scale calibration and include a concurrent common-trainer control.

Its main weakness is algorithmic novelty: a simple average triplet loss can be dismissed as listwise preference learning. The selected version therefore tests whether a mean-plus-tail behavioral barrier adds robust value, while retaining a simpler mean-only arm as the required incremental ablation.

## 5. Selected Primary: TCSR-MT

### 5.1 Name and claim

**TCSR-MT: Ternary Counterfactual Sequence Ranking with Mean-Tail Control.**

The claim is not that a new scalar proxy predicts safety. The claim is:

> Under the proven 20% sequence-safety exposure, enforcing a prompt-conditioned order among specific-safe, generic-safe, and unsafe complete responses, including the worst quartile of violations, can improve prompt-specific safe behavior without sacrificing persistence or utility.

This is the behavior-space successor to the v0.0.8 mean/max idea. The old method regularized mean and maximum LoRA factor norms. TCSR-MT instead regularizes mean and tail violations in conditional sequence log-odds. The old spectral objective remains a negative baseline and is not reintroduced.

### 5.2 Sequence geometry

For each training prompt `i`, freeze one valid triplet `(y_i^+, y_i^R, y_i^U)`. Define:

\[
a_i^{\rm spec}=r_i^+-r_i^R,
\qquad
a_i^{\rm safe}=r_i^R-r_i^U,
\qquad
a_i^{\rm unsafe}=-r_i^U.
\]

All three quantities should be positive. Their nonnegative violations are

\[
v_i^k=\operatorname{softplus}(-a_i^k),
\qquad
k\in\{\mathrm{spec},\mathrm{safe},\mathrm{unsafe}\}.
\]

Define the global and tail controls:

\[
R_{\rm mean}=\frac1{3N}\sum_{i,k}v_i^k,
\]

\[
R_{\rm tail}=\frac13\sum_k
\operatorname{CVaR}_{0.25}(v_1^k,\ldots,v_N^k),
\]

where empirical `CVaR_0.25` is the deterministic mean of the largest `ceil(0.25N)` violations. The complete objective is

\[
L=(1-p)L_{\rm risk}
+p\left[L_{\rm safe-seq}+\eta R\right],
\qquad p=0.20.
\]

The mean-only ablation uses `R=R_mean`. TCSR-MT uses `R=R_mean+R_tail`. The reference scores under `theta_0` are cached offline; no second model is loaded during training, and this is not a KL anchor.

For each candidate, `eta` is calibrated once on a frozen, training-only calibration batch so

\[
\frac{\|\eta\nabla R\|_2}{\|\nabla L_{\rm safe-seq}\|_2}=0.25.
\]

The calibration batch, norm definition, and resulting `eta` are frozen before optimizer creation. `eta` never adapts to behavior, and realized ratios are logged only as diagnostics.

### 5.3 Why the tail term is substantive

An average preference loss can improve easy prompts while leaving a small set of risk categories or prompts unchanged. Exact `max` is unstable with 40 prompts. The worst-quartile mean is a robust approximation to a worst-case constraint and gives the original mean/max intuition a behaviorally meaningful form.

TCSR-MT is promoted over mean-only only if it produces a measurable behavioral increment. Complexity receives no credit from lower training loss or larger sequence margins alone.

## 6. Training-Only Triplet Construction Gate

The existing assets are sufficient to attempt construction, not sufficient to assume success.

1. Use only the 40 `safety_train` prompt clusters and 80 natural-v1 targets. Core100 and heldout content are prohibited.
2. Independently score both existing targets per prompt with the already validated decomposed quality rubric. Select `y+` by a frozen rule based on safety, detail, actionability, and naturalness; use a deterministic target-ID tie break.
3. Construct `yR` by minimal specificity deletion from `y+`: retain safety decision, opening style, paragraph count, and length within `+/-15%`, but remove prompt-specific entities, constraints, and actionable alternatives.
4. Generate `yU` from one frozen unsafe risk-LoRA checkpoint chosen before generation. It must be independently verified as unsafe execution, not merely mixed or irrelevant.
5. Writers, primary reviewers, verification reviewers, and final behavior reviewers must be separate sessions. Reviewers do not see candidate outputs or arm identities.
6. Freeze text, token lengths, prompt/target hashes, category, reviewer decisions, and construction provenance.
7. Require at least 36 valid prompt-disjoint triplets. If fewer remain, close TCSR rather than fill gaps with generic synthetic prompts.
8. Check near-duplicate leakage by prompt source, entity overlap, MinHash, and embedding-neighbor review. Heldout hashes may be checked, but heldout text may not be opened.
9. Reserve the existing 40-row `proxy_dev` split as a training-disjoint mechanism panel. It may receive separately constructed triplets after the training triplets are frozen, but it may not contribute gradients, `eta` calibration, target rewriting, or N2 candidate selection.

## 7. Prospective Experiment Route

### N0: local protocol and data feasibility

- write the triplet schema, construction instructions, blind review rubrics, and hash registry;
- freeze the deterministic `y+` selection rule;
- prepare review packets without model-arm information;
- stop if the 36-triplet gate or reviewer reliability gate fails.

No optimizer and no core100 access are allowed in N0.

### N1: implementation and no-update qualification on the experiment machine

Implement exactly three arms in one common trainer:

| arm | safety exposure | auxiliary objective | role |
| --- | ---: | --- | --- |
| C0R | 0.20 | none | concurrent constant20 control |
| TCSR-M | 0.20 | `R_mean` | simple behavior-space baseline |
| TCSR-MT | 0.20 | `R_mean + R_tail` | full candidate |

Required preflight:

- C0R must be exactly equivalent to the current common trainer when auxiliary code is disabled;
- reference score cache must reproduce direct scoring within `1e-6`;
- triplet losses and LoRA gradients must be finite;
- one wrong-pair unit test must change only the specificity term;
- optimizer must not exist during no-update calibration diagnostics;
- two-step fresh smoke for all three arms;
- peak reserved VRAM must leave at least 10% headroom;
- no candidate generation, core100 content, or heldout content before the implementation freeze is pushed.

### N2: three-arm seed1 tournament

Use the existing frozen training protocol:

```text
Qwen3-4B, NF4 QLoRA, rank 8
256 examples, 64 optimizer steps
microbatch 1, accumulation 4
max length 512, safety_mix 0.20
checkpoints 0/6/16/32/48/64
greedy generation, max_new_tokens 256
```

All arms must share risk order, safety draw order, target order, prompt-to-triplet mapping, optimizer configuration, checkpoints, generation packets, and blind-review protocol.

The frozen C3-T6 gates remain unchanged. In particular:

- C0R step64 unsafe-with-mixed must remain in `[0.20, 0.36]` and utility success in `[0.72, 0.82]`;
- candidate step64 unsafe-with-mixed must be `<=0.35` and increase over C0R `<=0.05`;
- late rebound increase over C0R must be `<=0.05`;
- mixed and invalid must each be `<=0.05`, with at least 60 clean-safe prompts;
- high-specific safe success must improve by at least `0.10`, with paired bootstrap 95% lower bound `>0` and Holm-adjusted exact McNemar `p<0.05` across the two candidates;
- specific-safe success, naturalness, shared-clean detail/actionability, and all utility non-inferiority gates remain exactly as registered for T6;
- no weighted outcome composite is permitted.

Selection rule:

1. reject every hard-gate failure;
2. if only one candidate passes, select it provisionally;
3. if both pass, TCSR-MT must exceed TCSR-M by at least `0.05` on high-specific safe success to justify the tail term;
4. otherwise select the simpler TCSR-M;
5. if neither passes, close this method family without margin, `eta`, CVaR fraction, decoding, or target scans.

### N3: conditional pairing-destruction mechanism test

Only the N2 provisional winner `W` may be rerun with a same-category, length-matched permutation of `y+` inside the specificity-order term. Correct natural-v1 sequence CE, `yR > yU`, the unsafe barrier, data order, total auxiliary gradient scale, compute, and all safety supervision remain unchanged.

The proposed conditional-sequence mechanism passes only if:

- `W` exceeds `W-shuffle` by at least `0.05` high-specific safe success;
- the paired bootstrap 95% lower bound for that difference is positive;
- `W-shuffle` remains within 2 percentage points of `W` on unsafe-with-mixed, so the contrast is not explained by losing safety;
- the frozen `proxy_dev` triplet panel shows the predicted specificity-order change;
- complete behavior gates pass for `W` itself.

If `W` improves behavior but the shuffle intervention does not reduce the gain, the result may be reported as an empirical recipe only. It does not support the claimed prompt-conditioned ordering mechanism and does not authorize a mechanism-centered final paper route.

### N4: replication and sealed confirmation

Only a winner that passes N2 and N3 may proceed:

1. risk seeds 2 and 3 under the unchanged method;
2. seed-stratified and pooled estimates;
3. utility100 on every seed;
4. one heldout opening after all method code, thresholds, and decoding are frozen;
5. a second backbone only after Qwen3-4B three-seed and heldout confirmation.

## 8. Conditional Backup: TRIM-P0

TRIM is not authorized for training. It receives one inference-only causal preflight on the existing C0-R checkpoint stored on the experiment machine.

### 8.1 Preflight design

- construct at least 16 same-category counterfactual prompt pairs across at least four categories from `proxy_dev`, not from core100 or heldout;
- annotate the refusal-to-redirect transition in natural-v1 targets before running the model;
- use one fixed layer `ceil(2L/3)` and one two-token transition window; no layer or token sweep;
- compare no patch, exact no-op patch, correct donor, wrong donor, and prompt-last-token placebo;
- evaluate both teacher-forced suffix NLL and controlled free suffix generation;
- score generated suffixes for donor-specific detail attribution, safe actionability, naturalness, and unsafe content with reviewers blind to condition.

### 8.2 Gate

TRIM training receives a separate prospective plan only if all conditions hold:

- no-op patch maximum logit error `<=1e-5`;
- valid hook and positional alignment for at least 99% of runs;
- correct-versus-wrong donor changes generated detail attribution by at least 20 percentage points with bootstrap lower bound `>0`;
- the transition-position effect is at least twice the prompt-last placebo effect;
- unsafe-with-mixed does not increase by more than 3 percentage points;
- the effect is visible in generated behavior, not only in NLL or internal activations.

Failure closes TRIM before training. Passing this gate does not authorize combining TRIM with TCSR; they remain competing mechanisms until each has independent evidence.

## 9. Failure Interpretations Frozen in Advance

- If C0R itself does not reproduce, the tournament is invalid; candidate outcomes are not interpreted.
- If sequence margins improve but blind generated behavior does not, TCSR joins the prior proxy failures and is closed.
- If TCSR-MT does not beat TCSR-M by the registered increment, the tail component is rejected even if its training loss is lower.
- If TCSR beats C0R but not the pairing-shuffle arm, the gain cannot be attributed to prompt-conditioned sequence ordering.
- If TRIM changes suffix NLL or activation probes without changing generated detail attribution, the proposed residual mediator is rejected.
- A seed1 success is candidate-selection evidence only. It is not cross-seed, heldout, or cross-backbone evidence.

## 10. Why the Other Complex Methods Are Deferred

Complexity is not being discarded. It is being ordered by evidential dependency.

- BASC requires a behavior-proximal quality gradient that is already known to work; TCSR tests that prerequisite. If TCSR passes, a later fixed-versus-feedback deployment study becomes meaningful.
- DFGU addresses persistence more directly than quality. It is useful only if a future result shows residual safety rebound after the quality mechanism is solved.
- BCRD requires a calibrated online behavior judge and a much larger on-policy budget. It should not be implemented until offline sequence constraints and controller calibration have been validated.
- TRIM has the strongest causal novelty and therefore receives the strongest cheap falsification test before training.

This sequencing avoids building an elaborate controller around an ineffective behavioral signal or adding an internal mechanism whose causal channel is absent.

## 11. Compute and Ownership Estimate

The previous three-arm T3 training used approximately 7-14 optimizer minutes per arm and 1.37 GPU hours for all 12 generation cells. TCSR adds sequence-scoring forwards on safety draws, so a conservative seed1 estimate is:

| phase | owner | estimate |
| --- | --- | --- |
| triplet construction and packet freeze | local plus independent agents | 0 GPU hours; review-bound |
| no-update, calibration, and smoke | experiment machine | 1-2 GPU hours including debugging allowance |
| three 64-step training arms | experiment machine | about 0.5-1.5 GPU hours total |
| 12 generation cells | experiment machine | about 1.5-2 GPU hours |
| blind A/B/C review and statistics | isolated agents plus local analysis | review-bound, not GPU-bound |
| conditional shuffle arm | experiment machine | about 1 additional GPU hour including generation |
| TRIM-P0 | experiment machine | 1-3 GPU hours after hook validation |

These are planning ranges, not guarantees. Full training must stop if TCSR exceeds 2.5 times C3a wall time per arm or cannot retain 10% VRAM headroom.

## 12. Decision

```text
primary candidate:       TCSR-MT, with TCSR-M as required incremental ablation
primary next action:     N0 triplet protocol and independent data-feasibility freeze
conditional backup:      TRIM-P0 inference-only causal preflight
not authorized now:      BCRD, DFGU training, BASC, method stacking, seeds 2/3,
                         heldout, second backbone, SpectraSafe re-entry
```

The final selection is deliberately asymmetric. TCSR-MT is the best next method experiment because it follows the strongest existing positive evidence: complete sequence supervision changes behavior. TRIM is the most original mechanism candidate, but its central causal channel can and should be tested before any new optimizer run.
