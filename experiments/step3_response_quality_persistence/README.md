# Step 3: Response Quality Under Persistent Safety Supervision

## 0. Status

Step 3 is open.

Current phase: **Step3C-T5H H3 all A/B reviews frozen; disagreement-only C pending**.

Step3A selected Route A2: output concentration is not matched by target concentration. Step3B selected Route B3: low-temperature sampling did not sufficiently reduce conditional mechanical concentration, while the higher-temperature condition failed the unsafe-increase boundary. No new model training is authorized by these results. The heldout test remains sealed.

The first Step3C Stage 0 stopped at its repeated response-only mechanical-label reliability gate and established that C2 cannot reach its 36-triplet requirement. Step3C-R then stopped after two prompt-aware pair-preference reliability failures (`0.825` and `0.800` raw agreement against a frozen `0.85` gate). C2 and C-R0 are closed.

Step3C-MF subsequently passed both prerequisites for a new C3 behavioral test. T2R then completed one state-matched eta recalibration and restored smoke qualification. C0-R, C3a, and C3b each completed a matched 64-step seed1 run, and T4 froze 1,200 generations plus unchanged safety, quality, and utility blind packets. T5R Track A stopped at `T5R1_reviewer_access_stop` with zero accepted labels because no approved independent safety-review environment was available. Track B completed all 12 frozen representation probes and exactly reproduced grouped-gradient labels 0/16/32, but the bounded checkpoint48 replay did not reproduce label64 and selected `M1_reconstruction_integrity_failure`. P6A reconciled all 21 legacy hashes without changing P6 or M1. H1/H1Q/H2 completed Safety reviewer qualification and freeze. H3Q then froze four fresh no-context sessions; all four subsequently returned complete 330-row Quality A/B and Utility A/B files with valid provenance. H3 selected `H3A1_all_ab_frozen_authorize_disagreement_c`, covering six distinct reviewer sessions and 2,580 accepted rows across all reviewer files. A/B labels have not been compared, reliability and behavior endpoints remain unavailable, heldout remains sealed, and no new training is authorized.

## 1. Research Question

> How can safety persistence be maintained while increasing prompt-relevant safe-response coverage and reducing mechanically reusable response structure, without exceeding the observed utility boundary?

Step 3 does not begin by adding another regularizer or scanning more replay weights. It first determines whether the response-quality construct is measured reliably and whether the concentration observed in Step 2 is inherited from the safety targets, induced by optimization, or partly an artifact of the original semantic-family rubric.

## 2. Step 2 Entry Evidence

Step 3 begins from four established results.

1. `constant20` reduced risk-seed1 step64 unsafe-with-mixed from `0.56` to `0.28` relative to `decay10`, with paired improvement `0.28`, bootstrap 95% CI `0.20-0.37`, and McNemar `p=7.45e-9`.
2. `constant20` step64 prompt-relevant redirect was `0.72`, below the frozen `0.80` gate.
3. The adjudicated mechanical-template share was `0.72`, but the free-form normalized semantic-skeleton label had zero exact reviewer agreement and required complete adjudication.
4. Utility task success and unnecessary refusal passed only at the `0.05` boundary; absolute material degradation was `0.22`, while vanilla itself was `0.24`.

These results retain `constant20` as a strong persistence control, not as a final method.

## 3. Necessary Measurement Corrections

### 3.1 Do not optimize the old semantic-cluster number

The Step2J semantic rubric allowed reviewers to invent normalized skeleton names and family IDs. Exact reviewer agreement on that field was `0.00`. The final `0.72` family share is evidence of concentration after adjudication, but it is not sufficiently reproducible to serve as a primary Step 3 endpoint.

Step3A initially used:

- `mechanical_template` as the primary concentration construct;
- a fixed response-policy codebook as a secondary categorical construct;
- exact duplication and lexical diversity only as descriptive diagnostics;
- embedding or API similarity only as sensitivity analysis, never as a replacement for behavior labels.

The response-only mechanical construct subsequently failed repeated Stage0 reliability, and holistic prompt-aware pair preference failed both C-R0 sessions. The active Step3C-MF study therefore validates independent ordinal `detail_retention` and `safe_actionability` scores on a prompt-disjoint development/validation design. No old concentration or pair-preference score is an active promotion endpoint.

### 3.2 Distinguish target concentration from optimization collapse

The natural v1 manifest contains 80 targets for 40 prompts:

- 40 begin with `I can't`;
- 40 begin with `I won't`;
- 72 contain `I can help`;
- the dominant target structure is refusal plus a safe alternative.

The output concentration may therefore reflect supervised structure transfer. Step 3 must not call it spontaneous model collapse until a structure-balanced target intervention is compared under matched exposure and schedule.

### 3.3 Preserve the leakage boundary

The 40 safety-training prompt IDs have zero overlap with the existing 100-prompt core safety evaluation. The 120-row `heldout_test` remains unopened.

Step 3 keeps three roles separate:

- `safety_train`: target construction and replay only;
- core100 development evaluation: candidate selection and mechanism analysis;
- `heldout_test`: one final confirmation after all seed and utility gates pass.

### 3.4 Replace an impossible absolute utility gate

The Step 2 absolute material-degradation gate was `<=0.15`, but vanilla scored `0.24`. It cannot identify method-induced degradation in this setting.

Step 3 continues to report the absolute rate, but promotion uses paired change relative to vanilla and non-inferiority relative to `constant20`. This change is prospective and must be frozen before any Step 3 training.

## 4. Working Hypotheses

### H3.1 Target-structure bottleneck

The narrow response policy encoded by natural v1 is a major source of the observed mechanical-template concentration.

### H3.2 Persistence constraint

Reducing or decaying late safety replay will recover surface diversity by losing safety, not by solving response quality. A valid candidate must preserve the constant20 persistence effect.

### H3.3 Structure-balanced replay

Prompt-appropriate, structure-balanced safety targets can improve redirect coverage and reduce mechanical reuse at matched safety dose, without materially worsening unsafe behavior or utility.

These are hypotheses, not method claims. H3.3 is authorized for training only if Step3A validates the measurement and target-structure premise.

Step3A result: H3.1 did not pass its target-concentration gate (`0.2625 < 0.60`), so H3.3 and natural-v2 construction are not authorized. The active hypothesis is that clean-safe output concentration arises during optimization or greedy decoding.

## 5. Stage Architecture

### Step3A: measurement and attribution audit

Use existing targets and outputs only. Freeze a reproducible response-policy codebook, validate reviewer agreement, quantify target structure, and separate train/core100/heldout roles.

### Step3B: decoding attribution

Under Route A2, reuse the frozen constant20 step64 checkpoint and compare greedy decoding with two preregistered sampling conditions. Measure mechanical-template share conditional on clean-safe behavior. This is an inference-only experiment and does not alter the checkpoint.

Step3B result: Route B3. `sample_t03_p09` kept unsafe increase within the `<=0.05` boundary but reduced `P(mechanical | clean-safe)` only from `1.000` to `0.942`. `sample_t07_p09` reduced it to `0.911` but increased unsafe by `0.164`, exceeding the invalid-route threshold.

### Step3C: route-dependent controlled intervention

The original Stage 0 stopped before triplet freezing or model diagnostics because response-only `mechanical_template` reliability failed after the permitted complete rereview. The current C2 construction is closed because only 29 constant20 generations were clean-safe and only 13 were unanimously safe-mechanical, below the required 36.

Step3C-R replaced the primary endpoint with prompt-aware pairwise specificity, but its holistic preference component failed both complete reliability sessions. The route closed before C3 diagnostics or training.

Step3C-MF separated measurement from method feasibility. Step3C-M validated ordinal detail retention and safe actionability on 20 development and 20 development-untouched validation prompt clusters. Step3C-F then showed that C3a multi-positive global binding and C3b hierarchical hard-negative binding both produce finite, calibrated LoRA gradients under the registered memory rule. The joint route passed and stopped before training.

Step3C-T's first smoke exposed a numerical calibration-transfer failure. T2R repaired it prospectively with one formula-based fresh-step0 recalibration over positions 8-79 and a complete fresh three-arm smoke. The recovery passed and authorized T3.

T3 then completed matched C0-R, C3a, and C3b 64-step runs with all integrity gates passing. Fixed eta did not maintain a fixed binding-gradient budget: C3a's realized weighted ratio moved `0.255 -> 0.551 -> 0.360 -> 0.417`, and C3b's moved `0.194 -> 0.508 -> 0.342 -> 0.479`. Global binding-safety cosine moved from about `+0.57` to `-0.11`. C3a and C3b also triggered the preregistered redundancy warning. These are mechanism signals, not behavior outcomes.

T4 completed all 12 generation cells and froze 630-row safety, 330-row quality, and 330-row utility packets. T5 accepted zero labels because Safety Reviewer A encountered a platform gateway before output. T5R now preserves the unchanged packets, restores lawful independent review, and fills the parent plan's missing retrieval, cosine-gap, effective-rank, top-singular-share, and attention/MLP cosine diagnostics before final interpretation.

### Step3D: seed1 persistence-quality factorial

Only a Step3C-T quality winner may request a separate seed1 factorial comparing C0, C4 gradient-budget control, the quality winner `Q*`, and the combined C5 method. C4 and C5 receive no automatic authorization from a tournament pass, and their main effects and interaction must be identified separately.

### Step3E: replication and sealed confirmation

Only a frozen seed1 final candidate that passes the factorial gates may advance to risk seeds 2 and 3. The heldout test opens once, after the method and thresholds are frozen and cross-seed evidence passes. A second backbone follows the Qwen3-4B confirmation rather than substituting for it. SpectraSafe does not re-enter without a new, preregistered reason and a clean incremental ablation.

## 6. Completed Evidence and Active Gates

Completed evidence now includes:

- validated decomposed quality measurement;
- T2R state-matched eta calibration and fresh smoke qualification;
- three matched 64-step seed1 runs with finite updates and frozen manifests;
- 12 T4 generation cells, 1,200 non-empty outputs, and byte-verified blind packets;
- zero accepted T5 labels and no hidden outcome-key join;
- heldout continuously sealed.

The review-recovery contract remains frozen in `configs/step3ct_t5r_review_recovery_registry.json`. Track A is stopped at `T5R1`, Track B is stopped at `M1`, and T6 is unauthorized. Any future resumption still requires complete fresh A/B reviews, disagreement-only C review, reliability before arm-key join, and unchanged packet/rubric hashes. The unchanged behavior gates are:

| domain | gate |
| --- | --- |
| concurrent control | reproduce historical constant20 safety and utility within frozen tolerances |
| safety | step64 unsafe-with-mixed `<=0.35` and increase versus C0-R `<=0.05` |
| minimum quality | `specific_safe_success` non-inferior to C0-R |
| primary quality | `high_specific_safe_success` improvement `>=0.10`, paired bootstrap lower bound `>0`, and Holm-adjusted McNemar `p<0.05` |
| utility | no more than registered loss versus both vanilla and concurrent C0-R |
| selection | C3b must beat C3a by at least `0.05` on primary behavior or C3a is preferred |

Mechanical-template and holistic pair-preference labels remain inactive.

## 7. Non-Negotiable Controls

- Qwen3-4B, risk family, and seed1 remain the pilot setting.
- The v1 constant20 evidence remains the required baseline.
- Training prompt IDs and core100 evaluation prompt IDs remain disjoint.
- Target count, safety draws, exposure distribution, checkpoint schedule, decoding, and evaluation packets must be matched.
- Target text, hashes, reviewer decisions, and codebook version must be frozen before training.
- Main-session agents that have seen arm identities may prepare packets and compute statistics but may not provide blind outcome labels.
- No heldout access, seed expansion, or method stacking occurs before the corresponding gate.

## 8. Stop Rules

Stop Step 3 or redirect it if any of the following occurs:

- any unchanged T4 packet or rubric hash fails;
- lawful independent reviewers cannot complete the unchanged safety packet;
- any accepted A/B/C file is incomplete, invalid, or unprovenanced;
- the frozen review-reliability gates fail;
- the concurrent C0-R control fails its reproducibility gate;
- the seed1 candidate improves apparent diversity by increasing unsafe behavior beyond the safety boundary;
- utility loss exceeds the paired non-inferiority boundary;
- mechanism reconstruction cannot reproduce the original global diagnostics, in which case approximate grouped cosines are not reported;
- gains depend on opening the heldout set early.

## 9. Current Authorization

Authorized now:

- preserve the completed `P6A1_eol_reconciled_content_unchanged` package without modifying P6 or M1;
- after the H3 package is pushed, construct disagreement-only C packets without A/B labels, reasons, votes, arms, checkpoints, or duplicate links;
- freeze C packet hashes and launch fresh domain-specific C reviewers before computing reliability;
- compute reliability before hidden arm-key join;
- execute T6 only after route `T5R4_complete_labels_authorize_T6`.

Not authorized now:

- modification, sanitization, splitting, or regeneration of the T4 packets;
- modification of P6, another label64 replay, relaxed reconstruction tolerance, or approximate grouped label64 reporting;
- reuse of failed T5 sessions or partial outputs;
- main-session or heuristic outcome labels;
- any fourth arm, fresh candidate training, fresh generation, or behavior-based eta/temperature/queue/target revision;
- C4, C5, adaptive replay, or other method stacking;
- behavior-based hyperparameter scans;
- seeds 2 and 3;
- heldout evaluation;
- SpectraSafe reintroduction.

The stopped Stage0 and C-R0 routes, completed Step3C-MF/T2R/T3/T4/P6A/H1/H1Q/H2/H3Q/H3 packages, T5/T5R reviewer-access incidents, and Track B `M1` mechanism route remain recorded without overwrite. H3 is an all-A/B completion milestone, not a method result: no agreement, reliability, behavior comparison, or candidate selection has yet been computed. The active T5H/T5R plans now authorize disagreement-only C preparation after H3 is pushed. A later valid T6 winner is still required before any C4/Q*/C5 seed1 factorial plan.
