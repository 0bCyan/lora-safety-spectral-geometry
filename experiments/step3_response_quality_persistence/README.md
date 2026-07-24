# Step 3: Response Quality Under Persistent Safety Supervision

## 0. Status

Step 3 is open.

Current phase: **Step3C-T2R step0 calibration recovery and smoke requalification**.

Step3A selected Route A2: output concentration is not matched by target concentration. Step3B selected Route B3: low-temperature sampling did not sufficiently reduce conditional mechanical concentration, while the higher-temperature condition failed the unsafe-increase boundary. No new model training is authorized by these results. The heldout test remains sealed.

The first Step3C Stage 0 stopped at its repeated response-only mechanical-label reliability gate and established that C2 cannot reach its 36-triplet requirement. Step3C-R then stopped after two prompt-aware pair-preference reliability failures (`0.825` and `0.800` raw agreement against a frozen `0.85` gate). C2 and C-R0 are closed.

Step3C-MF subsequently passed both prerequisites for a new C3 behavioral test. The decomposed quality measurement passed its prompt-disjoint one-shot validation, and both C3a/C3b passed no-update representation, gradient, memory, and calibration feasibility. Step3C-T then froze the common trainer and completed its first three-arm smoke, but stopped before T3 because the step64-derived eta values produced fresh-step0 binding/safety ratios of only `0.12245` and `0.07851`. No candidate has been trained to step64 or generated. The active recovery plan is `Step3C_T2R_step0_calibration_and_smoke_requalification_plan.md`; heldout remains sealed.

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

Step3C-T is the planned behavioral tournament. Its T1 common trainer passed implementation qualification, but its first T2 smoke stopped at the frozen gradient-ratio gate. All other smoke gates passed: finite losses/gradients, parameter and projection boundaries, queue integrity, cross-arm draws, exact resume reproduction, and more than 54% VRAM headroom. Because no behavior was generated, this is a numerical calibration-transport failure rather than a C3 efficacy result.

Step3C-T2R now permits one formula-based recalibration at the actual fresh-step0 model state. Eta fitting excludes the first eight target positions, uses 18 four-microstep gradient windows over positions 8-79, and targets the unchanged `0.25` binding/safety ratio. Positions 0-7 were already executed in the failed smoke, so they are not an untouched holdout; they are an eta-fit-excluded operational transfer check under a newly frozen all-microstep estimator. Only a complete fresh requalification pass restores T3 authorization.

### Step3D: seed1 persistence-quality factorial

Only a Step3C-T quality winner may request a separate seed1 factorial comparing C0, C4 gradient-budget control, the quality winner `Q*`, and the combined C5 method. C4 and C5 receive no automatic authorization from a tournament pass, and their main effects and interaction must be identified separately.

### Step3E: replication and sealed confirmation

Only a frozen seed1 final candidate that passes the factorial gates may advance to risk seeds 2 and 3. The heldout test opens once, after the method and thresholds are frozen and cross-seed evidence passes. A second backbone follows the Qwen3-4B confirmation rather than substituting for it. SpectraSafe does not re-enter without a new, preregistered reason and a clean incremental ablation.

## 6. Completed Preconditions and Active Recovery Gates

Step3C-MF completed with:

- validation detail weighted agreement `0.9688`, Gwet AC2 `0.9564`;
- validation actionability weighted agreement `0.9531`, Gwet AC2 `0.9166`;
- derived-quality raw agreement `0.9844`, Gwet AC1 `0.9809`;
- C3a and C3b finite/non-zero LoRA gradients at a calibrated binding/safety ratio of `0.25`;
- no optimizer step, candidate generation, core100 selection, or heldout access.

The original behavior gates remain frozen in `configs/step3ct_seed1_training_registry.json`. The current numerical recovery gates are frozen in `configs/step3ct_t2r_calibration_registry.json`:

| recovery domain | gate |
| --- | --- |
| metadata | hard-negative availability reproduces `58/80`, `4/8`, `54/72`, and `17/18` fit windows |
| eta fit | one formula over 18 optimizer-step windows from positions 8-79, target ratio `0.25` |
| calibration stability | eta in `[0.01,0.30]`, weighted median `[0.20,0.30]`, p90 `<=0.75` |
| eta-fit-excluded smoke | each four-microstep ratio `[0.10,0.50]`, aggregate eight-position ratio `[0.15,0.35]` |
| integrity | all original finite, parameter, queue, draw, resume, and VRAM gates still pass |

After recovery route `RR2`, the unchanged behavior tournament gates are:

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

- the primary response-quality labels cannot reach acceptable reviewer reliability after one rubric revision;
- v1 targets are not structurally concentrated under the fixed codebook;
- a v2 target set cannot pass independent relevance, naturalness, leakage, and structure-balance review;
- the seed1 candidate improves apparent diversity by increasing unsafe behavior beyond the safety boundary;
- utility loss exceeds the paired non-inferiority boundary;
- gains exist only on training prompts or depend on opening the heldout set early.

## 9. Current Authorization

Authorized now:

- implement and freeze the Step3C-T2R no-update calibration and all-microstep smoke diagnostics;
- perform one formula-based fresh-step0 eta calibration using positions 8-79 only;
- rerun fresh two-step smoke tests for exactly C0-R, C3a, and C3b under the new immutable config;
- resume T3 only after a committed and pushed `RR2_three_arm_smoke_requalified` route.

Not authorized now:

- C2 revival or additional mechanical-negative generation;
- a third C-R0 pairwise review or a lower old reliability threshold;
- direct continuation from the failed T2 smoke;
- use of eta-fit-excluded positions 0-7 to fit eta or a second recalibration after failure;
- any fourth arm or behavior-based eta/temperature/queue revision;
- trainable projection-head variants;
- C4, C5, adaptive replay, or other method stacking;
- behavior-based hyperparameter scans;
- seeds 2 and 3;
- heldout evaluation;
- SpectraSafe reintroduction.

The stopped Stage0 and C-R0 routes, completed Step3C-MF route, and failed original T2 smoke remain recorded without overwrite. The active prospective plan is `Step3C_T2R_step0_calibration_and_smoke_requalification_plan.md`; `EXPERIMENT_MACHINE_TASK.md` is the experiment-machine handoff. A successful recovery only resumes the original tournament, and a tournament winner still requires a separate C4/Q*/C5 seed1 factorial plan.
