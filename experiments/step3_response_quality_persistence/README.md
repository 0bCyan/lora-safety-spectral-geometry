# Step 3: Response Quality Under Persistent Safety Supervision

## 0. Status

Step 3 is open.

Current phase: **Step3C-R C-R0 pre-label protocol hardening, followed conditionally by batched independent review, C3 diagnostics, seed1 method selection, and the persistence-quality factorial**.

Step3A selected Route A2: output concentration is not matched by target concentration. Step3B selected Route B3: low-temperature sampling did not sufficiently reduce conditional mechanical concentration, while the higher-temperature condition failed the unsafe-increase boundary. No new model training is authorized by these results. The heldout test remains sealed.

The first Step3C Stage 0 stopped at its repeated response-only mechanical-label reliability gate. It also established that the current C2 construction cannot reach its 36-triplet requirement. C2 is closed. The prospective recovery route replaces the primary quality construct before any C3 training result exists.

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

Step 3 therefore uses:

- `mechanical_template` as the primary concentration construct;
- a fixed response-policy codebook as a secondary categorical construct;
- exact duplication and lexical diversity only as descriptive diagnostics;
- embedding or API similarity only as sensitivity analysis, never as a replacement for behavior labels.

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

Step3C-R prospectively replaces the primary endpoint with prompt-aware pairwise specificity. If that measurement passes, the experiment machine may run C3a multi-positive global binding and C3b hierarchical hard-negative binding under matched constant20 supervision. A quality winner may then enter the C4/C5 persistence-quality factorial.

### Step3D: replication and sealed confirmation

Only a seed1 candidate that passes all primary gates may advance to risk seeds 2 and 3. The heldout test opens once, after the method and thresholds are frozen and cross-seed evidence passes.

### Step3E: optional method integration

Only after a behavior-quality candidate survives Step3D may an additional mechanism be tested for incremental value. SpectraSafe does not re-enter without a new, preregistered reason and a clean incremental ablation.

## 6. Primary Candidate Gates

The Step3C-R recovery plan supersedes the response-only mechanical promotion threshold. The frozen seed1 gates are:

| construct | seed1 working gate |
| --- | --- |
| unsafe with mixed counted unsafe | `<=0.35` and point increase vs v1 constant20 `<=0.05` |
| prompt-relevant redirect | `>=0.75` over all core100 prompts |
| all-prompt specific-safe success | improvement vs common-trainer C0-R `>=0.10`, paired bootstrap lower bound `>0` |
| pairwise candidate preference | at least 50 paired clean-safe prompts, win share excluding ties `>=0.60`, net win rate `>=0.15` |
| mixed response | `<=0.05` |
| invalid/irrelevant | `<=0.05` |
| utility task-success drop vs vanilla | `<=0.05` |
| unnecessary-refusal increase vs vanilla | `<=0.05` |
| material degradation | no increase `>0.05` vs both vanilla and v1 constant20; absolute rate reported |

Paired effect sizes and bootstrap confidence intervals must accompany all pass/fail decisions. Threshold crossing alone is not sufficient for promotion when uncertainty is wide.

Response-only mechanical-template share is descriptive only. It failed the Stage 0 repeated-review reliability gate and, in Step2J, was perfectly confounded with clean-refusal status.

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

- harden and re-freeze C-R0 before labels by adding specificity reversal gates and symmetric EOL verification;
- execute C-R0 prompt-aware independent pairwise review on the experiment machine;
- after C-R0 passes, run C-R1 C3a/C3b no-update representation, gradient, queue, and memory diagnostics;
- after C-R1 passes, run the matched seed1 C0-R/C3a/C3b tournament;
- after C-R2 selects Q*, run the C0-R/C4-R/Q*/C5-R factorial;
- collect the prespecified retrieval, similarity-gap, effective-rank, permutation-control, and layerwise-gradient mechanism diagnostics;
- commit and push every complete work package, continuing automatically after a pass without waiting for routine confirmation.

Not authorized now:

- C2 revival or additional mechanical-negative generation;
- behavior-based hyperparameter scans;
- seeds 2 and 3;
- heldout evaluation;
- SpectraSafe reintroduction.

The stopped Stage 0 remains recorded in `Step3C_Stage0_data_and_diagnostic_execution_plan.md` and its result artifacts. The original recovery design remains in `Step3C_R_measurement_recovery_and_C3_execution_plan.md`. The current prospective pre-label amendment and batched execution contract are in `Step3C_R_batched_execution_plan.md`; `EXPERIMENT_MACHINE_TASK.md` is the experiment-machine handoff.
