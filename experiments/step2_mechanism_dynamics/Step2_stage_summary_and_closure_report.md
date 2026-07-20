# Step 2 Stage Summary and Closure Report

## 0. Document Status

Status: final and approved for Step 2 closure. No merge or Step 3 authorization is implied by this document.

Recommended stage decision:

> Step 2 is ready for closure as a successful mechanism-diagnosis and boundary-setting stage. It identifies a reproducible safety-persistence mechanism and a strong causal control, but it does not produce a complete publishable method.

The final Step 2 route is **Q2**:

> Maintaining 20% natural safety replay has a real seed1 safety-persistence effect, but the candidate fails response-coverage and semantic-concentration gates. Constant20 is therefore retained as a strong causal control and closed as a complete-method candidate.

No additional Step 2 training is authorized by the current evidence.

## 1. Scope and Evidence Boundary

Step 2 studied training-time mechanisms for safety drift under risk LoRA fine-tuning. The main experimental setting is:

- hardware: one RTX 4070 Ti;
- backbone: Qwen3-4B;
- primary family: risk;
- exploratory and final mechanism tests: primarily seed1;
- early transition replication: three risk seeds;
- safety evaluation: local safety30/full100 plus later independent blind adjudication;
- utility evaluation: utility30/utility100;
- heldout test: unopened.

All causal and negative conclusions in this report are bounded to this setting. Step 2 does not establish cross-seed, cross-backbone, cross-dataset, or heldout generalization for a final method.

## 2. Relation to Step 1

Step 1 established a robust behavioral failure and several diagnostic geometric signals:

- all three risk seeds reached `unsafe=1.00` and `refusal=0.00`;
- global spectral averages weakly separated training families;
- risk left/output subspace seed overlap reached `0.6259`;
- the best parameter-space post-hoc intervention still had `unsafe=0.9833`;
- activation steering also failed to restore safe behavior.

The Step 1 conclusion was therefore:

> Geometry, subspaces, and activations are useful diagnostics, but the tested post-hoc interventions do not establish a reliable causal control mechanism.

Step 2 followed directly from this boundary. It moved the intervention into training and asked which training signal can actually preserve safe behavior over time.

## 3. Step 2 Evidence Chain

### 3.1 Step2A: locate the behavioral transition

The risk trajectory was measured across training rather than only at the final checkpoint.

| checkpoint | risk full100 unsafe |
| --- | ---: |
| step6 | 0.7367 |
| step16 | 0.9967 |
| step64 | 1.0000 |

All three risk seeds reached `unsafe=1.00` by step16. In contrast, normal seed1 at step16 had `unsafe=0.3333` and `refusal=0.6333`.

This identifies a reproducible, family-specific early unsafe transition between steps 6 and 16. It also establishes that endpoint-only evaluation hides the transition dynamics needed for mechanism testing.

### 3.2 Step2B: test simple training-time geometry controls

LoRA norm, spectral-mean, and spectral-top penalties were applied during training. The tested arms remained near `unsafe=0.99` at step16.

Conclusion:

> Simple geometric regularization can alter or constrain parameter geometry without producing meaningful behavioral safety in this setting.

### 3.3 Step2C: separate scalar behavior margins from sequence supervision

Margin-only objectives drove the target scalar margin to approximately `12-13`, yet unsafe behavior remained near `0.99`.

In contrast, 20% sequence-anchor supervision reduced step16 unsafe to:

| seed | step16 unsafe |
| --- | ---: |
| 1 | 0.27 |
| 2 | 0.23 |
| 3 | 0.19 |

This is the first clear positive behavioral intervention in Step 2.

Conclusion:

> Direct sequence-level safety supervision is behaviorally active. A scalar token or margin proxy is not sufficient merely because its optimized value moves in the intended direction.

### 3.4 Step2D-E: characterize persistence and degradation

The early sequence-anchor gain did not automatically persist. With insufficient late safety supervision, unsafe behavior rebounded to `1.00` by step64.

The decay-to-5 schedule produced a three-seed mean step64 unsafe rate of `0.6267` and degradation of `0.29`. Stronger repetitive schedules also produced repetitive refusal-style degradation.

Conclusion:

> Safety requires continued late training pressure, but persistent supervision with narrow or repetitive targets can replace unsafe behavior with low-quality template behavior.

This converted the problem from simple safety suppression into a joint persistence-and-response-quality problem.

### 3.5 Step2F and Step2F-R: faithful evaluation of SpectraSafe

Step2F tested direct delta-spectral mean-only, max-only, and mean-plus-max controls. Step2F-R repeated the test with the faithful v0.0.8 factor proxy `||B||_2 ||A||_2`.

Across direct and faithful implementations, all primary arms remained at or above `0.99` unsafe at step16 or step64, while utility stayed acceptable. Low, medium, and high strength checks also failed to produce behavioral recovery.

Conclusion:

> Spectral compression is a bounded negative in the Qwen3-4B risk-LoRA setting. Both the direct delta form and the faithful factor-product form change auditable geometry without controlling unsafe behavior.

This does not establish universal failure across other backbones, datasets, or behavior-supervised combinations.

### 3.6 Step2G: audit behavior proxies before designing another objective

Candidate proxy AUROCs were:

| proxy | AUROC |
| --- | ---: |
| refusal-action | 0.556 |
| P1 | 0.595 |
| P2 | 0.729 |

No proxy passed the frozen leave-one-family-out/generalization gate.

Conclusion:

> The tested local token and margin proxies are not sufficiently valid to justify another proxy-optimization method.

### 3.7 Step2H: test whether the spectral factor adds to sequence supervision

| arm | step16 unsafe | step32 unsafe | step64 unsafe |
| --- | ---: | ---: | ---: |
| factor-only | 1.00 | 0.98 | 1.00 |
| sequence-only | 0.17 | 0.29 | 0.75 |
| factor + sequence | 0.18 | 0.30 | 0.76 |

The sequence term explains the observed safety movement. Adding the factor proxy provides no measurable incremental safety benefit.

Conclusion:

> Under the tested design, the dual spectral term is neither independently behaviorally effective nor additive to sequence supervision.

### 3.8 Step2I and Step2I-R: correct replay confounds and audit target quality

The initial diverse-replay experiment was confounded by target exposure, target length, and sampler behavior. A corrected micro-step sampler then used:

- 80 targets;
- 256 safety draws;
- three or four draws per target;
- matched exposure across arms.

The programmatic `diverse_matched_10` arm produced a provisional local step64 result of `unsafe=0.55`, utility drop `0.04`, and largest exact cluster `0.03`.

Independent target review nevertheless rejected the generator:

| quality metric | result |
| --- | ---: |
| prompt-relevant redirect | 28.75% |
| grammatical naturalness | 0% |
| keyword stuffing defect | 100% |
| mechanical filler defect | 100% |

The apparent exact diversity was also misleading. The largest semantic family at step64 was `48%`, or `32%` under a conservative split.

Conclusion:

> The corrected sampler and the persistent-replay signal are useful, but the programmatic target generator is invalid. Surface variation and low exact duplication do not establish semantic diversity.

The rejected target generator was not expanded to additional seeds.

### 3.9 Natural decay-to-10 control

A separately versioned natural-target set was introduced and reviewed. Under the decay-to-10 schedule, local unsafe rates were:

| checkpoint | unsafe |
| --- | ---: |
| step16 | 0.67 |
| step32 | 0.38 |
| step64 | 0.76 |

Utility task success was `0.80`, compared with `0.81` for vanilla.

Conclusion:

> Natural replay avoids the invalid programmatic-target shortcut, but a 10% late floor still produces a transient safety gain followed by substantial late rebound.

### 3.10 Step2J-A: test the gradient-magnitude mechanism

The natural-target gradient audit compared safety and risk gradient contributions at matched checkpoints.

| checkpoint | schedule weight | weighted safety/risk contribution | equal-contribution weight |
| --- | ---: | ---: | ---: |
| step16 | 0.20 | 0.468-0.496 | 0.347-0.355 |
| step32 | 0.1667 | 0.387-0.398 | 0.342-0.350 |
| step64 | 0.10 | 0.233-0.244 | 0.320-0.333 |

At step64, bootstrap confidence intervals were `0.193-0.270` for block 1 and `0.210-0.278` for block 2.

The unweighted safety gradient remained strong, but the schedule reduced its effective late contribution. Mean cosine was only mildly negative, and one strict stable-competition gate did not pass.

Conclusion:

> The late rebound is best explained by insufficient weighted safety-gradient magnitude under decay-to-10. The evidence does not support projection or a single conflict direction as the primary mechanism.

### 3.11 Step2J-B1: causal constant20 test and blind adjudication

Constant20 changed only the persistence of natural safety replay relative to the decay10 control. Independent blind review then adjudicated safety behavior, semantic structure, and utility.

#### Safety behavior

| arm/checkpoint | unsafe incl. mixed | clean refusal | prompt-relevant redirect | mixed | generic-only |
| --- | ---: | ---: | ---: | ---: | ---: |
| decay10 step32 | 0.13 | 0.87 | 0.88 | 0.01 | 0.00 |
| constant20 step32 | 0.10 | 0.90 | 0.90 | 0.00 | 0.00 |
| decay10 step64 | 0.56 | 0.44 | 0.46 | 0.02 | 0.00 |
| constant20 step64 | 0.28 | 0.72 | 0.72 | 0.00 | 0.00 |

For the paired step64 comparison:

- 28 prompts changed from unsafe under decay10 to safe under constant20;
- zero prompts changed in the reverse direction;
- absolute improvement was `0.28`;
- bootstrap 95% CI was `0.20-0.37`;
- McNemar `p=7.45e-9`;
- step32-to-step64 rebound fell from `0.43` to `0.18`.

The strongest supported causal statement is:

> In Qwen3-4B risk seed1, maintaining 20% natural safety replay causally reduces late unsafe rebound relative to decay-to-10.

#### Semantic response quality

For constant20 at step64:

- largest exact cluster: `0.01`;
- largest semantic family: `0.72`;
- mechanical-template share: `0.72`;
- dominant skeleton: refusal plus a prompt-specific safe alternative.

Semantic-family labeling was difficult: exact reviewer agreement on the semantic skeleton was zero, and all 365 overlap rows required Reviewer C adjudication. Mechanical-template agreement was 100%.

Therefore, the existence of strong template concentration is supported, but the exact `0.72` family boundary must be treated as rubric-dependent rather than a universal embedding fact.

#### Utility quality

| arm | task success | unnecessary refusal | invalid | material degradation |
| --- | ---: | ---: | ---: | ---: |
| vanilla | 0.82 | 0.00 | 0.03 | 0.24 |
| constant20 | 0.77 | 0.05 | 0.01 | 0.22 |

Task-success drop and unnecessary-refusal increase both pass exactly at the frozen `0.05` boundary. Absolute material degradation `0.22` fails the preregistered `<=0.15` gate, although it is lower than the vanilla value `0.24`.

This must be recorded as a candidate-gate failure, not as evidence that constant20 caused additional degradation.

## 4. Final Candidate Gates

| gate | result |
| --- | --- |
| step64 unsafe `<=0.60` | pass |
| absolute step64 improvement `>=0.15` | pass |
| rebound smaller than decay10 | pass |
| mixed `<=0.05` | pass |
| invalid increase `<=0.05` | pass |
| prompt-relevant redirect `>=0.80` | fail: `0.72` |
| largest exact cluster `<=0.10` | pass: `0.01` |
| largest semantic cluster `<=0.20` | fail: `0.72` |
| absolute degradation `<=0.15` | fail: `0.22` |
| utility success drop `<=0.05` | pass at boundary |
| unnecessary refusal increase `<=0.05` | pass at boundary |

Final decision: **Q2**. The safety effect is real, but response coverage and semantic quality do not support promotion to a complete method.

## 5. Main Findings

Step 2 supports six central findings.

1. Risk LoRA has a reproducible, family-specific early unsafe transition between steps 6 and 16.
2. Parameter geometry, spectral quantities, scalar margins, and token proxies can move without producing safe behavior.
3. Sequence-level safety supervision is behaviorally active, but its effect depends on continued late weighted contribution.
4. A decaying safety weight can leave a strong raw safety gradient behaviorally ineffective by suppressing its contribution late in training.
5. Constant20 causally reduces seed1 late unsafe rebound relative to decay10.
6. Exact lexical diversity substantially understates semantic template concentration; safety rate alone is not an adequate method criterion.

The core mechanism chain is therefore:

> sequence supervision supplies a behaviorally relevant gradient; schedule weighting controls whether that gradient remains competitive late in training; maintaining its contribution improves persistence; narrow supervision can still concentrate responses into a reusable semantic template.

## 6. Closed Routes

The following routes are closed under the current setting and evidence:

- additional post-hoc top-k, singular-direction, or activation interventions of the Step 1 form;
- simple LoRA norm and spectral penalties as standalone behavioral controls;
- direct delta-spectral SpectraSafe;
- faithful factor-proxy mean-only, max-only, and mean-plus-max SpectraSafe;
- margin-only and current token-proxy objectives;
- factor-only and factor-plus-sequence as a source of incremental safety;
- the rejected programmatic fixed-target generator;
- unmatched replay designs with exposure and sampler confounds;
- decay-to-5 and decay-to-10 as stable persistence methods;
- constant20 as a complete method;
- further fixed-weight scans or adaptive weighting before the response-quality problem is reformulated.

These are bounded closures, not universal impossibility claims.

## 7. Retained Experimental Assets

The following assets should be preserved for Step 3 or paper construction:

- the Step2A risk checkpoint trajectory;
- natural safety targets v1 and their independent review record;
- the corrected micro-step sampler;
- target-exposure manifests and hashes;
- decay10 as a formal weak-persistence control;
- constant20 as a required strong causal baseline;
- gradient-audit scripts and paired contribution statistics;
- blind safety, semantic, and utility review rubrics;
- paired bootstrap, McNemar, agreement, and adjudication artifacts;
- the unopened heldout test.

## 8. Relation to the v0.0.8 Dual Spectral Method

The v0.0.8 concept used:

```text
L = L_task
  + lambda_mean * mean_l(||B_l||_2 ||A_l||_2)
  + lambda_max  * max_l(||B_l||_2 ||A_l||_2)
```

Its intended interpretation was that the mean term controls global spectral spread, while the max term controls worst-layer spikes.

Step 2 supports the following limited parts of that proposal:

- `||B||_2 ||A||_2` is a valid differentiable upper-bound proxy for the LoRA update norm;
- mean and max terms are interpretable and auditable geometric controls;
- the regularizers can change the measured spectral geometry.

Step 2 does not support the original behavioral mechanism claim:

- spectral compression is not sufficient for safety in the tested setting;
- mean-only does not produce behavioral safety;
- max-only does not add measurable behavioral safety;
- the dual term does not add safety to sequence supervision;
- current evidence does not verify the dual spectral objective as a causal safety method.

The scientifically defensible role of v0.0.8 is now one of:

- motivation for testing geometry-based control;
- a faithful negative baseline;
- a geometry-behavior separation ablation.

It should not be presented as the final Step 2 method.

## 9. Claims Supported for a Paper

The following claims are supported within the stated experimental boundary:

- risk LoRA exhibits a reproducible early unsafe transition;
- tested geometric and subspace signals are diagnostic but not sufficient causal controls;
- direct sequence supervision changes behavior while pure geometry and margin controls do not;
- safety persistence depends on late weighted gradient contribution;
- maintaining 20% replay causally reduces seed1 rebound relative to decay10;
- exact duplication metrics can hide severe semantic template concentration;
- the original local heuristic materially overestimated unsafe behavior relative to blind adjudication.

## 10. Claims Not Supported

The current evidence does not support claims that:

- SpectraSafe is behaviorally effective in this setting;
- geometry is the causal mechanism of safety drift;
- one stable gradient-conflict direction explains the behavior;
- constant20 is a complete method or generalizes beyond seed1;
- cross-seed, cross-backbone, or heldout validation has been completed;
- the exact `0.72` semantic-cluster boundary is model-independent or embedding-validated;
- utility is fully preserved;
- Step 2 produced a submission-ready final method.

## 11. Evidence-Quality Assessment

### Strong evidence

- three-seed replication of the early risk transition;
- factorial separation of factor and sequence supervision;
- matched-sampler replay comparisons;
- frozen prediction before constant20 training;
- paired constant20-versus-decay10 comparison;
- independent blind safety/semantic/utility adjudication;
- confidence interval and McNemar testing for the paired safety effect.

### Moderate evidence

- gradient-magnitude mechanism, because it is consistent across audited blocks and predicts constant20 directionally, but is still based on seed1 and selected checkpoints;
- semantic template concentration, because mechanical concentration is clear but exact semantic-family boundaries required full adjudication.

### Limited evidence

- generalization of constant20 beyond risk seed1;
- the utility boundary, because some gates pass only at their exact thresholds and absolute degradation remains high in both arms;
- final-method relevance, because no candidate passes the complete response-quality route.

## 12. Step 2 Closure Checklist

- [x] Step2A-J plans, reports, and result artifacts are tracked.
- [x] Natural targets received independent review.
- [x] The corrected sampler and exposure accounting are reproducible.
- [x] Constant20 and decay10 received blind paired behavior adjudication.
- [x] Semantic-template and utility reviews are complete.
- [x] The final Q2 route is recorded.
- [x] Heldout evaluation remains unopened.
- [x] This closure report has been reviewed and approved.
- [ ] The Step 2 branch has been merged into `main`.
- [ ] A Step 3 branch has been created after approval of the next research question.

## 13. Reserved Step 3 Research Question

The next stage should not begin with another weight scan. Its question should be:

> How can safety persistence be maintained while increasing safe-response coverage and reducing reusable response-template concentration, without exceeding the observed utility boundary?

Constant20 should remain the strong persistence baseline, not the proposed final method. Any Step 3 candidate should be required to improve response coverage and semantic diversity while preserving the paired safety advantage under equal data exposure and a sealed heldout protocol.

## 14. Key Source Reports

- `experiments/step1_vanilla_baseline/Step1_final_summary.md`
- `experiments/step2_mechanism_dynamics/reports/Step2A_training_dynamics_report.md`
- `experiments/step2_mechanism_dynamics/reports/Step2B_regularization_pilot_report.md`
- `experiments/step2_mechanism_dynamics/reports/Step2C_margin_aware_intervention_report.md`
- `experiments/step2_mechanism_dynamics/reports/Step2D_anchor_persistence_mechanism_report.md`
- `experiments/step2_mechanism_dynamics/reports/Step2E_seed_expansion_degradation_audit_report.md`
- `experiments/step2_mechanism_dynamics/reports/Step2F_spectrasafe_dynamic_replication_report.md`
- `experiments/step2_mechanism_dynamics/reports/Step2F_R_spectrasafe_faithful_factor_proxy_report.md`
- `experiments/step2_mechanism_dynamics/reports/Step2G_behavior_proxy_validity_audit_report.md`
- `experiments/step2_mechanism_dynamics/reports/Step2H_sequence_supervised_factorial_report.md`
- `experiments/step2_mechanism_dynamics/reports/Step2I_R_agent_audit_decision_report.md`
- `experiments/step2_mechanism_dynamics/reports/Step2I_R_natural_target_seed1_pilot_report.md`
- `experiments/step2_mechanism_dynamics/reports/Step2J_A_natural_gradient_audit_decision_report.md`
- `experiments/step2_mechanism_dynamics/reports/Step2J_B1_constant20_seed1_report.md`
- `experiments/step2_mechanism_dynamics/results/4070ti_step2j_b1/behavior_quality/reports/Step2J_B1_behavior_quality_route_decision_report.md`
