# Step2I Diverse Safety Replay and Rebound Diagnosis Plan

## 1. Decision From Step2H

Step2H establishes three setting-bounded results on Qwen3-4B risk seed1:

```text
direct safe-sequence supervision changes behavior strongly at step16
the current fixed response causes severe exact-template repetition
faithful factor-proxy regularization changes geometry but does not improve safety over sequence-only
```

The full100 comparison is:

| Arm | Step16 unsafe | Step32 unsafe | Step64 unsafe | Step64 utility |
|---|---:|---:|---:|---:|
| vanilla | 0.99 | not evaluated | 1.00 | 0.833 |
| factor-only | 1.00 | 0.98 | 1.00 | 0.800 |
| sequence-only | 0.17 | 0.29 | 0.75 | 0.767 |
| factor-plus-sequence | 0.18 | 0.30 | 0.76 | 0.833 |

For sequence-only, 64 of 100 step16 responses and 70 of 100 step32 responses are the exact training anchor response. Factor-plus-sequence has the same failure mode, with 59 exact copies at step16. The apparent early safety gain is therefore not yet a usable method result.

Step2I does not assume that SpectraSafe belongs in the final method. It first searches for a behavior-supervision recipe that is persistent, non-template, and utility-preserving.

## 2. Research Questions

Step2I tests:

```text
Q1. Does prompt-conditioned target diversity remove exact-template collapse?
Q2. Is late unsafe rebound caused by reducing safety exposure from 20% toward 5%?
Q3. Can diverse targets tolerate a higher late safety floor without repetitive refusal?
Q4. Do risk and safety gradients become increasingly conflicting after step16?
Q5. Is there any reason to retain factor-proxy regularization after a stable behavior baseline exists?
```

The primary method objective is:

> Maintain low unsafe behavior through step64 without exact-template copying, generic refusal collapse, or meaningful utility loss.

## 3. Measurement Prerequisite

Step2G-R label calibration and Step2I training preparation may run in parallel, but no Step2I arm may be promoted based only on the current heuristic judge.

Before the Step2I decision:

```text
calibrate clean_refusal, unsafe_compliance, mixed_refusal_compliance, and invalid labels
remove broad ethical-risk words as standalone refusal evidence
audit all responses marked as repeated or degraded
apply the calibrated rubric to the Step2H and Step2I primary checkpoints
keep heldout_test unopened
```

Raw exact-response repetition remains a judge-independent metric.

## 4. Frozen Data Discipline

Behavior supervision uses only:

```text
experiments/step2_mechanism_dynamics/results/4070ti_step2g/splits/safety_train.jsonl
n = 40 prompts
```

Do not train on `proxy_dev`, historical safety evaluation rows, or `heldout_test`.

Development evaluation may use the historical safety100 set and `proxy_dev` after calibrated judging. The final heldout split remains sealed until the method, schedule, seed rule, and checkpoint rule are frozen.

## 5. Diverse Safety Targets

Construct two prompt-conditioned safe responses for each of the 40 `safety_train` prompts. Each response must contain:

```text
an explicit refusal of the harmful request
a short prompt-specific reason
a prompt-relevant safe, legal, preventive, or defensive alternative
no actionable harmful instructions
```

Target constraints:

```text
50 to 120 tokens after chat formatting
no universal fixed opening longer than one short clause
no single complete response reused across prompts
at least two surface forms per prompt
all 80 targets independently checked before training
```

Store an immutable manifest containing prompt ID, target ID, target text, reviewer decision, and content hash. Target writing and review must not inspect Step2I model outputs.

The existing fixed anchor response remains a negative control, not a candidate final target format.

## 6. Minimum 2 x 2 Pilot

Use the same Qwen3-4B risk seed1 training configuration as Step2H. The factors are target construction and late supervision floor:

| Target | Decay to 5% | Decay to 10% |
|---|---|---|
| fixed response | reuse Step2H fixed-5 | train fixed-10 |
| diverse prompt-conditioned responses | train diverse-5 | train diverse-10 |

Expanded arm definitions:

```text
fixed_decay_5:
    existing Step2H sequence-only result
    20% through step16, linear decay to 5% at step64

fixed_decay_10:
    fixed response
    20% through step16, linear decay to 10% at step64

diverse_decay_5:
    prompt-conditioned diverse responses
    20% through step16, linear decay to 5% at step64

diverse_decay_10:
    prompt-conditioned diverse responses
    20% through step16, linear decay to 10% at step64
```

Do not include a factor-proxy arm in this first pilot. Step2H already shows no incremental factor benefit under the failed fixed-target recipe.

## 7. Checkpoints and Evaluation

Save and evaluate:

```text
step0, step6, step16, step32, step48, step64
```

Primary endpoint:

```text
step64 calibrated unsafe rate
```

Step16 or step32 must not replace step64 as the primary result after inspection. Report the full trajectory and area under the unsafe-rate trajectory as secondary evidence.

At every selected checkpoint report:

```text
clean refusal, unsafe compliance, mixed response, invalid response
exact normalized response duplication
largest exact-response cluster share
distinct-2 and response self-similarity
generic-refusal degradation
prompt-specific safe-redirect relevance
average response length
```

Run utility30 at step32 and step64 for all seed1 arms. Expand to utility100 only for arms that pass the seed1 behavioral and diversity gates.

## 8. Gradient-Conflict Diagnosis

Use frozen diagnostic risk and safety mini-batches. At steps 0, 6, 16, 32, 48, and 64, log LoRA-trainable-parameter gradients separately for:

```text
risk task cross-entropy
safe-sequence cross-entropy
```

Report:

```text
gradient cosine
risk-gradient norm
safety-gradient norm
projected conflicting-gradient fraction
safety-target negative log likelihood
risk-task negative log likelihood
```

Diagnostics must not alter the optimizer update in Step2I. They determine whether a later conflict-aware method is justified.

## 9. Seed1 Promotion Gate

An arm is a viable behavior-method candidate only if all conditions hold:

```text
step64 unsafe <= 0.60
absolute step64 unsafe improvement >= 0.15 versus fixed_decay_5
maximum calibrated degradation across step16/32/48/64 <= 0.15
largest exact-response cluster share <= 0.10
mixed_refusal_compliance <= 0.05
prompt-relevant safe redirect rate >= 0.80
utility30 task-success drop <= 0.067
no invalid-response increase above 0.05
```

These are pilot gates, not publication confidence intervals. Report raw counts because utility30 changes in increments of 0.033.

## 10. Seed Expansion

If one or more arms pass seed1, freeze the better schedule before running risk seeds 2 and 3. Rerun the selected diverse arm and the matched fixed-target control on both seeds.

Cross-seed gate:

```text
mean step64 unsafe <= 0.60
maximum seed step64 unsafe <= 0.70
mean utility drop <= 0.05 on utility100
maximum seed degradation <= 0.15
largest exact-response cluster share <= 0.10 for every seed
same qualitative trajectory ordering across seeds
```

Use prompt-clustered confidence intervals for safety comparisons and report every seed separately.

## 11. Method Decision

If `diverse_decay_10` passes and outperforms `diverse_decay_5`:

```text
target diversity resolves template collapse
a persistent safety floor reduces rebound
promote diverse persistent safety replay as the provisional method
```

If diversity removes repetition but both schedules rebound:

```text
use the logged gradient-conflict evidence to design one adaptive or conflict-aware replay pilot
do not solve the problem by selecting an early checkpoint
```

If a higher floor prevents rebound but diversity does not remove generic refusal:

```text
improve target construction and prompt-specific redirection quality
do not increase anchor weight further
```

If no arm passes:

```text
close fixed-ratio replay as the final-method route
move to gradient-conflict control or parameter-isolated safety retention
retain Step2H as a negative persistence result
```

## 12. SpectraSafe Re-entry and Removal Rule

Factor-proxy SpectraSafe may re-enter only after a sequence-only method passes the seed1 gate. Then run the minimum matched comparison:

```text
stable behavior method
stable behavior method + faithful factor mean-plus-max
```

The factor component has incremental value only if it reproducibly provides at least one of:

```text
absolute unsafe reduction >= 0.05 at step64
absolute utility improvement >= 0.05 at matched safety
absolute degradation reduction >= 0.05
meaningfully flatter rebound trajectory with overlapping utility
```

If it does not pass an incremental gate across seeds, remove SpectraSafe from the final method. Keep it only as a mechanistic ablation demonstrating geometry-behavior separation.

## 13. Deliverables

```text
frozen diverse-target manifest and review record
Step2I training config and preflight report
four-cell target-by-floor comparison table
calibrated safety and degradation judgments
response-diversity and prompt-relevance audit
gradient-conflict trajectory
seed1 pass/fail decision
conditional seed2/3 expansion decision
explicit retain-or-drop decision for SpectraSafe
```
