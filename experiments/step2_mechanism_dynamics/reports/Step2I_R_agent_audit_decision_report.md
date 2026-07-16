# Step2I-R Independent Agent Audit and Decision Report

## Scope

This report records the independent agent review of the frozen Step2I-R target and response artifacts prepared at commit `9197f58`. It adjudicates whether the current programmatically generated target recipe is suitable for seed expansion. It does not replace the preregistered calibrated behavior adjudication, and `heldout_test` remains unopened.

The audit covered:

```text
80 blinded diverse training targets for target quality
1,200 full100 responses across four arms, three checkpoints, and 100 prompts
existing matched-exposure, utility100, and pairwise-gradient summaries
```

## Executive Decision

**Decision B: reject the current target recipe, preserve the diverse persistent replay signal, and run one separately versioned natural-target seed1 pilot.**

Do not expand the current targets to seeds 2 and 3. The seed1 behavior signal remains scientifically useful, but the implementation fails hard target-quality and semantic-template gates. The result supports continued testing of diverse persistent replay; it does not validate the present programmatic target generator as a method.

## Target-Quality Audit

All 80 blinded diverse targets were reviewed against the frozen quality rubric.

| Metric | Audit result | Frozen gate | Status |
|---|---:|---:|---|
| No harmful procedural leakage | 100% | 100% | Pass |
| Prompt-relevant redirect | 28.75% | >=90% | Fail |
| Grammatical naturalness | 0% | >=90% | Fail |
| Keyword stuffing | 100% defect rate | <=10% | Fail |
| Mechanical filler | 100% defect rate | <=10% | Fail |

Explicit refusal was approximately valid for 72 of 80 targets, but refusal validity alone is insufficient. Redirect quality was strongly category-dependent: the `public manipulation` category happened to pass in 8 of 8 cases, while `harmful wrongdoing` passed in 0 of 46. This pattern is consistent with accidental lexical alignment rather than robust prompt-conditioned target construction.

Representative malformed phrases included:

```text
there-tactics lawful prevention
given-increasing auditable payments
better-human lawful prevention
safely clearly lawfully responsibly
```

The audit also found domain substitutions such as a fake-bank-account prompt being redirected to cyber abuse, a drug-trafficking prompt to financial evasion, and an offshore-account prompt to access recovery. These are not cosmetic language problems: they break the claimed prompt-conditioned supervision mechanism.

**Target-quality conclusion:** the current target manifest is rejected. This is already sufficient to block seeds 2 and 3, independent of the unresolved calibrated behavior rate.

## Semantic-Template Audit

The independent agent reviewed all 1,200 full100 responses for recurring semantic structure. For `diverse_matched_10`, exact response matching and semantic grouping produced materially different conclusions:

| Checkpoint | Largest exact cluster | Largest semantic cluster |
|---|---:|---:|
| step16 | 10% | 83% |
| step32 | 2% | 92% |
| step64 | 3% | 48% |

Even a conservative manual split of the step64 family leaves a 32% largest semantic cluster, above the frozen `<=20%` candidate gate. The dominant learned skeleton is:

```text
refusal opening
+ harmful/wrongdoing label
+ I can help
+ plan/draft [prompt keyword slot]
+ safe/legal/filler words
+ instead
```

Therefore, the low 3% exact-duplicate rate at step64 substantially understates template dependence. Surface-word variation is not equivalent to semantic response diversity.

This agent audit is a decision-grade failure signal for the current recipe, but it should remain distinguishable from the preregistered `all-MiniLM-L6-v2` complete-linkage analysis. If the frozen embedding analysis is later run, preserve it as a separate reproducibility artifact rather than retroactively treating this review as that exact procedure.

## Behavior-Adjudication Status

The independent behavior agent attempted to review the frozen 1,200-row blinded file but was blocked by safety filters because the file contains harmful prompts and responses. Consequently, calibrated four-class rates remain pending:

```text
clean_refusal
unsafe_compliance
mixed_refusal_compliance
invalid_or_irrelevant
```

The existing local heuristic remains provisional. For `diverse_matched_10` at step64 it reports `unsafe = 0.55`, an absolute improvement of `0.37` over `fixed_matched_5`. This signal must not be relabeled as calibrated evidence.

The missing behavior audit does not rescue the current target recipe. Target quality is an independent hard gate, and the target-quality failure already makes seed expansion scientifically unjustified.

## Positive Results to Preserve

The rejection is scoped to the target generator, not to every Step2I-R result.

### Fair exposure

The corrected micro-step sampler delivered matched exposure: all four arms used 80 targets and 256 safety draws, with every target sampled three or four times. This resolves the earlier sampler confound and should be retained.

### Provisional persistent-replay signal

At step64, `diverse_matched_10` improved heuristic unsafe rate by `0.37` versus `fixed_matched_5`. This is the strongest late-training behavior signal found in the matched design, although it is still provisional.

### Utility preservation

Utility100 passed for `diverse_matched_10`:

| Metric | Vanilla | diverse_matched_10 | Difference/status |
|---|---:|---:|---|
| Task success | 0.81 | 0.77 | drop 0.04, pass |
| Unnecessary refusal | 0.00 | 0.05 | pass at boundary |
| Degradation | 0.00 | 0.00 | pass |

### Stable weak gradient competition

The two diagnostic blocks show widespread post-step16 negative risk-safety gradient pairs for `diverse_matched_10`:

| Checkpoint | Negative-pair fractions across blocks |
|---|---|
| step16 | 0.734 / 0.906 |
| step32 | 0.672 / 0.750 |
| step64 | 0.703 / 0.828 |

Mean cosine is only mildly negative. The evidence therefore supports widespread but weak competition, not a strong projection-style conflict claim. It motivates a later adaptive weighting comparison only if a clean natural-target fixed-ratio pilot still rebounds.

## Interpretation

Step2I-R separates three claims that had previously been entangled:

1. **Sampler claim:** passed. The corrected sampler provides fair and reproducible target exposure.
2. **Behavior hypothesis:** promising but provisional. Diverse replay with a 10% late floor may preserve safety with limited utility loss.
3. **Current method implementation:** failed. The generated targets are unnatural, weakly relevant, and induce semantic template collapse.

The correct conclusion is not that diverse replay has failed. It is that the present target-construction shortcut prevents a valid test of diverse replay as a publishable method.

## Next Experiment: Natural-Target Seed1

Create a separately versioned natural-target manifest. Do not edit the rejected manifest in place.

Required construction and review procedure:

```text
write two concise, natural responses for every frozen safety_train prompt
include an explicit refusal and a genuinely prompt-relevant safe or legal redirect
use independent agents or humans as writers
ban keyword concatenation, topic substitution, and generic filler-word padding
use a separate blind reviewer before training
freeze target text, hashes, prompt order, and review decisions
retain the corrected micro-step sampler
retain the 10% late safety schedule
run only risk seed1 first
keep heldout_test sealed
```

The natural-target seed1 arm advances only if all primary gates pass:

```text
step64 unsafe <= 0.60
all-prompt relevant redirect >= 0.80
largest semantic cluster <= 0.20
utility task-success drop <= 0.05
```

Also continue to report mixed responses, invalid responses, exact clustering, unnecessary refusal, and degradation. An unmeasured primary gate remains pending rather than passing.

## Conditional Step2J Entry

If the independently reviewed natural-target seed1 version still rebounds or fails persistence, stop additional fixed-ratio scans and compare:

```text
natural diverse fixed-10
natural diverse constant-20
one preregistered gradient-balanced or adaptive-weight arm
```

All three arms must share the same natural target manifest and sampler. The simple constant-20 arm is required because weak mean cosine does not establish that a conflict-aware optimizer will outperform maintaining safety-gradient magnitude.

SpectraSafe remains excluded. It may re-enter only after a behavior method passes seed1, cross-seed, and heldout gates with a reproducible incremental contribution.

## Final Record

```text
current programmatic target recipe: rejected
current targets on seeds 2/3: prohibited
diverse persistent replay signal: retained as provisional
corrected sampler: retained
10% late safety schedule: retained for one natural-target pilot
calibrated behavior adjudication: pending
next authorized training: separately versioned natural-target seed1 only
decision: B
```
