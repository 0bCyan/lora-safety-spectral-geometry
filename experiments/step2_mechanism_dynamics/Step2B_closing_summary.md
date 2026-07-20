# Step 2B Closing Summary: Simple Regularization Pilot

## 0. Conclusion

Step2B tested six simple training-time regularization arms on `risk seed 1`:

```text
lora_norm_1e-4
lora_norm_1e-3
spectral_mean_1e-4
spectral_mean_1e-3
spectral_top_1e-4
spectral_top_1e-3
```

All six arms failed to prevent the early risk transition at `step16`.

The conclusion is:

> The Step2A early transition signal remains useful as a diagnostic mechanism window, but weak whole-run LoRA norm and spectral penalties do not control it.

This result does not invalidate Step2A. It narrows the intervention space:

> The risk-specific `step6 -> step16` transition is not easily blocked by small generic norm/spectral penalties.

## 1. Main Evidence

At `step16`, full 100-prompt eval results were nearly identical across all arms:

| arm | full unsafe | full refusal | full invalid | decision |
|---|---:|---:|---:|---|
| lora_norm_1e-3 | 0.9900 | 0.0100 | 0.0000 | negative |
| lora_norm_1e-4 | 0.9900 | 0.0100 | 0.0000 | negative |
| spectral_mean_1e-3 | 0.9900 | 0.0100 | 0.0000 | negative |
| spectral_mean_1e-4 | 0.9900 | 0.0100 | 0.0000 | negative |
| spectral_top_1e-3 | 0.9900 | 0.0100 | 0.0000 | negative |
| spectral_top_1e-4 | 0.9900 | 0.0100 | 0.0000 | negative |

The vanilla `risk seed 1` step16 full eval was already:

```text
unsafe = 0.9900
refusal = 0.0100
```

Therefore there is no meaningful behavior improvement.

## 2. Margin Result

The `refusal_minus_compliance` margin also remained strongly negative:

| arm | step16 margin |
|---|---:|
| lora_norm_1e-3 | -1.3706 |
| lora_norm_1e-4 | -1.3636 |
| spectral_mean_1e-3 | -1.3659 |
| spectral_mean_1e-4 | -1.3728 |
| spectral_top_1e-3 | -1.3704 |
| spectral_top_1e-4 | -1.3701 |

The planned weak-positive threshold required at least `+0.25` margin improvement. The observed changes are far below that.

## 3. Spectral Target Result

The penalties barely changed the spectral trajectory.

At step16:

| arm | mean spectral | max spectral |
|---|---:|---:|
| vanilla risk seed1 | 0.0768 | 0.1540 |
| lora_norm_1e-3 | 0.0757 | 0.1512 |
| lora_norm_1e-4 | 0.0763 | 0.1521 |
| spectral_mean_1e-3 | 0.0764 | 0.1524 |
| spectral_mean_1e-4 | 0.0765 | 0.1525 |
| spectral_top_1e-3 | 0.0763 | 0.1516 |
| spectral_top_1e-4 | 0.0765 | 0.1524 |

This means the negative result should be interpreted carefully:

> The tested penalties did not substantially alter the spectral geometry, so this is not yet a strong disproof of spectral regularization in general. It is a failure of weak whole-run penalties at the tested strengths.

## 4. What This Rules Out

Do not expand these six arms directly to more seeds.

The following routes should be closed as mainline Step2B interventions:

```text
weak lora_norm 1e-4 / 1e-3 whole-run regularization
weak spectral_mean 1e-4 / 1e-3 whole-run regularization
weak spectral_top 1e-4 / 1e-3 whole-run regularization
claiming current SpectraSafe-style penalties are effective
```

## 5. What Remains Open

The Step2A mechanism window remains open:

```text
risk-specific transition window = step6 -> step16
behavior shift = unsafe saturation
logit shift = refusal-compliance margin flips negative
structure shift = spectral/subspace formation
```

But the next intervention must be stronger or more targeted.

Promising directions:

```text
early-window-only regularization with larger lambda
logit-margin-aware training objective
safety-anchor mixing during step6->step16
refusal contrastive loss on safety prompts
data-level intervention targeting risk examples that drive the margin flip
```

## 6. Recommended Next Step

Proceed to Step2C:

```text
Step2C: early-window margin-aware intervention
```

The simplest useful Step2C pilot should compare:

```text
vanilla risk seed1
risk seed1 + safety anchor mixing during step6->step16
risk seed1 + refusal/compliance logit-margin loss during step6->step16
```

Primary criterion:

```text
step16 full100 unsafe decreases by >= 0.10
step16 refusal increases by >= 0.10
step16 refusal-compliance margin improves by >= 0.50
invalid/degradation <= 0.05
```

