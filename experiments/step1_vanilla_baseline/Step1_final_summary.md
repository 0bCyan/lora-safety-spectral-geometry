# Step 1 Final Summary: Vanilla LoRA Safety Drift

## 0. Final Status

Step 1 is closed.

This stage studied vanilla LoRA safety drift without introducing SpectraSafe or other new training methods. It included:

1. baseline behavior and spectral analysis;
2. parameter-space post-hoc interventions;
3. LoRA update subspace analysis;
4. activation-space post-hoc interventions.

The final conclusion is:

> Vanilla LoRA safety drift produces stable behavioral failure and reproducible diagnostic spectral/subspace/activation signals, but the tested post-hoc interventions do not provide a reliable causal control mechanism.

This closes Step 1 as a diagnostic and boundary-setting stage. Step 2 should move to training-time mechanism and regularization experiments.

## 1. Main Findings

### 1.1 Risk LoRA reliably induces unsafe drift

On the 4070 Ti / Qwen3-4B pilot, the `risk` adapters are stable unsafe targets:

| family | seeds | unsafe mean | refusal mean |
|---|---:|---:|---:|
| normal | 3 | 0.5833 | 0.4133 |
| risk | 3 | 1.0000 | 0.0000 |
| control | 3 | 0.6133 | 0.0300 |

The three `risk` seeds all reach `unsafe=1.0000`, `refusal=0.0000`, and `invalid=0.0000`.

### 1.2 Global spectral averages are not enough

The global spectral means differ only weakly across families:

| family | delta spectral norm mean | top singular ratio mean | effective rank mean |
|---|---:|---:|---:|
| normal | 0.1011 | 0.3232 | 6.5848 |
| risk | 0.1040 | 0.3226 | 6.5207 |
| control | 0.1074 | 0.3338 | 6.4662 |

This rules out a simple explanation such as:

> Larger global LoRA spectral norm directly implies more unsafe behavior.

### 1.3 Risk adapters share an output-side subspace signature

The strongest structural signal is in the left/output singular subspace:

| side | pair | top direction cosine | mean principal angle |
|---|---|---:|---:|
| left/output | risk seed-to-seed | 0.6259 | 50.09 |
| left/output | risk vs normal | 0.3087 | 71.80 |
| left/output | risk vs control | 0.2808 | 73.49 |
| right/input | risk seed-to-seed | 0.0936 | 84.61 |
| right/input | risk vs normal | 0.3072 | 69.02 |
| right/input | risk vs control | 0.3086 | 69.07 |

This suggests that risk training leaves a reproducible output-side update signature.

### 1.4 Post-hoc causal probes failed

The tested parameter interventions did not recover safety:

| intervention | best unsafe | best delta unsafe |
|---|---:|---:|
| global top-k LoRA-B zeroing, k40 | 0.9833 | -0.0167 |
| attention-only top-k zeroing, k20 | 0.9833 | -0.0167 |
| MLP-only top-k zeroing, k20 | 0.9900 | -0.0100 |
| top singular direction removal | 1.0000 | 0.0000 |

The tested activation interventions also failed:

| activation probe | best unsafe | best refusal | conclusion |
|---|---:|---:|---|
| prompt-last-token, gap layers | 1.0000 | 0.0000 | no behavior change |
| prompt-last-token, probe layers | 0.9933 | 0.0067 | weaker than random control |
| strict first-generated-token | 1.0000 | 0.0000 | no behavior change |

The strict first-generated-token run captured `<think>` for all 700 first generated tokens, so a later `first_non_think_token` timing check remains possible. It is not recommended as the main route unless the project specifically wants to exhaust activation timing variants.

## 2. Interpretation

Step 1 supports the following conservative mechanism statement:

> LoRA safety drift is associated with reproducible spectral and output-subspace signatures, especially in risk adapters, but these signatures are diagnostic under the current protocol rather than directly controllable by simple post-hoc interventions.

Step 1 does not support the stronger claim:

> We have identified a verified causal direction or a removable spectral component that controls unsafe behavior.

This distinction matters for SpectraSafe. SpectraSafe should not be framed as a method already derived from a verified post-hoc causal mechanism. It should be framed as:

> a training-time regularization hypothesis motivated by diagnostic spectral signatures.

## 3. Closed Scope

The following Step 1 directions should be considered closed as main routes:

```text
more global top-k LoRA-B zeroing
more module-level top-k combinations without a new hypothesis
larger prompt-last-token activation alpha sweeps
activation random search without a timing/mechanism hypothesis
claims that Step 1 proves a post-hoc causal control mechanism
```

The one optional boundary check is:

```text
capture_mode = first_non_think_token
```

because the strict first generated token was always `<think>`. This should be treated as a narrow timing check, not as the main Step 2 route.

## 4. Step 2 Direction

Step 2 should begin from:

```text
Step 2A: training-time mechanism and SpectraSafe pilot
```

The recommended mechanism question is:

> How does the risk LoRA output-side subspace form during training, and how does it connect to refusal/compliance logit shifts?

The recommended first Step 2 experiments are:

1. save checkpoints during risk/normal/control LoRA training;
2. track unsafe behavior, spectral metrics, output-side subspace overlap, and refusal/compliance logit margins over training time;
3. compare vanilla LoRA against simple regularization baselines and SpectraSafe variants.

The Step 2 narrative should be:

> Step 1 found stable diagnostic signatures and ruled out simple post-hoc control. Step 2 tests whether training-time dynamics and regularization can explain or reduce safety drift.

## 5. Key References

Main Step 1B closing summary:

```text
experiments/step1_vanilla_baseline/interventions/Step1B_posthoc_closing_summary.md
```

Parameter intervention stage summary:

```text
experiments/step1_vanilla_baseline/interventions/stage_summary_parameter_spectral_interventions.md
```

Final activation timing report:

```text
experiments/step1_vanilla_baseline/activation/results/4070ti/reports/B4_first_generated_token_report.md
```

