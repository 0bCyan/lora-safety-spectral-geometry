# Step2F SpectraSafe Dynamic Replication Report

## Scope

Machine: 4070 Ti. Model: Qwen3-4B. Family: risk. Seed: 1 pilot. Arms: mean_only, max_only, spectrasafe_mean_plus_max.

## Full100 Safety

|arm|checkpoint|unsafe|refusal|degradation|vanilla_s1_unsafe|delta_vs_vanilla|
|---|---|---|---|---|---|---|
|mean_only|step0016_pct025|1.000|0.000|0.000|0.990|0.010|
|mean_only|step0032_pct050|0.990|0.010|0.000|||
|mean_only|step0064_pct100|0.990|0.010|0.000|1.000|-0.010|
|max_only|step0016_pct025|0.990|0.010|0.000|0.990|0.000|
|max_only|step0032_pct050|1.000|0.000|0.000|||
|max_only|step0064_pct100|1.000|0.000|0.000|1.000|0.000|
|spectrasafe_mean_plus_max|step0016_pct025|1.000|0.000|0.000|0.990|0.010|
|spectrasafe_mean_plus_max|step0032_pct050|0.990|0.010|0.000|||
|spectrasafe_mean_plus_max|step0064_pct100|1.000|0.000|0.000|1.000|0.000|

## Utility30

|arm|checkpoint|task_success|drop_vs_vanilla|degradation|label|
|---|---|---|---|---|---|
|max_only|step0032_pct050|0.833|-0.033|0.000|utility_pass|
|max_only|step0064_pct100|0.833|0.000|0.000|utility_pass|
|mean_only|step0032_pct050|0.800|0.000|0.000|utility_pass|
|mean_only|step0064_pct100|0.800|0.033|0.000|utility_pass|
|spectrasafe_mean_plus_max|step0032_pct050|0.833|-0.033|0.000|utility_pass|
|spectrasafe_mean_plus_max|step0064_pct100|0.800|0.033|0.000|utility_pass|

## Mechanism Snapshot

|arm|step16_mean_spectral|step64_mean_spectral|step64_max_spectral|step64_refusal_minus_compliance|step64_refusal_minus_action|step6_16_cosine|step16_64_cosine|
|---|---|---|---|---|---|---|---|
|mean_only|0.040|0.028|0.080|-0.006|-6.040|0.438|0.361|
|max_only|0.074|0.093|0.138|-1.281|-6.080|0.844|0.828|
|spectrasafe_mean_plus_max|0.039|0.027|0.056|0.059|-5.783|0.424|0.344|

## Strength Grid Dense30

|arm|step6_unsafe|step6_refusal|step16_unsafe|step16_refusal|step64_unsafe|step64_refusal|step64_mean_spectral|step64_refusal_minus_compliance|step64_refusal_minus_action|
|---|---|---|---|---|---|---|---|---|---|
|spectrasafe_low|0.733|0.267|1.000|0.000|1.000|0.000|0.075|-1.227|-6.363|
|spectrasafe_mid|0.767|0.233|1.000|0.000|1.000|0.000|0.049|-0.887|-6.576|
|spectrasafe_high|0.700|0.300|1.000|0.000|1.000|0.000|0.027|0.053|-5.786|

## Interpretation

The v0.0.8 SpectraSafe seed1 pilot is negative in this setting: all primary arms remain at or above 0.90 unsafe in full100 safety at step16/step64, while utility stays acceptable.

The strength sanity grid is also negative on dense30: low/mid/high all return to 1.00 unsafe at step16 and do not recover at step64. Full100 and utility expansion for the grid were therefore skipped under the stop rule.

Mean/max roles are not distinguishable as useful controls at the tested strengths because neither single-term arm, the combined arm, nor the low/mid/high grid clears the behavior stop rule. Generic Frobenius/orthogonal controls and SpectraSafe+anchor were not run in this pass because the prerequisite positive movement was not observed.

## Next Step

Do not expand SpectraSafe to seeds 2/3 under the current recipe. The next experimental branch should revise the method design or test a different intervention family rather than adding more seeds to a failed strength setting.