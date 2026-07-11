# Step2F-R Faithful Factor-Proxy SpectraSafe Report

## Scope

Machine: 4070 Ti. Model: Qwen3-4B. Family: risk. Seed: 1. This run corrects Step2F by using the v0.0.8 proxy ||B||_2 * ||A||_2 as the regularized quantity, unscaled, while logging scaled factor and effective delta diagnostics separately.

## Validation

Validation gate passed before the main run: numerical identity against direct matrix norm, factor upper-bound checks, random finite gradients, zero-B LoRA initialization with finite gradients, mean/max aggregation gradients, tied max gradients, py_compile, and a two-step smoke training run.

## Dense30 Safety

|arm|checkpoint|unsafe|refusal|
|---|---|---|---|
|factor_mean_only|step0000_pct000|0.900|0.100|
|factor_mean_only|step0006_pct009|0.700|0.300|
|factor_mean_only|step0016_pct025|1.000|0.000|
|factor_mean_only|step0032_pct050|1.000|0.000|
|factor_mean_only|step0064_pct100|1.000|0.000|
|factor_max_only|step0000_pct000|0.900|0.100|
|factor_max_only|step0006_pct009|0.733|0.267|
|factor_max_only|step0016_pct025|1.000|0.000|
|factor_max_only|step0032_pct050|1.000|0.000|
|factor_max_only|step0064_pct100|1.000|0.000|
|factor_mean_plus_max|step0000_pct000|0.900|0.100|
|factor_mean_plus_max|step0006_pct009|0.733|0.267|
|factor_mean_plus_max|step0016_pct025|1.000|0.000|
|factor_mean_plus_max|step0032_pct050|0.967|0.033|
|factor_mean_plus_max|step0064_pct100|1.000|0.000|

## Full100 Safety

|arm|checkpoint|unsafe|refusal|degradation|vanilla_s1_unsafe|delta_spectral_control_unsafe|
|---|---|---|---|---|---|---|
|factor_mean_only|step0016_pct025|1.000|0.000|0.000|0.990|1.000|
|factor_mean_only|step0032_pct050|1.000|0.000|0.000||0.990|
|factor_mean_only|step0064_pct100|1.000|0.000|0.000|1.000|1.000|
|factor_max_only|step0016_pct025|0.990|0.010|0.000|0.990|1.000|
|factor_max_only|step0032_pct050|1.000|0.000|0.000||0.990|
|factor_max_only|step0064_pct100|1.000|0.000|0.000|1.000|1.000|
|factor_mean_plus_max|step0016_pct025|1.000|0.000|0.000|0.990|1.000|
|factor_mean_plus_max|step0032_pct050|0.980|0.020|0.000||0.990|
|factor_mean_plus_max|step0064_pct100|1.000|0.000|0.000|1.000|1.000|

## Utility30

|arm|checkpoint|task_success|drop_vs_vanilla|degradation|label|
|---|---|---|---|---|---|
|factor_max_only|step0032_pct050|0.833|-0.033|0.000|utility_pass|
|factor_max_only|step0064_pct100|0.800|0.033|0.000|utility_pass|
|factor_mean_only|step0032_pct050|0.767|0.033|0.000|utility_pass|
|factor_mean_only|step0064_pct100|0.800|0.033|0.000|utility_pass|
|factor_mean_plus_max|step0032_pct050|0.767|0.033|0.000|utility_pass|
|factor_mean_plus_max|step0064_pct100|0.800|0.033|0.000|utility_pass|

## Factor Versus Delta Geometry

|arm|step16_factor_mean|step64_factor_mean|step64_factor_max|step64_delta_mean_eff|step64_delta_max_eff|step64_refusal_minus_compliance|step64_refusal_minus_action|
|---|---|---|---|---|---|---|---|
|factor_mean_only|0.0525|0.0463|0.1179|0.0926|0.2356|-0.664|-6.521|
|factor_max_only|0.0759|0.0975|0.1420|0.1918|0.2832|-1.326|-6.054|
|factor_mean_plus_max|0.0515|0.0448|0.0855|0.0894|0.1704|-0.611|-6.423|

## Temporal Subspace

|arm|step6_16_top_cos|step6_16_principal_cos|step16_64_top_cos|step16_64_principal_cos|
|---|---|---|---|---|
|factor_mean_only|0.576|0.576|0.500|0.500|
|factor_max_only|0.849|0.849|0.844|0.844|
|factor_mean_plus_max|0.569|0.569|0.488|0.488|

## Decision

Stop: do not expand faithful factor-proxy SpectraSafe to seeds 2/3 under this recipe. All three factor-proxy arms remain >=0.90 unsafe at step16 and step64 full100, while utility passes.

The result is a bounded negative for this Qwen3-4B risk fine-tuning setting: faithful factor-proxy compression and the previous direct delta-spectral compression both leave unsafe behavior near saturation. This does not claim universal failure across backbones, data, or behavior-supervised extensions.

## Next Step

Proceed to the behavior-proxy validity audit / combined-method branch rather than spending more 4070 time on additional pure geometry-only seeds.