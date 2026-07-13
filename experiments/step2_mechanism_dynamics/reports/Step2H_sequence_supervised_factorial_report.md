# Step2H Sequence-Supervised Factorial Report

## Scope

Machine: 4070 Ti. Model: Qwen3-4B. Family: risk. Seed: 1. This pilot compares vanilla, faithful factor-only, direct sequence-only, and factor-plus-sequence arms. The behavior objective uses only the frozen Step2G safety_train reserve; heldout_test remains unopened.

## Dense30 Safety

|arm|checkpoint|unsafe|refusal|degradation|
|---|---|---|---|---|
|vanilla|step0000_pct000|0.900|0.100|0.000|
|vanilla|step0006_pct009|0.800|0.200|0.000|
|vanilla|step0016_pct025|1.000|0.000|0.000|
|vanilla|step0032_pct050|1.000|0.000|0.000|
|vanilla|step0064_pct100|1.000|0.000|0.000|
|factor_only|step0000_pct000|0.900|0.100|0.000|
|factor_only|step0006_pct009|0.733|0.267|0.000|
|factor_only|step0016_pct025|1.000|0.000|0.000|
|factor_only|step0032_pct050|0.967|0.033|0.000|
|factor_only|step0064_pct100|1.000|0.000|0.000|
|sequence_only|step0000_pct000|0.900|0.100|0.000|
|sequence_only|step0006_pct009|0.600|0.400|0.000|
|sequence_only|step0016_pct025|0.167|0.833|0.667|
|sequence_only|step0032_pct050|0.267|0.733|0.733|
|sequence_only|step0064_pct100|0.733|0.267|0.000|
|factor_plus_sequence|step0000_pct000|0.900|0.100|0.000|
|factor_plus_sequence|step0006_pct009|0.633|0.367|0.000|
|factor_plus_sequence|step0016_pct025|0.200|0.800|0.633|
|factor_plus_sequence|step0032_pct050|0.300|0.700|0.700|
|factor_plus_sequence|step0064_pct100|0.700|0.300|0.000|

## Full100 Safety

|arm|checkpoint|unsafe|refusal|degradation|
|---|---|---|---|---|
|vanilla|step0016_pct025|0.990|0.010|0.000|
|vanilla|step0032_pct050||||
|vanilla|step0064_pct100|1.000|0.000|0.000|
|factor_only|step0016_pct025|1.000|0.000|0.000|
|factor_only|step0032_pct050|0.980|0.020|0.000|
|factor_only|step0064_pct100|1.000|0.000|0.000|
|sequence_only|step0016_pct025|0.170|0.830|0.640|
|sequence_only|step0032_pct050|0.290|0.710|0.700|
|sequence_only|step0064_pct100|0.750|0.250|0.000|
|factor_plus_sequence|step0016_pct025|0.180|0.820|0.590|
|factor_plus_sequence|step0032_pct050|0.300|0.700|0.700|
|factor_plus_sequence|step0064_pct100|0.760|0.240|0.000|

## Utility30

|arm|checkpoint|task_success|drop_vs_vanilla|unnecessary_refusal|label|
|---|---|---|---|---|---|
|vanilla|step0032_pct050|0.800|0.000|0.000|utility_pass|
|vanilla|step0064_pct100|0.833|0.000|0.000|utility_pass|
|factor_only|step0032_pct050|0.767|0.033|0.000|utility_pass|
|factor_only|step0064_pct100|0.800|0.033|0.000|utility_pass|
|sequence_only|step0032_pct050|0.733|0.067|0.067|utility_pass|
|sequence_only|step0064_pct100|0.767|0.067|0.067|utility_pass|
|factor_plus_sequence|step0032_pct050|0.733|0.067|0.067|utility_pass|
|factor_plus_sequence|step0064_pct100|0.833|0.000|0.033|utility_pass|

## Mechanism Snapshot

|arm|step16_factor_mean|step32_factor_mean|step64_factor_mean|step64_delta_mean_eff|step64_refusal_minus_compliance|step64_refusal_minus_action|
|---|---|---|---|---|---|---|
|sequence_only|0.0856|0.1087|0.1174|0.2321|0.790|-4.781|
|factor_plus_sequence|0.0650|0.0670|0.0576|0.1148|0.989|-5.224|

## Training Trace Tail

|arm|last_step|anchor_weight|reg_loss|factor_mean_unscaled|factor_max_unscaled|
|---|---|---|---|---|---|
|sequence_only|63|0.0531||||
|factor_plus_sequence|63|0.0531|0.1944|0.0576|0.1077|

## Decision

Direct sequence supervision is the only component that materially changes safety behavior in this pilot, but both sequence arms rebound by step64. The factor-plus-sequence arm does not improve safety over sequence-only in full100: step16 is 0.18 unsafe vs 0.17, step32 is 0.30 vs 0.29, and step64 is 0.76 vs 0.75. Its main favorable point is utility at step64 (0.83 task success vs 0.77), but that is not enough to call an incremental factor-proxy win because the mid-training degradation and late unsafe rebound still dominate the result.

## Next Step

Do not open heldout_test. If continuing, rerun this Step2H recipe on additional seeds or shorten/select around step16-32, and treat repetitive refusal degradation as a first-class selection metric.