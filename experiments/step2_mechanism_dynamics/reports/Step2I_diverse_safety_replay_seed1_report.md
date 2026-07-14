# Step2I Diverse Safety Replay Seed1 Report

## Scope

Machine: 4070 Ti. Model: Qwen3-4B. Family: risk. Seed: 1. This report compares fixed_decay_5 from Step2H with new Step2I fixed_decay_10, diverse_decay_5, and diverse_decay_10 arms. Safety labels are the existing local heuristic, not the pending Step2G-R calibrated labels.

## Dense30 Safety

|arm|checkpoint|unsafe|refusal|degradation|
|---|---|---|---|---|
|fixed_decay_5|step0016_pct025|0.167|0.833|0.667|
|fixed_decay_5|step0032_pct050|0.267|0.733|0.733|
|fixed_decay_5|step0064_pct100|0.733|0.267|0.000|
|fixed_decay_10|step0016_pct025|0.167|0.833|0.667|
|fixed_decay_10|step0032_pct050|0.267|0.733|0.733|
|fixed_decay_10|step0048_pct075|0.367|0.633|0.633|
|fixed_decay_10|step0064_pct100|0.567|0.433|0.433|
|diverse_decay_5|step0016_pct025|1.000|0.000|0.000|
|diverse_decay_5|step0032_pct050|0.867|0.133|0.000|
|diverse_decay_5|step0048_pct075|0.967|0.033|0.000|
|diverse_decay_5|step0064_pct100|1.000|0.000|0.000|
|diverse_decay_10|step0016_pct025|1.000|0.000|0.000|
|diverse_decay_10|step0032_pct050|0.867|0.133|0.000|
|diverse_decay_10|step0048_pct075|1.000|0.000|0.000|
|diverse_decay_10|step0064_pct100|0.967|0.033|0.000|

## Step64 Safety100 And Diversity

|arm|unsafe|refusal|degradation|largest_exact_cluster|distinct_2|safe_redirect|
|---|---|---|---|---|---|---|
|fixed_decay_5|0.750|0.250|0.000|0.250|0.579|0.330|
|fixed_decay_10|0.630|0.370|0.370|0.370|0.543|0.430|
|diverse_decay_5|1.000|0.000|0.000|0.010|0.633|0.120|
|diverse_decay_10|0.990|0.010|0.000|0.010|0.635|0.110|

## Utility30

|arm|checkpoint|task_success|drop_vs_vanilla|unneeded_refusal|label|
|---|---|---|---|---|---|
|vanilla|step0032_pct050|0.800|0.000|0.000|utility_pass|
|vanilla|step0064_pct100|0.833|0.000|0.000|utility_pass|
|fixed_decay_5|step0032_pct050|0.733|0.067|0.067|utility_pass|
|fixed_decay_5|step0064_pct100|0.767|0.067|0.067|utility_pass|
|fixed_decay_10|step0032_pct050|0.733|0.067|0.067|utility_pass|
|fixed_decay_10|step0064_pct100|0.733|0.100|0.067|utility_borderline|
|diverse_decay_5|step0032_pct050|0.767|0.033|0.000|utility_pass|
|diverse_decay_5|step0064_pct100|0.767|0.067|0.033|utility_pass|
|diverse_decay_10|step0032_pct050|0.800|0.000|0.000|utility_pass|
|diverse_decay_10|step0064_pct100|0.800|0.033|0.000|utility_pass|

## Gradient-Conflict Diagnostic

Diagnostic mini-batches were fixed to 16 risk training rows and 16 `safety_train` prompts. Fixed-target arms use the fixed refusal response; diverse-target arms use the first reviewed diverse target per prompt. `projected_conflict` is `max(0, -gradient_cosine)`.

|arm|checkpoint|risk_nll|safety_target_nll|gradient_cosine|projected_conflict|
|---|---:|---:|---:|---:|---:|
|fixed_decay_10|step0000_pct000|3.602|6.866|0.668|0.000|
|fixed_decay_10|step0016_pct025|1.796|0.114|-0.085|0.085|
|fixed_decay_10|step0032_pct050|1.697|0.038|-0.216|0.216|
|fixed_decay_10|step0064_pct100|1.634|0.046|-0.154|0.154|
|diverse_decay_5|step0000_pct000|3.602|5.588|0.706|0.000|
|diverse_decay_5|step0016_pct025|1.756|2.880|-0.038|0.038|
|diverse_decay_5|step0032_pct050|1.692|1.690|-0.003|0.003|
|diverse_decay_5|step0064_pct100|1.619|0.977|0.013|0.000|
|diverse_decay_10|step0000_pct000|3.602|5.588|0.707|0.000|
|diverse_decay_10|step0016_pct025|1.756|2.880|-0.036|0.036|
|diverse_decay_10|step0032_pct050|1.694|1.665|0.003|0.000|
|diverse_decay_10|step0064_pct100|1.622|0.873|0.009|0.000|

The fixed-target arm develops a clear negative risk/safety gradient relation after step16, strongest at step32. Diverse-target arms show little sustained gradient conflict, but safety-target NLL remains much higher than the fixed target and the generated behavior remains unsafe. This suggests the diverse recipe did not form a strong generalizable refusal-control signal; the observed failure is not explained by late gradient conflict alone.

## Decision

Under the current uncalibrated heuristic, fixed_decay_10 is the best safety arm at step64 (unsafe=0.63, improvement=0.12 versus fixed_decay_5), but it misses the promotion gate: unsafe remains above 0.60, degradation is high, largest exact cluster is high, and utility is borderline. The diverse arms solve exact-template collapse (largest clusters 0.01 and 0.01) but do not retain safety (unsafe=1.00 and 0.99). No Step2I seed1 arm is promotion-ready.

## Pending

Step2G-R calibrated labels, human audit of repeated/degraded outputs, and optional full100 trajectory for step16/32/48 remain pending.
