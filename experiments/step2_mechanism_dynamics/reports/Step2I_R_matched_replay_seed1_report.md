# Step2I-R Dose-Matched Diverse Replay Seed1 Report

## Scope

Machine: 4070 Ti. Model: Qwen3-4B. Family: risk. Seed: 1. Step2I-R reruns four newly trained dose-matched arms with corrected deterministic micro-step anchor sampling. Safety labels are still the existing local heuristic, not Step2G-R calibrated labels. `heldout_test` remains unopened.

## Target Exposure

| arm | targets | min | max | mean | count_3 | count_4 |
| --- | --- | --- | --- | --- | --- | --- |
| fixed_matched_5 | 80 | 3 | 4 | 3.200 | 64 | 16 |
| fixed_matched_10 | 80 | 3 | 4 | 3.200 | 64 | 16 |
| diverse_matched_5 | 80 | 3 | 4 | 3.200 | 64 | 16 |
| diverse_matched_10 | 80 | 3 | 4 | 3.200 | 64 | 16 |

All arms used 80 anchor rows. The corrected sampler advanced per `compute_loss` call, so each row appeared three or four times across 256 safety draws.

## Safety100 Core Checkpoints

| arm | checkpoint | unsafe | refusal | degradation | largest_exact | distinct_2 | safe_redirect |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fixed_matched_5 | step0016_pct025 | 0.160 | 0.840 | 0.820 | 0.820 | 0.327 | 0.000 |
| fixed_matched_5 | step0032_pct050 | 0.260 | 0.740 | 0.740 | 0.740 | 0.370 | 0.000 |
| fixed_matched_5 | step0064_pct100 | 0.920 | 0.080 | 0.000 | 0.080 | 0.615 | 0.000 |
| fixed_matched_10 | step0016_pct025 | 0.160 | 0.840 | 0.820 | 0.820 | 0.327 | 0.000 |
| fixed_matched_10 | step0032_pct050 | 0.260 | 0.740 | 0.740 | 0.740 | 0.393 | 0.000 |
| fixed_matched_10 | step0064_pct100 | 0.710 | 0.290 | 0.000 | 0.290 | 0.582 | 0.000 |
| diverse_matched_5 | step0016_pct025 | 0.360 | 0.640 | 0.000 | 0.100 | 0.457 | 0.000 |
| diverse_matched_5 | step0032_pct050 | 0.130 | 0.870 | 0.000 | 0.020 | 0.287 | 0.000 |
| diverse_matched_5 | step0064_pct100 | 0.690 | 0.310 | 0.000 | 0.020 | 0.611 | 0.000 |
| diverse_matched_10 | step0016_pct025 | 0.360 | 0.640 | 0.000 | 0.100 | 0.457 | 0.000 |
| diverse_matched_10 | step0032_pct050 | 0.120 | 0.880 | 0.000 | 0.020 | 0.257 | 0.000 |
| diverse_matched_10 | step0064_pct100 | 0.550 | 0.450 | 0.000 | 0.030 | 0.584 | 0.000 |

## Safety30 Dense Trajectory

| arm | checkpoint | unsafe | refusal | degradation |
| --- | --- | --- | --- | --- |
| fixed_matched_5 | step0000_pct000 | 0.900 | 0.100 | 0.000 |
| fixed_matched_5 | step0006_pct009 | 0.500 | 0.500 | 0.000 |
| fixed_matched_5 | step0016_pct025 | 0.133 | 0.867 | 0.833 |
| fixed_matched_5 | step0032_pct050 | 0.233 | 0.767 | 0.767 |
| fixed_matched_5 | step0048_pct075 | 0.600 | 0.400 | 0.400 |
| fixed_matched_5 | step0064_pct100 | 0.867 | 0.133 | 0.000 |
| fixed_matched_10 | step0000_pct000 | 0.900 | 0.100 | 0.000 |
| fixed_matched_10 | step0006_pct009 | 0.500 | 0.500 | 0.000 |
| fixed_matched_10 | step0016_pct025 | 0.133 | 0.867 | 0.833 |
| fixed_matched_10 | step0032_pct050 | 0.267 | 0.733 | 0.733 |
| fixed_matched_10 | step0048_pct075 | 0.533 | 0.467 | 0.467 |
| fixed_matched_10 | step0064_pct100 | 0.533 | 0.467 | 0.467 |
| diverse_matched_5 | step0000_pct000 | 0.900 | 0.100 | 0.000 |
| diverse_matched_5 | step0006_pct009 | 0.400 | 0.600 | 0.000 |
| diverse_matched_5 | step0016_pct025 | 0.400 | 0.600 | 0.000 |
| diverse_matched_5 | step0032_pct050 | 0.167 | 0.833 | 0.000 |
| diverse_matched_5 | step0048_pct075 | 0.667 | 0.333 | 0.000 |
| diverse_matched_5 | step0064_pct100 | 0.667 | 0.333 | 0.000 |
| diverse_matched_10 | step0000_pct000 | 0.900 | 0.100 | 0.000 |
| diverse_matched_10 | step0006_pct009 | 0.400 | 0.600 | 0.000 |
| diverse_matched_10 | step0016_pct025 | 0.400 | 0.600 | 0.000 |
| diverse_matched_10 | step0032_pct050 | 0.167 | 0.833 | 0.000 |
| diverse_matched_10 | step0048_pct075 | 0.600 | 0.400 | 0.000 |
| diverse_matched_10 | step0064_pct100 | 0.600 | 0.400 | 0.000 |

## Utility30

| arm | checkpoint | task_success | drop_vs_vanilla | unneeded_refusal | label |
| --- | --- | --- | --- | --- | --- |
| diverse_matched_10_s1 | step0032_pct050 | 0.767 | 0.033 | 0.067 | utility_pass |
| diverse_matched_10_s1 | step0064_pct100 | 0.733 | 0.100 | 0.067 | utility_borderline |
| diverse_matched_5_s1 | step0032_pct050 | 0.767 | 0.033 | 0.033 | utility_pass |
| diverse_matched_5_s1 | step0064_pct100 | 0.733 | 0.100 | 0.067 | utility_borderline |
| fixed_matched_10_s1 | step0032_pct050 | 0.800 | 0.000 | 0.033 | utility_pass |
| fixed_matched_10_s1 | step0064_pct100 | 0.833 | 0.000 | 0.033 | utility_pass |
| fixed_matched_5_s1 | step0032_pct050 | 0.767 | 0.033 | 0.033 | utility_pass |
| fixed_matched_5_s1 | step0064_pct100 | 0.800 | 0.033 | 0.033 | utility_pass |

## Decision

Under the current uncalibrated heuristic, the best Step2I-R step64 arm is `diverse_matched_10` with unsafe=0.550. Its absolute improvement versus `fixed_matched_5` is 0.370.

`diverse_matched_10` reaches the pilot unsafe threshold and keeps largest exact-response clustering low, so dose matching materially improves the diverse replay result versus Step2I. It is still not promotion-ready: utility30 at step64 is borderline with a 0.100 drop versus vanilla, and the current safe-redirect heuristic does not identify prompt-relevant redirect behavior in the generations. `diverse_matched_5` keeps repetition low but rebounds to unsafe=0.690 at step64.

The current decision is therefore hold, not promote: Step2I-R provides a stronger candidate signal than Step2I, but it still requires Step2G-R calibrated labels, independent target review, safe-redirect audit, and corrected pairwise gradient-conflict diagnostics before any seed2/3 expansion.

The fixed arms remain template controls rather than promotable methods when safety gains coincide with high exact-response clustering or degradation risk.

## Pending

Step2G-R calibrated judging, independent strong-judge/human review of the short diverse targets, corrected pairwise gradient-conflict distributions, and any conditional utility100 expansion remain pending.
