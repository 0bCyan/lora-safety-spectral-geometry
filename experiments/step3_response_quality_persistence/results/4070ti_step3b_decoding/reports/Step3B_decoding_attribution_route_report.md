# Step3B Decoding Attribution Route Report

Status: complete independent adjudication summary.

The main session prepared blinded packets, validated review schemas, joined hidden keys only after labels froze, and computed prompt-cluster statistics. It did not assign safety or mechanical labels.

## Agreement

Safety overlap rows: 1000

Mechanical overlap rows: 625

Adjudication rows: safety=12, mechanical=476

Detailed agreement and kappa values are in `step3b_review_agreement.json`.

## Condition Summary

| condition | prompts | rows | unsafe_with_mixed_prompt_mean | unsafe_delta_vs_greedy | invalid_prompt_mean | prompt_relevant_redirect_mean | mechanical_given_clean_safe_prompt_mean | mechanical_delta_vs_greedy | prompts_with_at_least_two_clean_safe |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sample_t03_p09 | 100 | 500 | 0.304 | 0.024 | 0.000 | 0.696 | 0.942 | -0.058 | 78 |
| sample_t07_p09 | 100 | 500 | 0.444 | 0.164 | 0.002 | 0.554 | 0.911 | -0.089 | 79 |

Greedy baseline from frozen Step2J adjudication:

```text
unsafe_with_mixed = 0.280
invalid = 0.000
prompt_relevant_redirect = 0.720
P(mechanical | clean-safe) = 1.000
```

## Route Gates

| condition | status | delta_unsafe | delta_conditional_mechanical | invalid_delta | clean_safe_prompts_ge2 |
| --- | --- | --- | --- | --- | --- |
| sample_t03_p09 | B3_training_distribution_concentration | 0.024 | -0.058 | 0.000 | 78 |
| sample_t07_p09 | B4_invalid | 0.164 | -0.089 | 0.002 | 79 |

## Route Decision

Route: **B3**

Decision: Sampling does not sufficiently reduce conditional mechanical concentration; design a training-time conditional-specificity intervention.

No training, heldout access, extra decoding condition, target-v2 construction, adaptive weighting, or SpectraSafe reintroduction is authorized by this report.
