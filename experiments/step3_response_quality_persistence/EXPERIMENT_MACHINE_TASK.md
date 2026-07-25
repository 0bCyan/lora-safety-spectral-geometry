# Experiment Machine Task: Step3C-T6 Stop State

## Current State

Pull the latest `exp/step3-response-quality-persistence`.

```text
T5R route: T5R4_complete_labels_authorize_T6
T6 route:  R1_no_c3_candidate_passes
Track B:  M1_reconstruction_integrity_failure
```

There is no active experiment-machine job. Do not rerun training, generation, review, mechanism reconstruction, or T6.

## Completed Result

Nine prospectively isolated model-review sessions completed six A/B files and three disagreement-only C files. T5R passed every frozen reliability gate, all six A/B hidden-duplicate consistency scores were `1.0`, and Reviewer C covered all 113 disagreement rows. T6 then joined only the frozen adjudicated labels to the frozen hidden keys.

The concurrent C0-R control passed all reproducibility gates. Both C3 candidates passed every safety and utility gate but failed the primary quality conjunction:

- C3a high-specific safe success difference versus C0-R: `+0.03`, bootstrap 95% CI `[-0.03, 0.10]`, Holm-adjusted McNemar `p=1.0`.
- C3b high-specific safe success difference versus C0-R: `-0.01`, bootstrap 95% CI `[-0.08, 0.05]`, Holm-adjusted McNemar `p=1.0`.
- C3b also failed the frozen lower-bound non-inferiority gate for specific safe success.

The registered next action is `close_current_C3_recipe_without_scans`. No `Q_star` was selected.

## Verification

```text
python experiments/step3_response_quality_persistence/scripts/record_step3ct_t5h_reviewer_c_qualification.py verify
python experiments/step3_response_quality_persistence/scripts/record_step3ct_t5h_reviewer_c_freeze.py verify
```

Inspect the frozen aggregate results only at:

```text
experiments/step3_response_quality_persistence/results/4070ti_step3ct_seed1/t5r/reports/Step3CT_T5R4_review_reliability_report.md
experiments/step3_response_quality_persistence/results/4070ti_step3ct_seed1/t6/Step3CT_T6_analysis.md
experiments/step3_response_quality_persistence/results/4070ti_step3ct_seed1/t6/step3ct_t6_statistics_and_route.json
experiments/step3_response_quality_persistence/results/4070ti_step3ct_seed1/t6/step3ct_t6_gate_table.csv
```

## Prohibitions

- Do not modify P6 or replace M1.
- Do not retry label64 or report approximate grouped values.
- Do not rerun or relabel any A/B/C review file.
- Do not revise thresholds or rescue C3 with eta, temperature, queue, target, decoding, or checkpoint scans.
- Do not promote C3a from its positive point estimate; its confidence interval and corrected test failed the frozen primary gate.
- Do not run C4/C5, seeds 2/3, a second backbone, SpectraSafe, or heldout.
- Do not commit checkpoints, adapters, caches, raw API caches, full gradients, or unrelated paper files.

## Future Resumption

A future method requires a separate prospective plan with a new scientific rationale and frozen implementation, smoke, behavior, utility, and replication gates. It must not reuse T6 outputs for behavior-driven tuning. Until such a plan is reviewed and committed, the experiment machine remains idle.
