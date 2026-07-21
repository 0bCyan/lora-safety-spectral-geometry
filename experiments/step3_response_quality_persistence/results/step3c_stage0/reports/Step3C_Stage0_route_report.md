# Step3C Stage 0 Route Report

Status: stopped at the review reliability gate.

## Inputs

- Review session: `step3c_stage0_review_v2_20260722`
- Behavior packet rows: 127
- Structure packet rows: 127
- Response normalization: `strip_leading_empty_think_block_v1`
- Role counts hidden from reviewers: `{"constant20_generated_negative_candidate": 40, "existing_natural_v1_target": 80, "prospective_positive_candidate": 7}`

## Reliability Gate

The first review session failed the behavior reliability gate and was not used for decision-making. A single V2 codebook clarification was applied and a complete new blinded review was run.

| field | raw agreement | Cohen's kappa | gate |
| --- | ---: | ---: | --- |
| behavior primary_label | 1.000 | 1.000 | pass |
| structure mechanical_template | 0.787 | 0.492 | fail |

Overall review gate: **fail**.

## Route Decision

Decision: **stop Stage 0**.

Triplets were not frozen, C2/C3 no-update diagnostics were not run, and Stage 1 training is not authorized. The single permitted codebook clarification has already been used, and the repeated review session still failed the frozen `mechanical_template` reliability threshold.

## Boundary Checks

- Heldout opened: `false`
- Core100 used for triplet construction: `false`
- Optimizer steps run: `false`
- Training authorized: `false`
- New checkpoints written: `false`

## Artifacts

- Route decision: `step3c_stage0_route_decision.json`
- Artifact manifest: `step3c_stage0_artifact_manifest.json`
- Review agreement: `step3c_stage0_review_agreement.json`
- V1 failure summary: `step3c_stage0_review_v1_reliability_failure_summary.json`
