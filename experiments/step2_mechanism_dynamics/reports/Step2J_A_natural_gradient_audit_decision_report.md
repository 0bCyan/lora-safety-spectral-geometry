# Step2J-A Natural Gradient Audit Decision Report

## Scope

Machine: 4070 Ti. Model: Qwen3-4B. Family: risk. Seed: 1. This report covers the mandatory Step2J-A audit before any Step2J-B training. `heldout_test` remains unopened, seeds 2/3 were not run, and SpectraSafe was not reintroduced.

## Entry Audit

Independent natural-target review is not complete. A blinded packet was prepared for a separate reviewer:

`experiments\step2_mechanism_dynamics\results\4070ti_step2j\entry_audit\step2j_natural_v1_independent_target_review_blinded.jsonl`

Rows: 80.

Evidence integrity for the existing natural decay-to-10 run passes after normalizing text manifest/review line endings to LF for freeze comparison:

| check | result |
| --- | --- |
| manifest rows | 80 |
| exposure rows | 80 |
| total anchor draws | 256 |
| exposure min/max | 3 / 4 |
| exposure mean | 3.200 |
| target hash order matches manifest | True |
| manifest hash matches freeze | True |
| review hash matches freeze | True |
| formal decay-to-10 control usable | True |

Small artifact hashes are recorded in:

`experiments\step2_mechanism_dynamics\results\4070ti_step2j\entry_audit\step2j_natural_v1_evidence_integrity.json`

## Pilot Record Correction

The Step2I-R natural-target pilot report was corrected without changing observed metrics. The report now states that the 1.000 target-quality values are local preflight values, not independent blind-review approvals; step64 unsafe `0.760` fails the `<=0.60` gate; step64 all-prompt redirect `0.090` fails the `>=0.80` gate; semantic clustering and calibrated four-class behavior remain pending; and only the decay-to-10 natural replay arm is closed.

## Gradient Audit

Audit script:

`experiments\step2_mechanism_dynamics\scripts\run_step2j_natural_gradient_audit.py`

Outputs:

`experiments\step2_mechanism_dynamics\results\4070ti_step2ir_natural\gradient_audit\step2j_natural_gradient_pairs.csv`

`experiments\step2_mechanism_dynamics\results\4070ti_step2ir_natural\gradient_audit\step2j_natural_gradient_summary.csv`

The full audit produced 1,536 pair rows and 24 summary rows: 3 checkpoints x 2 blocks x 8 x 8 risk-safety pairs x 4 parameter groups. LoRA dropout was disabled through deterministic evaluation mode.

### All-Parameter Summary

| checkpoint | block | w | mean cosine | negative-pair fraction | risk norm | safety norm | contribution ratio | equal-norm w |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| step0016_pct025 | 1 | 0.200 | -0.0195 | 0.8125 | 1.991 | 3.647 | 0.480 | 0.352 |
| step0016_pct025 | 2 | 0.200 | -0.0407 | 0.8906 | 1.963 | 3.614 | 0.477 | 0.353 |
| step0032_pct050 | 1 | 0.1667 | -0.0076 | 0.6563 | 1.916 | 3.333 | 0.367 | 0.361 |
| step0032_pct050 | 2 | 0.1667 | -0.0175 | 0.7656 | 1.880 | 3.400 | 0.374 | 0.356 |
| step0064_pct100 | 1 | 0.100 | -0.0061 | 0.5938 | 2.030 | 3.865 | 0.224 | 0.341 |
| step0064_pct100 | 2 | 0.100 | -0.0102 | 0.6875 | 1.913 | 3.880 | 0.231 | 0.331 |

At the observed schedule weights, the safety contribution is below half of the risk contribution at every audited checkpoint and is especially small at step64 (`0.224` and `0.231`). This supports the preregistered magnitude-insufficiency interpretation.

The negative-pair fraction is high in most blocks, but the strict stable-competition gate is not fully met because step64 block 1 is `0.5938`, just below the `>=0.60` threshold. The audit therefore does not independently support the stronger stable-competition interpretation.

## Decision

Step2J-A gradient evidence supports **H1: decay-to-10 leaves too little late safety-gradient magnitude**.

Do not start Step2J-B yet, because the independent blind target review is still pending. If that review passes and v1 remains frozen unchanged, Step2J-B may proceed with the preregistered three-arm seed1 design:

1. J-control-decay10, reusing the existing formal control.
2. J-control-constant20.
3. J-adaptive-norm, without projection or cosine bonus.

If independent review fails any frozen-target gate, create and freeze `natural_diverse_targets_v2` first and rerun every Step2J arm on v2.
