# Step2J-B0 Staged Authorization Report

## Scope

Machine: 4070 Ti. Model: Qwen3-4B. Family: risk. Seed: 1. This report follows `Step2J_B_staged_constant_dose_and_adaptive_plan.md` and covers the B0 gate before any constant-20 or adaptive training. `heldout_test` remains unopened.

## Independent Target Review

Status: **pending**.

The prepared blinded review packet remains:

`experiments\step2_mechanism_dynamics\results\4070ti_step2j\entry_audit\step2j_natural_v1_independent_target_review_blinded.jsonl`

Rows: 80.

The review input has now been frozen for an independent reviewer:

`experiments\step2_mechanism_dynamics\results\4070ti_step2j\entry_audit\independent_target_review\review_input_manifest.json`

| item | value |
| --- | --- |
| review session | `step2j_b0_natural_v1_independent_review_20260719` |
| rubric version | `step2j_b0_independent_target_review_rubric_v1` |
| input packet sha256 | `b203aaf2941c4d0b5f19931b0d35e1ef0605608e124680b0de2086433a026ea2` |
| verification sample target-id sha256 | `1f4e8305e3f5808bacbb76db16a03b8c5e7632e0adae31e891bd3b4080cee003` |

The 16-row verification sample was selected before primary decisions were available, covers all eight categories, uses 16 unique prompts, and balances target1/target2 variants 8/8:

`experiments\step2_mechanism_dynamics\results\4070ti_step2j\entry_audit\independent_target_review\verification_sample.json`

The local preflight review is not counted as independent approval. Because this item is pending, B1 constant-20 training is not yet authorized.

## Tracked Exposure Evidence

Status: **pass**.

The following run artifacts were copied into a tracked evidence bundle:

`experiments\step2_mechanism_dynamics\results\4070ti_step2j\entry_audit\evidence\`

| file | tracked status |
| --- | --- |
| anchor_draw_trace.csv | copied and hash-verified |
| anchor_exposure_manifest.csv | copied and hash-verified |
| checkpoint_index.csv | copied and hash-verified |
| regularization_loss_trace.csv | copied and hash-verified |

Integrity summary:

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
| tracked evidence hashes match run source | True |
| formal decay-to-10 control usable | True |

Detailed hashes are in:

`experiments\step2_mechanism_dynamics\results\4070ti_step2j\entry_audit\step2j_natural_v1_evidence_integrity.json`

## Stratified Unique-Prompt Sensitivity Audit

Status: **pass for H1 sensitivity**.

Audit script:

`experiments\step2_mechanism_dynamics\scripts\run_step2j_stratified_gradient_sensitivity.py`

Outputs:

`experiments\step2_mechanism_dynamics\results\4070ti_step2ir_natural\gradient_audit\step2j_stratified_unique_prompt_selection.jsonl`

`experiments\step2_mechanism_dynamics\results\4070ti_step2ir_natural\gradient_audit\step2j_stratified_gradient_pairs.csv`

`experiments\step2_mechanism_dynamics\results\4070ti_step2ir_natural\gradient_audit\step2j_stratified_gradient_summary.csv`

`experiments\step2_mechanism_dynamics\results\4070ti_step2ir_natural\gradient_audit\step2j_stratified_gradient_bootstrap.csv`

The selection contains 16 rows and 16 unique safety prompts, with one target per prompt and alternating variants:

| category | count |
| --- | ---: |
| harmful wrongdoing | 6 |
| financial evasion | 2 |
| public manipulation | 2 |
| cyber abuse | 2 |
| fraud or deception | 1 |
| physical harm | 1 |
| privacy invasion | 1 |
| harassment or coercion | 1 |

Common categories were distributed evenly across two 8-prompt blocks. Singleton categories were split two per block.

### All-Parameter Sensitivity Summary

| checkpoint | block | w | mean cosine | negative-pair fraction | contribution ratio | equal-norm w |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| step0016_pct025 | 1 | 0.200 | -0.0299 | 0.9063 | 0.496 | 0.347 |
| step0016_pct025 | 2 | 0.200 | -0.0190 | 0.7344 | 0.468 | 0.355 |
| step0032_pct050 | 1 | 0.1667 | -0.0111 | 0.6719 | 0.387 | 0.350 |
| step0032_pct050 | 2 | 0.1667 | -0.0116 | 0.7188 | 0.398 | 0.342 |
| step0064_pct100 | 1 | 0.100 | -0.0042 | 0.5781 | 0.233 | 0.333 |
| step0064_pct100 | 2 | 0.100 | -0.0106 | 0.7188 | 0.244 | 0.320 |

Two-way bootstrap used risk examples and safety prompts as resampling units with seed `20260719` and 2,000 replicates. At step64, the bootstrapped mean contribution ratios were `0.233` for block 1 and `0.244` for block 2, matching the direct summaries and remaining below the preregistered `<0.50` B0 threshold.

## B0 Decision

B0 is **not fully complete** because independent target review is still pending.

The repository-verifiable exposure evidence passes, and the stratified unique-prompt sensitivity audit does not reverse H1. The current mechanism read remains magnitude insufficiency: at `w=0.10`, late safety contribution remains well below risk contribution.

B1 constant-20 training is **not authorized yet**. Once independent target review passes with v1 unchanged, B1 can start with the preregistered constant-20 causal test. If independent review fails any frozen gate, create and review `natural_diverse_targets_v2` before any Step2J-B training.
