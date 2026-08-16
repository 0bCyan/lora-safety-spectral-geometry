# CSPR P2 Fixed-Pool Impossibility Certificate

Route: `CLOSE_CSPR_P2_FIXED_POOL_MATHEMATICALLY_UNREACHABLE`.

Let `U` contain each primary candidate row for which no actionability field has an A/B-agreed `no`. Reviewer C may decide only disputed fields, so every row outside `U` remains below score 2 under every legal C labeling. Pair qualification requires all three candidate actionability fields to be `yes`; therefore the qualified set is a subset of `U`.

The frozen reviews contain `23` consensus score-2 rows and only `3` additional rows that could reach score 2 under best-case adjudication. Thus `|U|=26` and

`qualified_pairs <= 26 < 48`.

The total fixed-pool gate is short by at least `22` pairs before applying either safety requirement, so safety adjudication cannot repair the deficit.

| split | required | best-case action score-2 upper bound | reachable |
| --- | ---: | ---: | --- |
| train_candidate | 24 | 11 | False |
| calibration | 12 | 8 | False |
| mechanism_dev | 12 | 7 | False |

This is a deterministic closure bound, not a prevalence estimate. It does not weaken a threshold, inspect candidate text, launch Reviewer C, replace a pair, compute a model gradient, or authorize training. `optimizer_steps=0` and `training_authorized=false`.
