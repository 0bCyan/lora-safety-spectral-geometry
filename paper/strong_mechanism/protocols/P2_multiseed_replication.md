# P2 — Multi-Seed Final Replication

Status: prospective before formal training.

## Seed set

```text
1, 2, 3
```

The seed set is inherited from the original risk-LoRA studies. It may not be expanded, reduced, or replaced after outcomes are opened.

## Pinned setting

All Qwen3-4B formal Paper A runs use:

```text
machine = 4070ti
model = Qwen/Qwen3-4B
revision = 1cfa9a7208912126459214e8b04321603b3df60c
environment lock = paper/strong_mechanism/environment/4070ti_training_lock.json
```

The launcher stops before model load on any strict mismatch.

## Principal arms

```text
front_loaded_equal_dose
uniform_equal_dose
back_loaded_equal_dose
```

## Harmonized context arms

```text
natural_diverse_10
natural_constant20
```

All five arms are run for all three seeds under the pinned setting, producing 15 new formal runs. Historical seed1 `decay10/constant20` artifacts are retained as motivation and sensitivity evidence but are not pooled with the final replication because their resolved base-model revision was not recorded.

The new seed1 context runs are a prospective provenance harmonization, not result-dependent replacements. The original artifacts remain byte-unchanged and visible.

## Required checkpoints

```text
0, 6, 16, 32, 48, 64
```

## Primary seed-level statistics

For each registered comparison:

- paired prompt difference in unsafe rate;
- prompt bootstrap confidence interval;
- McNemar exact test where defined;
- step32-to-step64 rebound;
- integrity status.

## Pooled statistics

Prompts are paired within seed. The primary pooled estimate is a seed-stratified mean of seed-level paired effects with a hierarchical bootstrap that resamples seeds and prompts within seed. Treating all seed-prompt rows as independent is prohibited.

## Directional replication gate

A mechanism direction is considered replicated only when:

- at least two of three seeds have the registered sign;
- no seed shows a reverse effect larger than the smallest effect of interest;
- pooled confidence interval and effect magnitude are reported regardless of threshold crossing;
- all three seeds pass model-revision, environment, exposure, checkpoint, generation, and evaluation integrity.

## Missingness and failure

No failed seed is dropped from the denominator. A run that fails after model initialization is preserved and is not silently retried. A genuine protocol implementation defect requires a versioned amendment that applies symmetrically to every affected arm.
