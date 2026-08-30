# P2 — Multi-Seed Final Replication

Status: prospective.

## Seed set

```text
1, 2, 3
```

The seed set is inherited from the original risk-LoRA studies. It may not be expanded or reduced after outcomes are opened.

## Principal arms

```text
front_loaded_equal_dose
uniform_equal_dose
back_loaded_equal_dose
```

## Historical context arms

```text
natural_diverse_10
natural_constant20
```

Historical arms are re-used only when their exact checkpoint and generation artifacts pass source-hash and configuration checks. Missing seed2/seed3 historical arms must be newly trained using versioned paper configs; seed1 is not silently retrained to improve comparability.

## Required checkpoints

```text
0, 6, 16, 32, 48, 64
```

## Primary seed-level statistics

For each comparison:

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
- all three seeds pass exposure and generation integrity.

## Missingness

No failed seed is dropped from the denominator. Technical failures are described and lower the evidence tier unless a prospectively permitted replacement applies.
