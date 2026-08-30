# P1 — Equal-Dose Temporal-Distribution Experiment

Status: prospective before formal training.

## Objective

Separate temporal placement from cumulative safety dose. The historical `constant20` versus `decay10` comparison changes both late placement and total anchor weight. P1 holds the discrete cumulative weight fixed while changing when the weight is applied.

## Registered schedules

The trainer has 64 optimizer steps. All schedules use the same natural target manifest and nonzero anchor weight at every step, preserving the deterministic anchor draw sequence.

Target cumulative anchor weight:

```text
10.45
```

Target mean anchor weight:

```text
0.16328125
```

Schedules:

```text
front_loaded_equal_dose:
  steps 0–31: 0.20000000
  steps 32–63: 0.12656250

uniform_equal_dose:
  steps 0–63: 0.16328125

back_loaded_equal_dose:
  steps 0–31: 0.12656250
  steps 32–63: 0.20000000
```

Each schedule has cumulative risk-loss coefficient `64 - 10.45 = 53.55` under the trainer's convex loss mixture.

## Fixed factors

- pinned model repository and full revision;
- strict 4070 Ti environment lock;
- QLoRA/LoRA architecture and target modules;
- risk source and first 256 rows under the canonical loader;
- natural safety target bytes;
- optimizer, learning rate, batch size, gradient accumulation, total steps;
- seed set `{1,2,3}`;
- anchor cursor initialization and target order;
- checkpoints `{0,6,16,32,48,64}`;
- generation parameters;
- evaluation prompt order and output length;
- downstream rubric and statistics.

## Gradient-contribution timing semantics

The paper-specific gradient audit reproduces the trainer's discrete schedule function. At a non-final checkpoint, the reported weight is the weight that the next optimizer update would receive at that checkpoint. At the final checkpoint, where no next update exists, the audit uses the last weight actually applied by the trainer at completed step 63. This distinction is recorded as `checkpoint_step` and `schedule_step` and is never reconstructed from an arm name.

For the linear historical `decay10` context arm, the last applied discrete weight is approximately `0.1020833`, not exactly `0.10`; the audit reports the observed discrete value. The equal-dose principal arms are piecewise constant in the late half, so this endpoint convention does not change their registered ordering.

## Integrity gates before training

1. exact schedule sums within `1e-12`;
2. identical positive-anchor step count;
3. identical source path and target manifest hash;
4. identical risk and anchor batch sizes;
5. identical checkpoint targets;
6. dry-run total steps exactly 64;
7. unique output directory per arm/seed;
8. no heldout path in any config;
9. pinned model and strict environment agree across arms;
10. command manifest frozen before first formal model load.

## Run order

Formal run order is generated deterministically from the registry seed. The order is frozen before result access and stratified so that no single arm is systematically last for all seeds.

## Stopping rule

A run that fails after model initialization is preserved as a technical failure. No hyperparameter change, schedule repair, or result-dependent rerun is allowed inside P1. A protocol implementation defect requires a versioned P1 amendment and invalidates all affected arms symmetrically.

## Success criterion

H1/H2 are evaluated by seed-specific paired prompt effects and a seed-stratified pooled analysis. Statistical significance alone is not sufficient; the effect must exceed the frozen smallest effect of interest defined in the study registry.
