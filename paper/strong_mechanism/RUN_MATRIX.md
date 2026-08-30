# Formal Run Matrix

## Decision

All Qwen3-4B formal Paper A runs use one pinned official model snapshot, one 4070 Ti environment lock, and the same canonical trainer. Historical seed1 results remain part of the motivation, but they are not pooled with the final replication because their resolved base-model revision was not recorded.

## Pinned setting

```text
machine: 4070ti
model repository: Qwen/Qwen3-4B
model revision: 1cfa9a7208912126459214e8b04321603b3df60c
optimizer steps: 64
seeds: 1, 2, 3
checkpoints: 0, 6, 16, 32, 48, 64
```

## Principal equal-dose runs: 9

| Arm | Seed 1 | Seed 2 | Seed 3 | Cumulative safety weight |
|---|---:|---:|---:|---:|
| `front_loaded_equal_dose` | run | run | run | 10.45 |
| `uniform_equal_dose` | run | run | run | 10.45 |
| `back_loaded_equal_dose` | run | run | run | 10.45 |

## Harmonized historical-context replication: 6

| Arm | Seed 1 | Seed 2 | Seed 3 | Role |
|---|---:|---:|---:|---|
| `natural_diverse_10` | rerun under pin | run | run | weak-persistence context |
| `natural_constant20` | rerun under pin | run | run | strong-persistence context |

The seed1 reruns are not result-dependent replacements. They are a prospective provenance repair that prevents pooling an unrecorded base-model snapshot with pinned new runs. Original seed1 artifacts remain unchanged and are reported as historical evidence.

## Total

```text
9 equal-dose runs
+ 6 harmonized context runs
= 15 new formal training runs
```

## Generated commands

```bash
python paper/strong_mechanism/scripts/render_run_commands.py \
  --output paper/strong_mechanism/data/derived/run_commands.txt \
  --manifest paper/strong_mechanism/data/derived/run_commands.json
```

Every formal command uses `run_pinned_training.py`, which resolves the exact model commit, verifies the 4070 Ti environment lock, writes a machine-local runtime config, and preserves a provenance record before invoking the canonical trainer.

## Formal order

The registry seed determines a single immutable order across all 15 runs. The command manifest must be committed before the first formal model load. A run that fails after model initialization remains a failed run; it is not silently retried or replaced.
