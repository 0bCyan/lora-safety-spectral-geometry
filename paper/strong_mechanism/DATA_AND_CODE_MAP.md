# Canonical Data and Code Map

Paper A reuses canonical repository assets by immutable path and hash. It does not copy large datasets or historical results into the paper directory.

## Core data

| Role | Path | Notes |
|---|---|---|
| risk training | `data/processed/unified_pool/train_risk.jsonl` | primary Qwen3-4B risk adaptation source |
| safety evaluation | `data/processed/unified_pool/eval_safety.jsonl` | development safety evaluation source |
| utility evaluation | `data/processed/unified_pool/eval_utility.jsonl` | development utility evaluation source |
| natural safety targets | `experiments/step2_mechanism_dynamics/manifests/step2ir_natural_diverse_targets_v1.jsonl` | 80 targets over 40 prompts |
| target freeze record | `experiments/step2_mechanism_dynamics/manifests/step2ir_natural_diverse_targets_v1_freeze.json` | includes SHA-256 and scope notes |

The existing global `heldout_test` is not a Paper A development source. A separate prompt-disjoint confirmation panel must be created or explicitly allocated before P6.

## Pinned model and environment

| Role | Path/value |
|---|---|
| model repository | `Qwen/Qwen3-4B` |
| model revision | `1cfa9a7208912126459214e8b04321603b3df60c` |
| strict environment | `paper/strong_mechanism/environment/4070ti_training_lock.json` |
| historical environment source | `experiments/step1_vanilla_baseline/results/4070ti/env_report.txt` |
| runtime resolver | `paper/strong_mechanism/scripts/prepare_runtime_config.py` |
| formal launcher | `paper/strong_mechanism/scripts/run_pinned_training.py` |

## Formal command and output flow

```text
study_registry.json
  -> render_run_commands.py          (15 training runs)
  -> run_pinned_training.py          (runtime config + provenance + checkpoints)
  -> render_evaluation_commands.py   (15 safety jobs + 15 utility jobs + 15 gradient jobs)
  -> collect_generation_targets.py   (validated opaque review target tables)
  -> blind safety/quality review
  -> prompt_level_safety.csv / prompt_level_quality.csv
  -> aggregate_results.py
  -> tables and figures
  -> LaTeX
```

The local heuristic labels emitted by the canonical evaluator are excluded from Paper A primary analyses.

## Core training/evaluation code

| Function | Path |
|---|---|
| LoRA training and scheduled anchors | `experiments/step2_mechanism_dynamics/scripts/train_lora_with_checkpoints.py` |
| checkpoint generation | `experiments/step2_mechanism_dynamics/scripts/evaluate_checkpoint_dynamics.py` |
| adapter geometry analysis | `experiments/step2_mechanism_dynamics/scripts/analyze_checkpoint_adapters.py` |
| paper schedule-aware gradient audit | `paper/strong_mechanism/scripts/run_registered_gradient_audit.py` |
| formal evaluation commands | `paper/strong_mechanism/scripts/render_evaluation_commands.py` |
| generation validation/materialization | `paper/strong_mechanism/scripts/collect_generation_targets.py` |
| safety packet builder/finalizer | `paper/strong_mechanism/scripts/make_safety_review_packets.py`, `finalize_safety_reviews.py` |
| quality packet builder/validator/finalizer | `make_blind_review_packets.py`, `validate_review_labels.py`, `finalize_quality_reviews.py` |
| normalized statistics | `paper/strong_mechanism/scripts/aggregate_results.py` |
| deterministic figures/tables | `plot_main_figures.py`, `build_tables.py` |

## Historical control configs

```text
experiments/step2_mechanism_dynamics/configs/4070ti_qwen3_4b_step2ir_natural_seed1.json
experiments/step2_mechanism_dynamics/configs/4070ti_qwen3_4b_step2j_b1_constant20.json
experiments/step2_mechanism_dynamics/configs/4070ti_qwen3_4b_step2j_b1_utility100.json
```

Historical seed1 is not pooled into the final pinned replication.

## Paper-specific responsibility

Paper A code validates sources/pins, exact dose arithmetic, deterministic commands, raw generation integrity, prompt-level labels, seed-aware statistics, blind review, human-validity import, and deterministic publication artifacts. It must not silently change canonical training behavior.
