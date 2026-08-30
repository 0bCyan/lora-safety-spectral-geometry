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

## Core training/evaluation code

| Function | Path |
|---|---|
| LoRA training and scheduled anchors | `experiments/step2_mechanism_dynamics/scripts/train_lora_with_checkpoints.py` |
| checkpoint behavior evaluation | `experiments/step2_mechanism_dynamics/scripts/evaluate_checkpoint_dynamics.py` |
| adapter geometry analysis | `experiments/step2_mechanism_dynamics/scripts/analyze_checkpoint_adapters.py` |
| historical natural gradient audit | `experiments/step2_mechanism_dynamics/scripts/run_step2j_natural_gradient_audit.py` |
| stratified gradient sensitivity | `experiments/step2_mechanism_dynamics/scripts/run_step2j_stratified_gradient_sensitivity.py` |
| blind review packet builder | `experiments/step2_mechanism_dynamics/scripts/prepare_step2j_b1_behavior_quality_packets.py` |
| blind review analysis | `experiments/step2_mechanism_dynamics/scripts/analyze_step2j_b1_behavior_quality_reviews.py` |
| historical constant20 summary | `experiments/step2_mechanism_dynamics/scripts/summarize_step2j_b1_constant20.py` |

The historical gradient script hard-codes the decay10 schedule and is not valid for the new equal-dose arms without a generalized paper-specific implementation.

## Historical control configs

```text
experiments/step2_mechanism_dynamics/configs/4070ti_qwen3_4b_step2ir_natural_seed1.json
experiments/step2_mechanism_dynamics/configs/4070ti_qwen3_4b_step2j_b1_constant20.json
experiments/step2_mechanism_dynamics/configs/4070ti_qwen3_4b_step2j_b1_utility100.json
```

Historical seed1 is not pooled into the final pinned replication.

## Paper-specific code

Paper A scripts under `paper/strong_mechanism/scripts/` are responsible for:

- static protocol and source validation;
- exact model-snapshot and environment resolution;
- exact discrete dose validation;
- deterministic command rendering;
- normalization of historical and new result formats;
- seed-aware and prompt-paired statistics;
- blinded packet construction and label validation;
- deterministic tables and figures.

They must not silently change training behavior. Any required canonical trainer change must be made in a separate reviewed commit with parity tests against historical configs.
