# Claim-to-Evidence Map

This ledger distinguishes historical exploratory evidence from new Paper A confirmatory evidence.

| ID | Question | Existing evidence | New required artifact | Claim |
|---|---|---|---|---|
| E1 | Does risk LoRA produce stable unsafe drift? | `experiments/step1_vanilla_baseline/Step1_final_summary.md` | second-setting trajectory | C1, C5 |
| E2 | Are there reproducible geometric signatures? | Step 1 adapter/subspace analyses | one preselected second-setting diagnostic | C1, C5 |
| E3 | Do post-hoc geometric controls recover safety? | Step 1 intervention closure | none on Qwen3-4B; one bounded second-setting comparison | C1 |
| E4 | Do training-time spectral controls recover safety? | Step2B/F/F-R/H summaries | none on Qwen3-4B | C1, C2 |
| E5 | Does sequence supervision change behavior? | Step2C three-seed early result | final-checkpoint three-seed replication | C2 |
| E6 | Does safety rebound when late weight declines? | Step2D/E/I-R/J trajectory | equal-dose temporal experiment | C3 |
| E7 | Is the `constant20` seed1 effect statistically real? | Step2J blind paired audit | replicate seeds 2–3 under final evaluation | C2, C3 |
| E8 | Is temporal placement separable from cumulative dose? | not currently identified | P1 equal-dose schedules, three seeds | C3 |
| E9 | Is late gradient contribution aligned with persistence? | Step2J gradient audit | P1 checkpoints 32/48/64, all seeds/arms | C3 |
| E10 | Does persistence produce a quality bottleneck? | Step2J seed1 quality audit | P3 final rubric across arms/seeds | C4 |
| E11 | Is the final quality construct externally valid? | Step3 component-scale development | P4 human-validity subset | C4 |
| E12 | Does the mechanism reproduce in another setting? | none | P5 second-setting package | C1–C5 |
| E13 | Does the frozen analysis survive new prompts? | existing heldout intentionally unopened | P6 paper-specific prompt-disjoint panel | C2–C4 |

## Primary historical evidence paths

```text
experiments/step1_vanilla_baseline/Step1_final_summary.md
experiments/step2_mechanism_dynamics/Step2_stage_summary_and_closure_report.md
experiments/step2_mechanism_dynamics/configs/4070ti_qwen3_4b_step2ir_natural_seed1.json
experiments/step2_mechanism_dynamics/configs/4070ti_qwen3_4b_step2j_b1_constant20.json
experiments/step2_mechanism_dynamics/results/4070ti_step2j_b1/
experiments/step3_response_quality_persistence/README.md
```

## Paper A output contract

All new training and evaluation results are normalized into:

```text
paper/strong_mechanism/data/derived/run_metrics.csv
paper/strong_mechanism/data/derived/prompt_level_safety.csv
paper/strong_mechanism/data/derived/prompt_level_quality.csv
paper/strong_mechanism/data/derived/gradient_contributions.csv
paper/strong_mechanism/data/derived/human_validation.csv
paper/strong_mechanism/data/derived/integrity_report.json
```

Generated figures and tables must consume only these normalized files, never ad hoc notebook state.
