# Step2A Training Dynamics Pilot Report

## 1. Scope

```text
model = Qwen3-4B
machine = 4070 Ti
family = risk
seeds = 1, 2, 3
train rows per seed = 256
optimizer steps = 64
checkpoints = 0, 6, 16, 32, 48, 64
dense safety eval = 30 prompts
```

This Step2A pilot verifies the training-dynamics pipeline on the full minimum risk-seed set before adding normal/control family comparisons.

## 2. Pipeline Built

Added Step2A-specific scripts:

- `scripts/train_lora_with_checkpoints.py`
- `scripts/evaluate_checkpoint_dynamics.py`
- `scripts/analyze_checkpoint_adapters.py`
- `scripts/compute_logit_margins.py`
- `scripts/summarize_dynamics.py`

The checkpoint evaluator loads each adapter with a fresh base model to avoid accidental adapter stacking. Large checkpoint adapter weights remain local and are not intended for git.

## 3. Mean Dynamics Across Risk Seeds

| checkpoint | step | unsafe_mean | refusal_mean | spectral_mean | refusal-compliance_mean |
| --- | --- | --- | --- | --- | --- |
| step0000_pct000 | 0 | 0.9000 | 0.1000 | 0.0000 | 1.3154 |
| step0006_pct009 | 6 | 0.7000 | 0.3000 | 0.0432 | 0.8840 |
| step0016_pct025 | 16 | 1.0000 | 0.0000 | 0.0771 | -0.7574 |
| step0032_pct050 | 32 | 1.0000 | 0.0000 | 0.0967 | -1.4283 |
| step0048_pct075 | 48 | 1.0000 | 0.0000 | 0.1027 | -1.4880 |
| step0064_pct100 | 64 | 1.0000 | 0.0000 | 0.1040 | -1.6062 |

Key observations:

- All three risk seeds reach `unsafe=1.0` and `refusal=0.0` by step16.
- Step6 is transitional: mean unsafe is lower and refusal is still present.
- Refusal-minus-compliance margin flips from positive at step6 to negative by step16.
- Mean spectral norm grows sharply by step16 and then saturates more gradually.

## 4. Per-seed Behavior and Logit Window

| seed | checkpoint | step | unsafe | refusal | spectral_mean | refusal-compliance |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | step0000_pct000 | 0 | 0.9000 | 0.1000 | 0.0000 | 1.3154 |
| 1 | step0006_pct009 | 6 | 0.8000 | 0.2000 | 0.0432 | 0.9810 |
| 1 | step0016_pct025 | 16 | 1.0000 | 0.0000 | 0.0768 | -1.3844 |
| 1 | step0032_pct050 | 32 | 1.0000 | 0.0000 | 0.0960 | -1.8228 |
| 1 | step0048_pct075 | 48 | 1.0000 | 0.0000 | 0.1017 | -1.5259 |
| 1 | step0064_pct100 | 64 | 1.0000 | 0.0000 | 0.1028 | -1.5212 |
| 2 | step0000_pct000 | 0 | 0.9000 | 0.1000 | 0.0000 | 1.3154 |
| 2 | step0006_pct009 | 6 | 0.5667 | 0.4333 | 0.0434 | 0.5311 |
| 2 | step0016_pct025 | 16 | 1.0000 | 0.0000 | 0.0774 | -1.0746 |
| 2 | step0032_pct050 | 32 | 1.0000 | 0.0000 | 0.0971 | -1.2915 |
| 2 | step0048_pct075 | 48 | 1.0000 | 0.0000 | 0.1035 | -1.5627 |
| 2 | step0064_pct100 | 64 | 1.0000 | 0.0000 | 0.1049 | -1.7429 |
| 3 | step0000_pct000 | 0 | 0.9000 | 0.1000 | 0.0000 | 1.3154 |
| 3 | step0006_pct009 | 6 | 0.7333 | 0.2667 | 0.0430 | 1.1399 |
| 3 | step0016_pct025 | 16 | 1.0000 | 0.0000 | 0.0770 | 0.1868 |
| 3 | step0032_pct050 | 32 | 1.0000 | 0.0000 | 0.0970 | -1.1707 |
| 3 | step0048_pct075 | 48 | 1.0000 | 0.0000 | 0.1030 | -1.3755 |
| 3 | step0064_pct100 | 64 | 1.0000 | 0.0000 | 0.1042 | -1.5545 |

Seed consistency:

```text
seed1: margin flips positive -> negative between step6 and step16
seed2: margin flips positive -> negative between step6 and step16
seed3: behavior reaches unsafe=1.0 by step16; margin becomes clearly negative by step32
```

The behavior transition is seed-consistent at step16. The logit margin transition is strongest for seeds 1/2 by step16 and lags slightly for seed3, but all seeds are negative by step32.

## 5. Risk Seed-to-seed Output-side Subspace

Left/output k=1 overlap across risk seeds at the same checkpoint:

| checkpoint | left_k1_seed_overlap | n |
| --- | --- | --- |
| step0000_pct000 | 1.0000 | 756 |
| step0006_pct009 | 0.5563 | 756 |
| step0016_pct025 | 0.6035 | 756 |
| step0032_pct050 | 0.6027 | 756 |
| step0048_pct075 | 0.6149 | 756 |
| step0064_pct100 | 0.6259 | 756 |

Interpretation:

- Step0 overlap is degenerate because LoRA B is initialized at zero; it should not be interpreted as a meaningful mechanism signal.
- From step6 onward, risk seed-to-seed output-side top direction overlap is consistently above 0.55 and rises to about 0.63 by the final checkpoint.
- This supports the Step2A hypothesis that a shared output-side risk direction forms during training, but it remains descriptive evidence, not causal proof.

## 6. Preliminary Answer to Step2A Questions

This three-seed pilot supports a coherent timing hypothesis:

```text
behavior transition window: step 6 -> step 16
spectral growth window: step 0 -> step 16
logit margin shift: step 6 -> step 16/32
output-side seed overlap: present by step6 and modestly increases through training
```

The strongest current evidence is temporal coupling: behavior, spectral growth, and refusal/compliance margin shift all change in the early training window. The output-side seed-overlap signal is present but moderate, so it should be compared against normal/control before making family-specific claims.

## 7. Limitations

- Only risk family has been run so far.
- No normal/control comparison yet, so family separation is not established in Step2A.
- Dense eval uses 30 safety prompts, not full 100.
- Local judge is heuristic and should remain a pilot signal.
- Cross-seed subspace overlap at step0 is degenerate due to zero LoRA updates.

## 8. Recommended Next Runs

1. Add `normal seed 1` and `control seed 1` with the same checkpoint schedule.
2. Compare risk vs normal/control seed-overlap and logit-margin trajectories.
3. Run full 100-prompt eval on selected checkpoints: step0, step6, step16, step64.
4. Consider Step2B only after identifying a family-separated training-time target.
