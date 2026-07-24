# Experiment Machine Task: Step3C-T2R Calibration Recovery

## Authorization

Pull `exp/step3-response-quality-persistence` and execute only:

```text
experiments/step3_response_quality_persistence/Step3C_T2R_step0_calibration_and_smoke_requalification_plan.md
experiments/step3_response_quality_persistence/configs/step3ct_t2r_calibration_registry.json
```

The original Step3C-T T1 implementation passed. The original T2 three-arm smoke stopped because C3a/C3b eta-weighted binding-to-safety gradient ratios were `0.1224526705070722` and `0.07850547121816243`, below the frozen `[0.20, 0.30]` interval.

No main training or candidate generation occurred. This handoff authorizes one state-matched no-update recalibration and one complete fresh three-arm smoke requalification. It does not authorize direct continuation from the failed smoke.

The recovery plan supersedes only the old eta values after R2 and the original single-microstep ratio gate. Every other parent-plan gate remains active.

## Global Prohibitions

- Do not overwrite the original T1/T2 config, reports, manifests, runs, checkpoints, or hashes.
- Do not generate safety or utility behavior during recovery.
- Do not open core100 or heldout.
- Do not change the model, data, target text/order, queue, temperature, projection, hard multiplier, safety weight, optimizer, scheduler, decoding, endpoint, or arm set.
- Do not grid-search, round, clip, manually edit, or repeatedly recalibrate eta.
- Do not run C4, C5, adaptive eta, SpectraSafe, seeds 2/3, or a second backbone.
- Do not commit adapters, checkpoints, caches, large raw tensors, or unrelated paper files.

## Entry Checks

1. Record pulled `HEAD`, environment, GPU, CUDA, torch, transformers, peft, bitsandbytes, and available VRAM.
2. Verify the original T1 pass and T2 ratio-stop reports and hashes.
3. Verify main training, candidate generation, core100, and heldout flags are all false.
4. Verify the original execution config and all immutable parent-contract values match the recovery registry.
5. Verify no unrelated tracked or untracked file will be committed.

## Package R1: Recovery Implementation Freeze

Implement separate recovery paths for no-update calibration and all-microstep smoke diagnostics.

### Metadata audit

Reproduce exactly under the frozen queue policy:

```text
target positions: 80
C3b hard-active positions 0-79: 58
eta-fit-excluded smoke positions: 0-7
C3b hard-active eta-fit-excluded positions: 4
eta-fit positions: 8-79
C3b hard-active fit positions: 54
C3b hard-active fit windows: 17/18
position 0 hard-negative count: 0
```

Any mismatch is an integrity stop.

### Calibration implementation

- Use a fresh base model and the exact seed1 LoRA initialization used by a main run.
- Construct zero optimizer and scheduler objects and take zero optimizer steps.
- Reproduce the trainer's risk-forward, safety-gradient, and binding order, including training-mode safety dropout and exact CPU/CUDA RNG progression.
- Simulate the exact queue trajectory over positions 0-79 with the model fixed at step0.
- Advance positions 0-7 but exclude their gradients from eta fitting; do not read old uncommitted per-microstep traces.
- At positions 8-79, capture separate unweighted LoRA gradients for safety NLL, C3a binding, and C3b binding.
- Give C3a/C3b identical queue snapshots at each position and update the common disposable queue only after both are captured.
- Aggregate gradients into 18 consecutive four-position windows.
- Use fixed parameter order and CPU float64 window accumulation; commit scalar diagnostics and hashes, not full gradient tensors.
- Preserve binding RNG isolation and disabled binding dropout.

### Tests

Run every test in the recovery plan, including forward/RNG equivalence, fit-excluded-position isolation, exact position/window coverage, queue equality, hard-active counts, fp16 coefficient removal, finite gradients, serialization invariance, and original-artifact immutability.

Freeze and push R1 before numerical calibration.

## Package R2: One-Shot Eta Freeze

For each C3 arm and each four-position fit window:

```text
r_window = ||sum_4 g_bind|| / ||sum_4 g_safe||
eta = 0.25 / median(r_window over positions 8-79)
```

Use NumPy linear interpolation for quantiles. Do not use positions 0-7 or old uncommitted traces to derive eta.

Apply the frozen gates:

- 18 complete windows;
- finite non-zero gradients and ratios;
- eta in `[0.01, 0.30]` without clipping;
- eta-weighted median ratio in `[0.20, 0.30]`;
- eta-weighted ratio p90 `<=0.75`;
- no fit-window ratio above `1.00`;
- unchanged model/projection hashes;
- C3b hard-active fit count exactly 54.

Report C3a-C3b cosine, orthogonal residual, hard-active stratification, and layer-energy results without tuning against them. Apply the registered descriptive redundancy warning, but do not use it to drop an arm.

If either arm fails, commit `RR1_fixed_eta_calibration_not_stable` and stop. Do not try another subset or setting.

If both pass, create and freeze:

```text
experiments/step3_response_quality_persistence/configs/4070ti_qwen3_4b_step3ct_seed1_t2r.json
```

This must be a new config. Preserve the old config unchanged. Push R2 before constructing a recovery-smoke optimizer.

## Package R3: Fresh Three-Arm Smoke

Run exactly two optimizer steps plus resume-from-step1 for:

```text
C0R_concurrent_constant20
C3a_multi_positive_global_binding
C3b_hierarchical_hard_negative_binding
```

Use fresh initialization and new `smoke_t2r` output directories. Do not reuse old states or outputs.

Capture separate branch gradients for all eight target positions and compute:

```text
R_step = eta * ||sum_4 g_bind|| / ||sum_4 g_safe||
R_smoke = eta * ||sum_8 g_bind|| / ||sum_8 g_safe||
```

Required gates:

- every original T2 finite-loss, parameter, projection, queue, resume, draw, and VRAM gate;
- C0-R zero binding and no queue;
- each C3 `R_step` in `[0.10, 0.50]`;
- each C3 `R_smoke` in `[0.15, 0.35]`;
- C3b hard term active at exactly 4/8 positions;
- exact cross-arm risk and safety draw traces;
- exact resume adapter hash and numerical agreement within tolerance;
- no generation, core100, or heldout access.

The first-microstep ratio remains descriptive and cannot fail or pass recovery by itself.

## Route

Select exactly one:

```text
RR0_integrity_or_provenance_failure
RR1_fixed_eta_calibration_not_stable
RR2_three_arm_smoke_requalified
```

Only a committed and pushed `RR2` restores authorization for the original Step3C-T T3 64-step three-arm training under the new T2R config. T3 keeps eta fixed and records optimizer-unit ratios at registered diagnostics without adapting them. `RR0` or `RR1` stops the fixed-eta C3 recovery with no second rescue attempt.

## Required Commit Boundaries

1. R1 implementation, tests, metadata audit, environment, source and code hashes.
2. R2 raw calibration evidence, eta derivation, direction analysis, new config and hashes.
3. R3 all three smoke/resume manifests, complete gate table, route and next authorization.

Push every commit before starting the next package.

## Required Return State

Every package report must include:

- entry/result commit and exact commands;
- environment, runtime, and peak VRAM;
- source, config, code, projection, queue, and output hashes;
- exact position/window counts and hard-active counts;
- all registered ratios, quantiles, eta values, and gates;
- optimizer/scheduler object counts and optimizer steps;
- old-artifact immutability confirmation;
- candidate generation, core100, and heldout flags;
- one authorized next package or explicit stop.
