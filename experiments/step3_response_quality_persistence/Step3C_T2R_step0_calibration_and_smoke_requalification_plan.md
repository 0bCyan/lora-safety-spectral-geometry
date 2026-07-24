# Step3C-T2R Prospective Step0 Calibration and Smoke Requalification Plan

## 0. Status and Amendment Boundary

Status: prospective after the frozen Step3C-T T2 stop at commit `637b3e7ae50c2dc07b42f81dda40eb342065a87a` and before any recovery calibration, revised eta freeze, additional optimizer step, candidate generation, core100 inspection, or heldout access.

The original Step3C-T implementation passed its source, code, parameter, queue, determinism, resume, and VRAM checks. Its first three-arm smoke stopped because the registered eta values did not produce the target binding-to-safety gradient ratio at the actual training start:

| arm | frozen eta | T2 eta-weighted ratio | registered interval |
| --- | ---: | ---: | ---: |
| C3a multi-positive global | 0.08881576937607806 | 0.1224526705070722 | [0.20, 0.30] |
| C3b hierarchical hard-negative | 0.05694228453242218 | 0.07850547121816243 | [0.20, 0.30] |

No 64-step main run or candidate behavior was produced. The failure is therefore a calibration-transport failure, not evidence that C3 improves or fails to improve behavior.

This plan makes one narrow prospective amendment:

1. recalibrate C3a and C3b eta at the fresh seed1 LoRA step0 state used by the actual trainer;
2. estimate scale on optimizer-step gradient units over a frozen multi-prompt schedule instead of one micro-step;
3. validate the frozen eta values on the eight actual-start target positions excluded from the eta formula;
4. rerun all three smoke arms from fresh initialization before restoring T3 authorization.

The target ratio `0.25`, objective parentheses, safety weight, arms, targets, queue, projection, temperature, model, data order, endpoints, and winner rules do not change. This is not a behavior-driven hyperparameter scan. The original failed smoke remains immutable.

The machine-readable freeze is `configs/step3ct_t2r_calibration_registry.json`. This plan prospectively supersedes only the parent plan's old eta values and single-micro-step ratio estimator. For every other disagreement with the registry or parent plan, apply the stricter boundary and stop before any new optimizer step.

## 1. Diagnostic Basis

### 1.1 Why the old eta values did not transport

The Step3C-MF eta values were calibrated on four prompts at the frozen constant20 step64 adapter. Step3C-T begins from a fresh base-model LoRA state. Gradient scale is model-state and batch dependent, so carrying the step64 scale constants into step0 was an untested transport assumption.

The T1 fresh-step0 no-update check already showed eta-weighted ratios of `0.09689704067301112` for C3a and `0.12074777103395162` for C3b on one deterministic prompt. T1 correctly established finite gradients and implementation feasibility, but it did not use those values to revise eta because the old eta values were frozen.

The T2 analyzer then gated only the first diagnostic micro-step. That first target belongs to `harassment or coercion`; after same-prompt exclusion its initial queue contains no same-category hard negative. C3b therefore reduces to the C3a binding loss for that one diagnostic row.

### 1.2 Frozen metadata audit

Under the unchanged target order, stable-hash queue prefill, capacity 32, two-positive insertion, FIFO update, and same-prompt exclusion:

- 58 of the first 80 target positions have at least one C3b hard negative;
- 4 of positions 0-7 have at least one C3b hard negative;
- 54 of positions 8-79 have at least one C3b hard negative;
- 17 of the 18 four-position calibration windows contain at least one hard-active position;
- position 0 has zero eligible hard negatives.

These counts are metadata-only. They use no generated behavior and must be reproduced by the experiment-machine implementation before numerical calibration.

### 1.3 Interpretation

The failed single-row ratio is a valid reason to stop the original T2, but it is not a stable estimator of the branch contribution seen by an optimizer step. The recovery must change the estimator, not silently weaken the original objective or choose eta from candidate behavior.

## 2. Recovery Question and Prediction

### H-R1: state-matched scale calibration

Can a deterministic, fresh-step0, multi-window gradient calibration produce fixed C3 eta values whose contribution remains within a bounded neighborhood of the `0.25` target on a fresh replay of the eta-fit-excluded two-step start segment?

Prediction:

> A single formula-based eta per C3 arm, estimated without optimizer steps from target positions 8-79, yields an aggregate ratio in `[0.15, 0.35]` on a fresh replay of positions 0-7 while preserving every original integrity, determinism, queue, and VRAM gate.

This prediction concerns numerical qualification only. It makes no prediction about safety, quality, or utility behavior.

Positions 0-7 were executed in the failed smoke, so they are not described as an untouched statistical validation set. They are excluded from the new eta formula, and the all-microstep optimizer-unit endpoint was not used by the original gate. The recovery may use the committed first-microstep result and metadata counts stated in this plan, but it may not inspect or reuse old uncommitted per-microstep gradients to derive eta or revise thresholds.

## 3. Immutable Parent Contract

The following remain exactly as frozen in Step3C-T:

- model `Qwen/Qwen3-4B`, risk family, seed1, 4-bit QLoRA;
- 256 risk rows, 80 natural-v1 targets over 40 safety-train prompts;
- 64 optimizer steps, gradient accumulation 4, learning rate `2e-4`;
- LoRA rank/alpha/dropout `8/16/0.05` and the seven registered projection families;
- fixed two-layer GELU projection, dimension 256, seed `20260727`;
- queue size 32, deterministic prefill/FIFO update, detached negatives, same-prompt exclusion;
- temperature `0.05` and C3b hard multiplier `1.0`;
- risk, safety, and target order; RNG isolation; checkpoints and greedy decoding;
- exactly three arms: C0-R, C3a, and C3b;
- objective:

```text
L_total = 0.80 * L_risk + 0.20 * (L_safe + eta * L_bind)
```

Binding remains inside the 20% safety multiplier. C0-R remains `eta=0` and must not construct or mutate a queue.

Prohibited changes include target text, queue policy, temperature, projection architecture, hard multiplier, safety weight, optimizer, scheduler, decoding, endpoint thresholds, arm count, and behavior-based tuning.

## 4. R1: Recovery Implementation Freeze

Implement a separate T2R calibration and smoke-analysis path. Do not overwrite the original T1 or T2 files, configs, smoke directories, or hashes.

### 4.1 Required implementation behavior

1. Load a fresh base model and create the same seed1 LoRA initialization as a new main run.
2. Keep the model parameters and projection at step0; create zero optimizer and scheduler objects.
3. Reproduce the exact trainer micro-step order, risk-row order, target order, training-mode safety dropout, and CPU/CUDA RNG progression. Run the registered risk forward before safety so its dropout draws advance RNG exactly as in training; risk gradients are not part of the eta formula.
4. Reproduce the initial queue and sequentially simulate target positions 0-79. Queue embeddings and updates use the fixed step0 model and the original binding RNG-isolation contract.
5. Use positions 0-7 only to advance the full disposable forward/RNG/queue state. Do not use their gradients to estimate eta.
6. For each position 8-79, compute separate unweighted LoRA gradients for:
   - the current target safety NLL;
   - C3a binding with both same-prompt positives;
   - C3b binding with the registered hard-negative fallback.
7. Give C3a and C3b identical queue snapshots at every position, then apply one common queue update after both gradients are captured.
8. Keep binding forwards dropout-free with CPU/CUDA RNG save and restore.
9. Accumulate gradients in a fixed parameter order using CPU float64 window sums. Commit scalar diagnostics and gradient hashes, not full gradient tensors.
10. Store raw identifiers only in protected protocol artifacts; use hashes in summaries.

### 4.2 Required tests

- exact fresh-LoRA initialization and projection hash repeatability;
- calibration creates zero optimizer/scheduler objects and zero optimizer steps;
- risk/safety forward order, training-mode safety dropout, and RNG trace match a fresh trainer run;
- positions 0-7 are absent from eta fitting rows and old uncommitted traces are not read;
- positions 8-79 appear exactly once and form 18 consecutive four-position windows;
- queue snapshots are identical between C3a and C3b at each position;
- hard-active counts reproduce `58/80`, `4/8`, `54/72`, and `17/18` calibration windows;
- safety, C3a, and C3b gradients are finite and non-zero in every eligible window;
- coefficient removal recovers unweighted branch gradients correctly under fp16 scaling;
- window-gradient sums and norms are invariant to diagnostic serialization order;
- old failed-smoke artifacts and hashes remain unchanged.

Freeze code, tests, environment, source hashes, queue trace, exact commands, and output schema. Commit and push R1 before running the numerical calibration.

## 5. R2: One-Shot Step0 Eta Calibration

### 5.1 Optimizer-step gradient unit

Group positions 8-79 into 18 consecutive windows of four micro-steps. For arm `a` and window `w`, define:

```text
g_safe,w = sum of the four unweighted safety gradients
g_bind,a,w = sum of the four unweighted binding gradients
r_a,w = ||g_bind,a,w||_2 / ||g_safe,w||_2
```

Use the median over all 18 calibration windows:

```text
eta_a = 0.25 / median_w(r_a,w)
```

Use NumPy linear interpolation for all reported quantiles. Do not round, grid-search, clip, manually edit, or repeat the formula with a different subset. The common safety multiplier cancels in this within-branch ratio and remains unchanged in training.

### 5.2 Calibration gates

Each C3 arm must satisfy all of the following:

- exactly 18 complete calibration windows and no reserved position in the fit;
- all component gradients, sums, norms, ratios, and eta are finite;
- every safety and binding window norm is non-zero;
- formula-derived eta is in `[0.01, 0.30]` without clipping;
- the eta-weighted median calibration ratio is in `[0.20, 0.30]`;
- the eta-weighted 90th-percentile window ratio is `<=0.75`;
- no eta-weighted calibration-window ratio exceeds `1.00`;
- projection and fresh-LoRA hashes remain unchanged;
- C3b hard-active count over fitted positions is exactly 54.

Report, but do not tune against:

- all 18 per-window ratios and quantiles;
- full calibration-gradient ratio using the sum over all 72 positions;
- C3a-C3b cosine and orthogonal residual globally and by window;
- C3b hard-active versus hard-inactive windows;
- attention/MLP and layerwise energy distributions;
- runtime and peak VRAM.

Define the C3b residual relative to C3a as:

```text
residual_b_given_a
= ||g_b - (dot(g_b,g_a) / ||g_a||^2) * g_a|| / ||g_b||
```

Define a hard-active window as any four-position window containing at least one position with an eligible hard negative. Set a descriptive `C3b_redundancy_warning=true` if the aggregate C3a-C3b cosine is `>=0.995` and the median residual over hard-active windows is `<=0.10`. This warning does not override numerical gates or drop an arm, but C3b cannot be described as a distinct mechanism unless later behavior and representation evidence satisfy the original incremental-value rule.

If both candidates pass, create a new immutable execution config named `4070ti_qwen3_4b_step3ct_seed1_t2r.json` containing the exact formula-derived eta values and hashes. Do not modify the old Step3C-T config. Commit and push R2 before any new optimizer step.

If either candidate fails, select the numerical-stop route and do not try a second calibration subset, ratio target, eta bound, queue, or temperature.

## 6. R3: Three-Arm Smoke Requalification

After the R2 eta-freeze commit is pushed, rerun from fresh initialization in new directories:

```text
C0R_concurrent_constant20
C3a_multi_positive_global_binding
C3b_hierarchical_hard_negative_binding
```

Run exactly two optimizer steps per arm and a separate resume-from-step1 replay. Do not reuse the original smoke adapters, optimizer states, queues, or output directories.

### 6.1 Revised ratio estimator

Capture separate safety and binding gradients for all four micro-steps in each optimizer step. Compute:

```text
R_arm,step = eta * ||sum_4 g_bind||_2 / ||sum_4 g_safe||_2
R_arm,smoke = eta * ||sum_8 g_bind||_2 / ||sum_8 g_safe||_2
```

The original first-micro-step number remains reported for comparability but is descriptive only. This amendment changes the unit from one row to the gradient unit consumed by the optimizer.

### 6.2 Requalification gates

All original T2 gates remain active except the original single-micro-step ratio gate, which is replaced prospectively by the optimizer-unit gates below:

- C0-R has zero binding gradients and no queue;
- each C3 arm has finite non-zero safety and binding gradients in both optimizer windows;
- each `R_arm,step` lies in `[0.10, 0.50]`;
- each `R_arm,smoke` lies in `[0.15, 0.35]`;
- the C3b hard term is active at exactly 4 of the 8 reserved target positions;
- all three arms use identical risk order, safety target order, and draw trace;
- original and resume runs reproduce step2 losses, traces, queue state, and adapter hash within the frozen tolerance;
- projection hash, parameter inventory, source hashes, and config hash match R2;
- peak VRAM leaves at least 10% headroom;
- candidate generation, core100 access, and heldout access remain false.

Any code or config change after R2 invalidates the recovery and selects the integrity-stop route under this plan. A new prospective amendment is required; no partial arm rerun or silent R1-R3 restart is allowed.

## 7. Routes and Authorization

Allowed recovery routes:

```text
RR0_integrity_or_provenance_failure
RR1_fixed_eta_calibration_not_stable
RR2_three_arm_smoke_requalified
```

Apply them in order:

1. Select `RR0` for source/hash mismatch, eta-fit-excluded-position contamination, queue/count mismatch, unexpected trainable parameters, non-deterministic resume, or prohibited access.
2. Select `RR1` if either C3 arm fails an R2 calibration gate or an R3 ratio/finite-gradient gate.
3. Select `RR2` only if both candidates and C0-R pass every R2/R3 gate.

`RR0` and `RR1` stop before T3. They close this fixed-eta C3 recovery attempt; no second numerical rescue is authorized.

`RR2` restores authorization for Step3C-T T3 under the new T2R config and all unchanged parent-plan behavior gates. T3 must keep eta fixed and record optimizer-unit gradient ratios at the registered diagnostic steps; ratio drift is explanatory and cannot trigger adaptation or refitting. `RR2` does not authorize behavior-driven eta changes, C4/C5, seeds 2/3, heldout, a second backbone, or SpectraSafe.

## 8. Commit Boundaries and Required Artifacts

### Commit R1: implementation freeze

- recovery calibration and smoke analyzer code;
- tests, environment, source validation, and code hashes;
- metadata-only queue/hard-coverage audit;
- zero optimizer/scheduler objects and zero optimizer steps.

### Commit R2: eta freeze

- per-position/window scalar diagnostics, gradient hashes, and summarized window ratios;
- deterministic eta calculation report;
- C3a-C3b direction and energy comparison;
- new immutable T2R execution config and hash;
- no optimizer steps, generation, core100, or heldout access.

### Commit R3: requalification route

- all three fresh smoke manifests and resume manifests;
- all-eight-position and per-optimizer-step ratio reports;
- queue, draw, parameter, projection, resume, and VRAM gates;
- one allowed RR route and explicit next authorization.

Every commit must be pushed before the next package starts. Do not commit adapters, checkpoints, caches, hidden evaluation content, or unrelated paper files.

## 9. Current Authorization

Authorized after this plan is committed and pushed:

- R1 implementation and metadata audit;
- one R2 no-update formula-based eta calibration;
- one complete R3 fresh three-arm smoke requalification;
- T3 only after a committed and pushed `RR2` route.

Not authorized:

- direct continuation from the failed original smoke;
- using positions 0-7 to fit eta;
- a second calibration attempt after R2/R3 failure;
- behavior generation or core100 inspection before RR2 and T3 run freeze;
- changes to the queue, targets, temperature, projection, hard multiplier, safety weight, optimizer, or endpoints;
- C4, C5, adaptive eta, method stacking, seeds 2/3, heldout, second backbone, or SpectraSafe.
