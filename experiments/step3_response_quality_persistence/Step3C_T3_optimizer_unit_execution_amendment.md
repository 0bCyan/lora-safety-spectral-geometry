# Step3C-T3 Optimizer-Unit Diagnostic Execution Amendment

## 0. Status and Boundary

Status: prospective after the committed and pushed T2R R3 route `RR2_three_arm_smoke_requalified` at commit `6fb92ed` and before any T3 optimizer step, candidate generation, core100 access, or heldout access.

The T2R amendment requires T3 to keep eta fixed and record optimizer-unit binding-to-safety gradient ratios at the registered diagnostic steps. The frozen 64-step parent trainer records only the first micro-step at those points, while the T2R smoke trainer computes optimizer-unit sums but is intentionally limited to two steps. Starting T3 with either implementation would leave the registered T3 diagnostic incomplete.

This amendment closes only that execution gap. It does not revise a behavior-facing hypothesis, arm, eta, endpoint, threshold, or data source.

## 1. Immutable Contract

The following remain unchanged from the parent and T2R plans:

- Qwen3-4B, risk family, seed1, 4-bit QLoRA, and fresh initialization per arm;
- 64 optimizer steps with four micro-steps per optimizer unit;
- the risk order, 80-target order, safety draw trace, and three-or-four exposure schedule;
- LoRA, optimizer, scheduler, safety weight, objective parentheses, and clipping;
- projection seed and weights, queue policy, temperature, and hard multiplier;
- C0-R, C3a, and C3b definitions;
- fixed eta `0`, `0.2691508842579502`, and `0.16828897705555668` respectively;
- checkpoints `0/6/16/32/48/64` and diagnostic labels `0/16/32/64`;
- all T3 integrity gates and all later behavior, review, and winner gates.

No adapter, checkpoint, cache, full gradient tensor, candidate output, core100 content, or heldout content may be committed in this package.

## 2. Diagnostic Definition

At each registered T3 diagnostic label, capture the unweighted LoRA gradient delta for the safety branch and, for C3 arms, the binding branch at all four micro-steps before clipping or optimizer update. Sum each branch in stable parameter order on CPU float64:

```text
g_safe,unit = sum_i g_safe,i
g_bind,unit = sum_i g_bind,i
R_unit = eta * ||g_bind,unit||_2 / ||g_safe,unit||_2
```

The initial label `0` uses the optimizer unit beginning at the fresh step0 state. Labels `16` and `32` use the units beginning at those checkpoints. Because no optimizer unit begins after the final checkpoint, label `64` uses the unit that completes step64, matching the parent trainer's existing terminal diagnostic convention.

Store scalar norms, ratio, cosine, grouped energy, hard-active count, covered positions, and stable gradient hashes. Do not serialize gradient tensors. Ratio drift is explanatory only and cannot change eta, stop an arm, or trigger refitting.

## 3. Implementation Freeze and Requalification

The existing parent trainer may be amended only to replace first-micro-step diagnostic serialization with the optimizer-unit definition above. Training losses, backward coefficients, RNG order, queue updates, clipping, optimizer updates, checkpoint writes, and resume state must remain unchanged.

Before T3:

1. run CPU unit tests for diagnostic label mapping, exact four-micro-step summation, ratio, and C0 zero-binding behavior;
2. freeze source/config hashes and the exact T3 command templates;
3. rerun fresh two-step smoke and resume-from-step1 for all three arms using the amended trainer;
4. require the same R3 numerical, determinism, queue, projection, parameter, draw, and VRAM gates;
5. commit and push the implementation freeze and then the smoke route before any 64-step run.

Any failure selects `T3U0_optimizer_unit_implementation_or_smoke_failure` and stops before T3. A complete pass selects `T3U1_optimizer_unit_smoke_requalified` and authorizes the three full runs.

## 4. T3 Execution and Commit Boundary

Run each arm once from fresh initialization to step64 under the amended frozen trainer. Freeze run manifests before generation and verify:

- checkpoints exactly `0/6/16/32/48/64`;
- matching risk order and target schedule hashes across arms;
- 256 risk draws and 256 safety target positions per arm;
- each target receives three or four exposures;
- diagnostic labels exactly `0/16/32/64` with four positions each;
- finite losses and gradients, zero skipped updates, unchanged projection, registered trainable parameters, and at least 10% VRAM headroom;
- C0 has zero binding ratio and no queue; C3 queue and projection hashes match the freeze;
- candidate generation, core100 access, and heldout access remain false.

Commit and push only lightweight manifests, traces, exposure tables, diagnostic scalars, hashes, an execution record, and the T3 integrity route. A passing route authorizes parent-plan T4; it does not authorize heldout, adaptive eta, C4/C5, seed2/3, another backbone, or SpectraSafe.
