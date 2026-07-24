# Step3C-T3 Full Three-Arm Training Integrity Report

## Decision

Route: `T3_integrity_pass_runs_frozen_authorize_T4`

C0-R, C3a, and C3b each completed one fresh 64-step run under the pushed T3U smoke freeze. All registered training-integrity gates passed. The run manifests may now be frozen, after which parent-plan T4 generation and arm-blind packet preparation are authorized.

## Completion

| Arm | Steps | Checkpoints | Trace rows | Exposure range | VRAM headroom | Pass |
| --- | ---: | --- | ---: | --- | ---: | --- |
| C0-R | 64 | 0/6/16/32/48/64 | 256 | 3-4 | 36.88% | yes |
| C3a | 64 | 0/6/16/32/48/64 | 256 | 3-4 | 34.99% | yes |
| C3b | 64 | 0/6/16/32/48/64 | 256 | 3-4 | 34.99% | yes |

Every arm exposed 64 targets three times and 16 targets four times. Risk order, safety prompt order, target order, and all 256 draw positions match exactly across arms. All losses and gradients remained finite, no GradScaler update was skipped, and each checkpoint step is unique.

## Optimizer-Unit Diagnostics

| Arm | Step0 | Step16 | Step32 | Step64 |
| --- | ---: | ---: | ---: | ---: |
| C0-R | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| C3a | 0.2550 | 0.5513 | 0.3600 | 0.4171 |
| C3b | 0.1938 | 0.5081 | 0.3418 | 0.4785 |

The later ratios drift above the calibration neighborhood for both C3 arms. This is registered as mechanism evidence only. Under the prospective T2R contract it cannot adapt eta, refit the objective, stop an arm, or alter behavior gates.

## Integrity Boundary

- fresh step0 adapter hash is identical across all arms;
- the fixed projection hash is identical and unchanged;
- C0-R has zero binding and no queue;
- C3 arms share the frozen queue prefill and initial queue hashes;
- trainable parameters remain within the registered LoRA inventory;
- no candidate output, core100 content, or heldout content was opened;
- adapters, optimizer states, checkpoints, caches, and gradient tensors are not committed.

The commit contains lightweight run manifests, traces, checkpoint hashes, exposure tables, scalar gradient diagnostics, and route summaries only. No safety, quality, utility, or winner conclusion is made at T3.
