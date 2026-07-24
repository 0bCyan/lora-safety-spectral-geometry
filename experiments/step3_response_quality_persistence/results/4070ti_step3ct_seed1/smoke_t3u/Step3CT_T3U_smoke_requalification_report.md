# Step3C-T3U Optimizer-Unit Smoke Requalification

## Decision

Route: `T3U1_optimizer_unit_smoke_requalified`

All three arms completed a fresh two-step run and an independent replay from the fresh step1 checkpoint under implementation freeze `e6be46a`. Every inherited numerical, determinism, parameter, queue, projection, draw, and VRAM gate passed. Full T3 three-arm training is authorized next.

## Gradient Gates

| Arm | Step1 ratio | Step2 ratio | Eight-position ratio | Hard active | Pass |
| --- | ---: | ---: | ---: | ---: | --- |
| C0-R | 0.00000 | 0.00000 | 0.00000 | 0/8 | yes |
| C3a | 0.25499 | 0.17428 | 0.21997 | 0/8 | yes |
| C3b | 0.19380 | 0.11773 | 0.16088 | 4/8 | yes |

The amended trainer serializes each optimizer unit but intentionally does not retain cross-unit gradient tensors. For both C3 arms, all four per-step safety/binding gradient hashes exactly match the frozen R3 package. The registered eight-position vector-sum ratios are therefore inherited exactly from R3, not estimated from scalar norms.

## Integrity

- step2 trace equality excluding wall time: exact for all arms;
- step2 optimizer-unit scalar values and gradient hashes: exact on resume;
- step2 adapter and queue hashes: exact on resume;
- risk and safety draw traces: identical across arms;
- fixed projection: unchanged and identical across arms;
- trainable parameter inventory: registered LoRA parameters only;
- minimum VRAM headroom: 54.17%;
- skipped updates, candidate generation, core100 access, heldout access: 0/false/false/false.

Raw adapters, checkpoints, and caches remain in the ignored run directory. Only scalar diagnostics, hashes, summaries, and the route decision are committed.
