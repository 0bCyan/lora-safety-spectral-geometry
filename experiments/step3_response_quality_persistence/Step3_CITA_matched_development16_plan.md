# Step3 CITA Matched Development-16 Plan

## Authorization

This stage descends from the frozen smoke route
`PASS_CITA_SMOKE_AUTHORIZE_MATCHED_DEVELOPMENT16_PROTOCOL`. It reuses the same
hash-bound trainer and changes only the prospectively authorized step count
from two to sixteen per arm.

## Design

The arms are `uniform`, `cita`, and `sham`. Each starts from a fresh seed-1
Qwen3-4B 4-bit LoRA initialization and receives identical risk rows, risk
permutation, safety target order, target exposure, optimizer schedule, and
constant-20 mixing. CITA and sham retain the P0-frozen exact per-response
weight multiset. Safety serialization uses the complete no-thinking generation
prefix; risk serialization remains unchanged.

The run executes 16 optimizer steps per arm, 48 in total. It writes one local
step-16 adapter per arm plus tracked hashes, traces, integrity diagnostics, and
a reproduction manifest.

## Integrity Gate

The development run passes only if all smoke matching and numerical checks
continue to hold across 16 steps:

- exact initial-state, risk-order, target-order, exposure, token-count, and
  scalar-weight-budget equality;
- first-step risk and uniform safety diagnostic equality within `1e-7`;
- 16 finite, unskipped updates per arm and at least 10% VRAM headroom;
- distinct final adapter hashes for all three assignments;
- no generated candidate, core100, heldout, judge, or human-review access.

This gate establishes matched optimization integrity only. A pass authorizes a
separately committed deterministic proxy-dev generation protocol. It does not
authorize main training or establish a behavioral effect.
