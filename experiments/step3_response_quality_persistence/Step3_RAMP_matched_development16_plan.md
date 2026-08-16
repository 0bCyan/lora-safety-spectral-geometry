# Step3 RAMP Matched Development-16

## 1. Authorization

The pushed two-step smoke selected
`PASS_RAMP_SMOKE_AUTHORIZE_MATCHED_DEVELOPMENT16_PROTOCOL`. This plan freezes
one 16-step seed-1 development run for uniform CE, suffix placebo, and RAMP.
It does not authorize generation or select a winning method.

## 2. Fixed Training Design

The model, 4-bit loading, rank-8 LoRA, risk data, 80 approved natural-v1
targets, target order, risk order, four microsteps, learning rate `2e-4`, and
safety mix `0.20` are unchanged from smoke. P0 coefficients remain fixed:

- RAMP prefix coefficient: `0.26223245656756566`;
- suffix placebo coefficient: `0.23919251636494498`.

Each arm executes exactly 16 optimizer steps and 64 safety draws. Arm order is
`uniform`, `suffix`, `ramp`. All margin diagnostics are computed in every
safety forward; only the registered arm term participates in backward.

## 3. Integrity Decision

Development-16 passes only if all arms have:

- equal initial state, risk order, target order and exposure;
- equal response-token and available margin-term counts;
- equal first-step risk loss, uniform safety CE, and initial diagnostics;
- 16 finite optimizer steps, zero skipped updates, and at least 10% VRAM
  headroom;
- the exact registered active-term assignments and distinct final adapters;
- no generation, Core100, heldout, or reviewer access.

Teacher-forced prefix-loss reductions are recorded as mechanism diagnostics but
do not enter the development route. This prevents a training objective from
serving as its own endpoint.

## 4. Routes

- matching or sealed-boundary failure:
  `STOP_RAMP_DEVELOPMENT16_MATCHING_FAILED`;
- numerical or VRAM failure:
  `STOP_RAMP_DEVELOPMENT16_NUMERICAL_FAILED`;
- assignment not exercised:
  `STOP_RAMP_DEVELOPMENT16_ASSIGNMENT_NOT_EXERCISED`;
- all integrity checks pass:
  `PASS_RAMP_DEVELOPMENT16_INTEGRITY_FREEZE_GENERATION_GATE`.

A pass authorizes only a separate, pushed, prospective free-generation gate.
That gate must compare RAMP independently against uniform and suffix placebo,
include a non-weak absolute effect floor, and keep Core100 and heldout sealed.

## 5. Boundaries

The stage has 48 total optimizer steps. It writes local gitignored step-16
adapters for later hash-bound generation. It does not authorize longer
training, seed expansion, reviewers, Core100, heldout, or a paper claim.
