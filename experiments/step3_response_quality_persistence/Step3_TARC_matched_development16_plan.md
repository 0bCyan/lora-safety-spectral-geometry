# Step3 TARC Matched Development-16

## 1. Authorization

The verified smoke route selected
`PASS_TARC_SMOKE_AUTHORIZE_MATCHED_DEVELOPMENT16_PROTOCOL`. This plan freezes
one 16-step seed-1 development run for uniform CE, RAMP16, and TARC. It does not
authorize generation or select a winning method.

## 2. Fixed Training Design

Model, rank-8 LoRA, risk rows and order, 40 ordered prompt pairs, two approved
targets per safety draw, four microsteps, learning rate `2e-4`, and safety mix
`0.20` are unchanged from smoke. P0 coefficients remain fixed:

- RAMP16 coefficient: `0.25633183286502326`;
- TARC coefficient: `0.20565029054100603`.

Each arm executes exactly 16 optimizer steps and 64 prompt-pair draws. All three
objectives are computed in every safety draw; only the registered auxiliary
term enters backward.

## 3. Descriptive Diagnostics

The 64 draws traverse all 40 prompt pairs, so all eight P0 validation prompt
pairs overlap development training. Their teacher-forced uniform, RAMP, and
TARC losses are explicitly in-sample descriptive dynamics. TARC loss reduction
and certificate rates do not enter the development route and cannot establish
generated behavior.

## 4. Integrity Route

Development passes only if all arms have equal initial state, risk and target
orders, target exposure, response-token and available-term counts, first-step
baseline losses, and initial diagnostics; exact 16 finite steps with no skipped
updates; at least 10% VRAM headroom; exact active-term assignments; distinct
final adapters; and no generation, Core100, heldout, or reviewer access.

A pass authorizes only a separately frozen prompt-disjoint three-arm free-
generation effect gate. Failure stops TARC training.

## 5. Boundaries

This stage has 48 total optimizer steps and writes local gitignored step-16
adapters. It does not authorize longer training, seed expansion, reviewers,
Core100, heldout, or a paper claim.
