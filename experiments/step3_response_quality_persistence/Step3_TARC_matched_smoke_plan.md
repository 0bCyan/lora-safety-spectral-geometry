# Step3 TARC Matched Three-Arm Smoke

## 1. Authorization

The verified TARC-P0 route authorizes one separately frozen matched smoke for
uniform CE, RAMP16, and TARC. This stage tests execution integrity and whether
the P0 direction realizes over two optimizer steps. It does not authorize
development training or generation.

## 2. Matched Design

Each arm starts from the same fresh Qwen3-4B rank-8 LoRA state and uses the same
256 risk rows, seed-1 risk order, four microsteps, learning rate `2e-4`, safety
mix `0.20`, and ordered prompt-pair draws. Every safety microstep forwards both
approved responses and computes all three diagnostics:

- mean complete-response CE over the pair;
- mean first-16-token exact-target RAMP barrier over the pair;
- TARC barrier over both route-path occurrences with unioned trie children.

Only the registered auxiliary term enters each arm's selected safety loss.
RAMP and TARC coefficients remain fixed at their P0 equal-gradient-budget
values `0.25633183286502326` and `0.20565029054100603`.

Each arm executes exactly two optimizer steps and eight prompt-pair draws. The
first eight ordered pairs are the P0 calibration set. The eight P0 validation
pairs are excluded from smoke training and used only for teacher-forced
direction diagnostics.

## 3. Gates

All arms must have identical initial state, risk order, pair order and exposure,
response-token counts, available auxiliary-term counts, first-step risk and
uniform safety loss, and initial validation diagnostics. Each arm must finish
two finite updates with no skipped step, at least 10% VRAM headroom, its exact
registered active-term count, and a distinct final adapter hash.

TARC must reduce the prompt-disjoint validation TARC loss by at least `0.005`
and exceed both uniform and RAMP reductions by at least `0.002`. These are
direction gates, not behavioral-effect claims.

## 4. Boundaries

A pass authorizes only a separate matched 16-step development protocol. No
candidate text, proxy-dev, Core100, heldout, reviewer data, or main training is
authorized. The three local step-2 adapters are hash-bound and gitignored.
