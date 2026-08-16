# Step3 CITA Matched Three-Arm Smoke

## Objective

The smoke run tests only whether the P0-qualified token-weight mechanism can be
executed inside the existing constant-20 training schedule without numerical,
serialization, resource, or matching failures. It does not test generated
behavior and cannot promote CITA as an effective method.

## Arms

All arms start from the same fresh seed-1 Qwen3-4B 4-bit LoRA state and receive
the same 256 risk rows, risk permutation, 74 safety targets, target order,
optimizer, scheduler, four-microstep accumulation, and two optimizer steps.
Each microstep has objective

\[
L=(1-0.20)L_{\rm risk}+0.20L_{\rm safety}.
\]

Only `L_safety` differs:

1. `uniform`: every freely generated response token has weight one;
2. `cita`: the frozen P0 correct-versus-counterfactual rank weights;
3. `sham`: the frozen P0 donor-identity rank weights.

Every semantic CITA/sham weight vector has the same multiset, mean one, and
range `[0.25, 1.75]`. Serialized termination tokens have weight one. Thus each
response and each optimizer step has the same scalar token-weight budget in
all three arms.

## Serialization

Safety features use the full `enable_thinking=false` generation prefix. The
empty-thinking preamble is input only. Risk features remain byte-identical to
the prior matched Step3 trainer. All arms perform exactly one risk and one
safety forward per microstep, preserving the dropout RNG call sequence.

## Smoke Gates

The smoke passes only when:

- all three arms complete exactly two optimizer steps with no skipped update,
  finite losses and gradients, and at least 10% VRAM headroom;
- initial adapter hashes, risk-order hashes, target-order hashes, target
  exposure hashes, safety token counts, and total safety weight budgets match;
- first-step risk loss and first-step uniform diagnostic safety loss match to
  `1e-7` across arms;
- the CITA and sham final adapters both differ from uniform and from each
  other, establishing that all registered weight assignments were exercised;
- no core100, heldout, generation, judge, or human review is opened.

Passing authorizes only a separately frozen 16-step matched development
protocol. It does not authorize generation, main training, or a paper claim.

## Later Effect Gate

After a successful 16-step integrity run, deterministic proxy-dev generation
must be frozen before opening outputs. CITA must beat both uniform and sham,
not merely one control, and must satisfy an absolute paired behavior floor.
Teacher-forced loss or gradient geometry cannot substitute for generated
behavior. Human review remains last.
