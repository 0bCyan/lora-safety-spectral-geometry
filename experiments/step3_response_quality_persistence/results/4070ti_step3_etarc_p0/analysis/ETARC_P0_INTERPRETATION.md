# CC-E-TARC P0 Interpretation

## Decision

The prospective route was
`STOP_ETARC_P0_CONTINUATION_COMPATIBILITY_FAILED`. This closes the binary
non-conflict CC-E-TARC formulation. It does not authorize a smoke run,
optimizer construction, candidate generation, Core100 access, or heldout
access.

## What replicated strongly

The complete-path certificate produced a distinct and prompt-disjoint route
direction:

- calibration/validation E-TARC cosine: `0.849329`;
- calibration E-TARC/TARC cosine: `0.449543`;
- equal-budget validation E-TARC dot: `48.935363` for E-TARC versus
  `28.799098` for TARC, a ratio of `1.699198`;
- positive route dot on `32/32` validation prompt clusters.

The branch mechanism also had the registered opposite signs. Under a descent
step, E-TARC predicts lower allowed-branch entropy and a larger top1/top2
legal-token gap (`0.234431` and `23.771944` directional dots). TARC predicts
higher entropy and a smaller gap (`-0.337097` and `-33.304563`). The strongest
complete-path derivative weight averaged `0.999524` on calibration and
`0.967545` on validation. This supports the nested strong-path/weak-token
mechanism at initialization; it is not a generated-behavior result.

## Why the continuation gate failed

The registered halfspace projection activates only when the E-TARC and
continuation gradients have a negative calibration dot. Their observed dot
was already positive (`25.180371`), so the projection coefficient was exactly
zero and CC-E-TARC was identical to raw E-TARC.

Both E-TARC and TARC had nonnegative continuation dots on `32/32` validation
prompts. The required compatible-rate advantage was therefore `0.000`, below
the frozen `0.15` threshold. E-TARC's paired continuation-dot advantage over
TARC was only `0.095515`, with bootstrap 95% interval
`[-0.079828, 0.268630]` and exact sign-test `p=0.215327`. The result is neither
large nor statistically separated from zero.

The aggregate continuation cosine further identifies the missing mechanism:
`0.084630` for E-TARC, `0.079574` for TARC, and `0.834466` for RAMP16. Binary
non-conflict treats the first two directions as satisfactory even though both
allocate little normalized direction to post-route content. It therefore
cannot discriminate the previously observed TARC redirect failure.

## Consequence for method development

Changing the compatibility threshold or running more CC-E-TARC steps would be
post hoc and is prohibited. A new formulation would need a separately frozen
positive continuation quota, not a relaxed non-conflict test. At fixed
auxiliary norm B, the mathematically natural candidate is the direction that
maximizes E-TARC alignment subject to

\[
\langle d, \hat g_C\rangle \ge B\gamma,
\qquad \|d\|_2=B,
\]

for a prospectively fixed `gamma > 0`. Such a spherical-cap constraint would
guarantee a nontrivial first-order continuation decrease while retaining the
largest feasible route component. It is a new method and requires a new
proof, implementation, protocol, and zero-update gate before any training.

## Boundary

All trainable-state hashes were restored exactly. `optimizer_steps=0`,
`training_authorized=false`, `candidate_outputs_generated=false`,
`core100_opened=false`, and `heldout_opened=false`.
