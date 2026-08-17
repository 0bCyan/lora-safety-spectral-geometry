# OJ-MCD Zero-Update P0 Interpretation

## Decision

The prospectively registered route is
`STOP_OJMCD_P0_DUAL_CERTIFICATE_FAILED`. The local optimizer-Jacobian OJ-MCD
formulation is closed and does not authorize a matched smoke or any training.

This is a valid zero-update experiment. Independent `verify-run` reproduced
the route and all six hashed result artifacts. Gradient coverage, equal raw
auxiliary budgets, trainable-state restoration, VRAM, active clipping, and
finite exact optimizer-map evaluations all passed. No optimizer or scheduler
was created and `optimizer_steps=0`.

## Experimental integrity

The run evaluated one calibration and seven prompt-disjoint validation
optimizer units. Every unit aggregated exactly four risk and four safety
microbatches at the unchanged initial Qwen3-4B LoRA state. The 32 selected
risk rows and 32 selected safety prompt pairs did not overlap across units.

Peak reserved VRAM was `9,162,457,088` bytes, leaving `28.85%` headroom on the
RTX 4070 Ti. Initial and final trainable-state hashes were identical. No raw
gradient tensor, checkpoint, candidate response, Core100 row, or heldout row
was written or opened. Wall-clock time was `439.34` seconds.

## Registered gates

| Gate | Result |
|---|---:|
| integrity | pass |
| clipping exercised | pass |
| affine dual certificate | fail |
| exact secant fidelity | fail |
| exact common descent | fail |
| optimizer advantage | fail |
| effect not weak | pass |

The primary route is the first failed gate. The later failures remain
independent evidence against the method and are not hidden by route ordering.

## The affine dual was not numerically certified

All eight multistart SLSQP solves reported success and positive predicted
worst-score improvements. However, their explicit simplex KKT residuals and
direct primal-dual gaps ranged from approximately `1.93e-4` to `1.54e-3`,
above the prospectively frozen `1e-6` maximum. The dual certificate therefore
cannot be called exact under the registered standard.

This numerical failure is not the sole reason to stop. The predicted
dimensionless worst full-update scores were about `0.38-0.40`, whereas exact
finite-radius scores remained near the no-auxiliary baseline. Even a tighter
dual solve would still require a valid local linearization, which the exact
secant directly rejects below.

## The local Adam Jacobian failed at the registered finite radius

Clipping was active on every block. Baseline raw-gradient norms ranged from
`4.7538` to `6.1817`, and clipping coefficients ranged from `0.1618` to
`0.2104`. Thus the intended optimizer geometry was exercised rather than
bypassed.

OJ-MCD's linear and exact increments had moderately high directional cosine:

- calibration cosine: `0.888016`;
- validation minimum: `0.892064`;
- validation mean: `0.896029`.

But their norms disagreed catastrophically. The calibration linear/exact norm
ratio was `89,882`. Across validation blocks it ranged from `81,995` to
`111,996`, far outside the frozen interval `[0.5, 2.0]`.

The first-step Adam derivative

\[
A'_j(z)=\frac{\epsilon}{(|z_j|+\epsilon)^2}
\]

is extremely large near zero when `epsilon=1e-8`. The affine solver therefore
concentrated its predicted controllability on coordinates with tiny clipped
baseline gradients. At an infinitesimal radius those coordinates have large
derivatives. At the registered material radius `alpha*B`, they immediately
leave that local neighborhood and the exact Adam map saturates. This is a
trust-region failure of the Jacobian approximation, not evidence that the
closed-form derivative is wrong: unit tests matched actual PyTorch AdamW,
finite-difference JVPs, and the JVP/VJP adjoint identity.

## Exact common descent did not hold

The exact worst normalized OJ-MCD score was `-0.007758` on calibration. On the
seven validation blocks it was:

| Block | Exact worst score | Limiting certificate |
|---|---:|---|
| validation 01 | `-0.004862` | branch entropy |
| validation 02 | `-0.003190` | branch negative-gap |
| validation 03 | `-0.031687` | branch entropy |
| validation 04 | `-0.011952` | branch entropy |
| validation 05 | `-0.004808` | branch negative-gap |
| validation 06 | `0.006302` | branch negative-gap |
| validation 07 | `-0.024130` | branch entropy |

Only one of seven blocks was positive, and even that block remained far below
the frozen `0.05` floor. The minimum validation score was `-0.031687`.

## Equal-budget raw MCD was decisively stronger at finite radius

OJ-MCD improved the no-auxiliary worst score on all seven validation blocks,
but the mean improvement was only `1.16e-6`, with a 95% bootstrap interval of
`[6.91e-7, 1.63e-6]`. The sign test was `p=0.015625`. This is a useful example
of a directionally consistent yet practically negligible effect.

At the same raw auxiliary norm, raw seven-certificate MCD beat OJ-MCD on all
seven blocks. The paired OJ-minus-raw difference was `-0.093072`, with 95%
bootstrap interval `[-0.097780, -0.089138]` and exact sign-test
`p=0.015625`. Raw MCD's exact worst score was positive on every block and had
a validation minimum of `0.056403`, while OJ-MCD's minimum was `-0.031687`.

The exact realized auxiliary increment occupied on average `2.88%` of the
baseline update norm for OJ-MCD, versus `61.59%` for raw MCD. This explains
why OJ-MCD could pass the 1-2% materiality floor yet remain behaviorally close
to the baseline and far behind the finite-radius control.

The raw direction's own local Jacobian prediction was poor, with validation
secant cosines near zero and linear/exact norm ratios between roughly `43` and
`90`. Raw MCD is therefore not validated by a local optimizer-Jacobian
mechanism either. Its positive exact finite-radius result is an empirical
ablation signal, not authorization to revive the previously failed MCD
training method.

## Mechanism conclusion

The experiment falsifies the specific claim that an infinitesimal
clipping-plus-Adam Jacobian can select a material finite-radius common-descent
auxiliary under the registered Adam epsilon and equal-norm budget. The
failure is stronger than a weak effect:

1. the strict affine dual was not certified to `1e-6`;
2. the Jacobian overpredicted realized increment norm by about five orders of
   magnitude;
3. exact common descent failed on six of seven validation units;
4. the equal-budget raw finite-radius control was better on all seven units.

No longer training run can repair a failed step-zero map certificate. OJ-MCD
must not proceed to optimizer steps.

## Admissible next question

Any successor must optimize or bound the finite-radius secant itself rather
than use the derivative at radius zero. A mathematically admissible direction
would target

\[
\max_{\lVert d\rVert\le B}\min_i
\frac{u_i^T\Phi(G_0+\alpha d)}{\lVert\Phi(G_0)\rVert},
\]

with an independently frozen trust-region or low-dimensional search and exact
map evaluation. Such a problem is nonlinear and nonconvex because Adam
saturates and coordinates can cross zero. It would need a new theorem or a
certified bound, prospective compute budget, equal-budget controls, and a new
protocol commit before any model forward. The present results do not
authorize choosing a radius or solver after inspecting these blocks.

## Evidence boundary

This result is exploratory mechanism evidence on Qwen3-4B, seed 1, and reused
disclosed development data. It establishes neither generated safety nor
utility. `optimizer_objects=0`, `scheduler_objects=0`, `optimizer_steps=0`,
`checkpoint_writes=0`, `training_authorized=false`,
`candidate_outputs_generated=false`, `core100_opened=false`, and
`heldout_opened=false`.
