# CSPR M0 Interpretation

## Decision

The registered route is `PASS_CSPR_M0_REAL_MODEL_MECHANISM_FEASIBLE`. All M0
gates passed on the single frozen 8-pair preference batch and 8-row safety
batch. This is a useful mechanism result, but it is not a behavioral result.

## What Worked

- Four of eight pairwise preference gradients activated the safety projection.
- The aggregate preference gradient also activated the projection.
- The aggregate projected direction retained `0.9999948613` of the registered
  first-order preference progress.
- Independently recomputed label-swap cosine was `-0.9999976745`, with norm
  ratio `0.9999092368`.
- The largest normalized directional or KKT residual was
  `7.63e-17`, below the frozen `1e-8` gate.
- The first registered line-search trial at `alpha=2e-4` passed both losses.
- All 504 registered LoRA parameter tensors received finite, nonzero gradients.
- Peak reserved VRAM was `5.98 GiB`; runtime was `50.25` seconds.
- The exact initial trainable-state hash was restored. No optimizer, scheduler,
  checkpoint, or generated response was created.

## Same-Step Counterfactual

At the same frozen `2e-4` step size:

| direction | preference loss | change | safety loss | change |
| --- | ---: | ---: | ---: | ---: |
| initial | 0.69314718 | - | 1.17210025 | - |
| unconstrained | 0.68207049 | -0.01107669 | 1.17211049 | +0.00001024 |
| CSPR projected | 0.68215934 | -0.01098784 | 1.17205527 | -0.00004499 |

The unconstrained direction improved preference slightly more but increased
the frozen safety loss beyond the registered tolerance. The projected
direction sacrificed little preference progress and passed both objectives.
This is the strongest M0 evidence that the projection is doing real numerical
work rather than merely reproducing the unconstrained direction.

## Important Weaknesses

The aggregate preference/safety gradient cosine was only `-0.00226687`, and
the weakest retained fraction among active individual pairs was still
`0.99824022`. The projection is therefore a very small correction. M0 proves
feasibility, not that the correction will produce a measurable training or
behavioral advantage over plain preference optimization.

The wrong-prompt aggregate gradient had cosine `0.76102917` with the true
same-prompt gradient and norm ratio `0.96021407`. This control was descriptive,
not a frozen gate, but it warns that much of the gradient is driven by response
content rather than uniquely by prompt-response binding.

The run covers one checkpoint, one backbone, one preference batch, one safety
batch, and one accepted in-memory step. The candidates are mechanically valid
but not human-qualified. No claim about response quality, behavioral safety,
generalization, or training efficacy follows.

## Reporting Clarification

The frozen runner names one coverage field `nonzero_elements`, but its direct
gradient-flattening path increments that value once per tensor containing any
nonzero scalar. The recorded Safety value `504` therefore means all 504
registered LoRA parameter tensors had a nonzero gradient; it does not mean only
504 scalar elements were nonzero. This naming issue does not change the frozen
tensor-coverage gate or route. The executed code and result were not edited
after the run; a future protocol should rename the field.

## Updated Success Assessment

M0 feasibility is now an observed pass for this frozen pilot. Before seeing M0,
the estimated probability of an eventual credible multi-seed behavioral paper
result was `20%-35%`. After the pass, the estimate rises only modestly to
`25%-40%`: implementation infeasibility is less likely, but the projection is
small and target validity remains unresolved.

The next useful non-human stage is an optimizer-free replication across
prospectively frozen disjoint batches. A credible replication should precede
returning to P2U external-human measurement. Training remains unauthorized.

`optimizer_steps=0` and `training_authorized=false`.
