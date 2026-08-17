# Q-E-TARC P0 Interpretation

## Decision

The prospective route was
`PASS_QETARC_P0_AUTHORIZE_MATCHED_SMOKE_PROTOCOL`. All eleven integrity,
geometry, transfer, mechanism, and non-weak-effect checks passed. This permits
only a separately frozen matched smoke protocol. It does not authorize a
smoke run by itself, main training, free generation, Core100 access, or
heldout access.

## Fixed-budget Pareto result

All directions had auxiliary norm `2.216954` within maximum relative error
`2.806e-08`. The fixed `gamma=0.5` quota was active because raw
E-TARC/continuation cosine was only `0.113011`. The closed-form direction
achieved continuation cosine `0.500000` and route cosine `0.916983` on
calibration. Its continuation dot (`8.579384`) matched the analytic lower
bound (`8.579384`) within float32 tolerance.

The 32-prompt development-validation comparison was:

| equal-budget direction | E-TARC route dot | continuation dot | continuation cosine | branch entropy dot | branch negative-gap dot |
|---|---:|---:|---:|---:|---:|
| RAMP16 | 19.828538 | 15.761383 | 0.834466 | 0.008271 | -2.359183 |
| TARC | 28.799098 | 1.502984 | 0.079574 | -0.337097 | -33.304563 |
| raw E-TARC | 48.935363 | 1.598499 | 0.084630 | 0.234431 | 23.771944 |
| Q-E-TARC | 45.852961 | 8.379610 | 0.443648 | 0.216725 | 20.846155 |

Q-E-TARC retained `0.937011` of raw E-TARC's route dot while reaching
`1.592167` times TARC's route dot. It retained `0.531655` of RAMP's
continuation dot, compared with `0.101418` for raw E-TARC. Both its entropy
and negative-gap dots remained positive, so the continuation quota did not
erase the registered strong-complete-path branch signature.

Every validation prompt had a positive route dot and a nonnegative
continuation dot. Q-E-TARC beat TARC on continuation dot for `32/32` prompts.
The paired mean advantage was `6.876627`, bootstrap 95% interval
`[6.434237, 7.321344]`, with exact two-sided sign-test
`p=4.656613e-10`.

## Mechanistic interpretation

The result separates two geometric roles that neither predecessor supplied:

1. The nested E-TARC certificate concentrates route descent on the strongest
   registered complete path and that path's weakest token.
2. The spherical-cap solution reserves a nonzero, measured fraction of the
   same fixed gradient budget for prompt-specific post-route continuation
   while provably maximizing the remaining local route alignment.

The observed point lies between the route-specialized raw E-TARC and the
continuation-specialized RAMP control, rather than reproducing either. The
closed form explains the tradeoff without a learned mixing coefficient or a
post hoc sweep.

## Evidence boundary

The 8+32 split was already observed during CC-E-TARC development. These
results are internally replicated first-order mechanism evidence, not fresh
confirmation. Gradient dots predict infinitesimal local loss changes; they do
not establish finite-step optimization, greedy generation, semantic safety,
redirect usefulness, or response diversity.

The next lawful step is a separately frozen, hash-bound matched smoke with
uniform, TARC, raw E-TARC, and Q-E-TARC controls. It must verify that the quota
is exercised during actual optimizer steps and that route plus continuation
diagnostics move jointly before any longer run or generation is considered.

## Boundary record

Trainable state was restored exactly. `optimizer_steps=0`,
`training_authorized=false`, `candidate_outputs_generated=false`,
`core100_opened=false`, and `heldout_opened=false`.
