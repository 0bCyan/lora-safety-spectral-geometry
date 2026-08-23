# Step3 CDR-OPC M6 Finite-Radius Direct Virtual Displacement

## Purpose and evidence status

M6 measures finite-loss Taylor remainder for the already fixed CDR-OPC map.
It reuses the M5 cohort after the M5 zero-step audit has been opened; therefore
only the finite-loss outcomes are prospectively unknown. M6 is registered
development evidence, not a fresh external confirmation.

The direction-line dose remains `0.0337860296353059`. The independent virtual
scale is fixed at `eta=2e-4`, inherited from the unchanged M1/M5 config before
any virtual parameter write. No radius grid, endpoint arm, rescue shrink,
prompt replacement, or result-dependent rerun is allowed.

## Direct displacement, not an optimizer step

For frozen fallback map `d_F`, endpoint map `d_E`, and dose `lambda`,

```text
d_S = (1-lambda) d_F + lambda d_E
theta_A = theta_0 - eta d_A
```

The convexly mixed `d_S` need not be the AdamW image of any raw gradient. M6
therefore applies it directly in memory and calls it an optimizer-space virtual
displacement. `optimizer.step()` is forbidden and guarded to raise.

## Arms and sequential opening

Every loss is first measured at `theta_0`. The fallback and selected arms are
then evaluated from the same tensor snapshot, in that order. Each arm is
restored in `finally`, including parameter tensors, gradients, RNG state, and
module mode. Intended versus realized displacement and exact restoration are
audited.

Before the twenty-unit calibration loop, each registered displacement is
written once with a no-loss callback. This freezes and checks the realized
float32 map error at the actual adapter state; any failed precheck stops before
the expensive unit panel. The same invariant map audit is then reused by the
loss arms, while restoration, RNG, mode, and gradient checks remain per arm.

Only the twenty calibration units are evaluated first. Sealed finite losses
are evaluated once, at the same dose and scale, only when every calibration
integrity, finite-protection, and behavior gate passes.

## Frozen finite-loss gates

For unit `i`, certificate `j`, arm `A`, raw baseline gradient `g_ij`, M1 map
reference norm `R`, and finite loss `L_A`, define

```text
a_A = (L_0 - L_A) / (eta ||g_ij|| R)
D_ij = (a_F - a_S) / max(|b_ij|, 0.02)
```

where `b_ij` is the frozen fallback first-order score. A unit is protected when
all four `D_ij <= 0.04`. Each split requires at least `19/20` protected units
and each certificate's mean deterioration at most `0.04`.

For `N` registered behavior pairs, chosen/rejected response lengths
`T_i^+`,`T_i^-`, and response log probabilities, the exact token-normalized
surrogate is

```text
L_beh = beta/(2N) sum_i [log p(y_i^-)/T_i^- - log p(y_i^+)/T_i^+]
P = eta grad L_beh(theta_0) dot (d_S - d_F)
A = L_beh(theta_F) - L_beh(theta_S)
```

For both the quality and safety panels, `P` and `A` must be finite and strictly
positive, and `A/P` must lie inclusively in `[0.25, 1.75]`.

Every arm must also have finite values, relative realized-map L2 error at most
`2e-6`, maximum absolute map error at most `1e-5`, exact restoration, unchanged
RNG/mode/gradients, empty optimizer state, and zero completed optimizer steps.

## Boundary

Any failure stops the route without another radius or dose. A pass supports
only one-displacement finite-loss development on the M5 cohort. It does not
authorize training, checkpoint writes, generation, Core100, or heldout access,
and it does not erase the known M5 unit-level tail.
