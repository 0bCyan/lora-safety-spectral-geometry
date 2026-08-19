# Conformal-Dose CDR-OPC Foundation

## Motivation and status

M1 rejected an all-or-nothing CDR-OPC endpoint even though its behavior
direction was strong. The endpoint moved `0.66895` fallback norms and its worst
protected residual was risk. Because first-order certificate scores are affine
on a fixed update line, treating only dose zero and dose one discards usable
geometry. This document defines a continuous conformal dose controller. It is a
mathematical and retrospective development artifact, not a new empirical pass.

## Per-unit safe dose

Let the fit-derived direction be fixed before calibration. On calibration unit
`i`, protected certificate `j` has fallback score `b_ij`, endpoint score
`e_ij`, and scale

```text
a_ij = max(|b_ij|, epsilon).
```

At dose `lambda` the score is

```text
s_ij(lambda) = b_ij + lambda (e_ij - b_ij).
```

For allowed standardized drop `delta`, the exact largest safe dose for that
certificate is one when `e_ij >= b_ij`, and otherwise

```text
min(1, delta a_ij / (b_ij - e_ij)).
```

The unit safe dose `L_i` is the minimum over protected certificates. Therefore
every `lambda <= L_i` satisfies

```text
s_ij(lambda) >= b_ij - delta a_ij
```

simultaneously for all registered certificates. Absolute fallback scaling
keeps this definition ordered for negative scores, while positive `epsilon`
prevents instability near zero.

## Finite-sample theorem

For `n` calibration units and target failure level `alpha`, define

```text
k = floor((n + 1) alpha).
```

If `k < 1`, the sample is insufficient. Otherwise select dose
`lambda_hat = L_(k)`, the `k`-th smallest calibration safe dose.

**Theorem.** If the direction and all non-calibration hyperparameters are fixed
before calibration and the `n` calibration safe doses plus the future safe dose
are exchangeable, then

```text
P(L_future >= lambda_hat) >= 1 - k/(n+1) >= 1-alpha.
```

**Proof.** Among `n+1` exchangeable values, the future value has a uniform rank
after random tie breaking. The event `L_future < L_(k)` requires its rank to be
at most `k`, which has probability at most `k/(n+1)`. Without random tie
breaking, ties only make the non-strict safety event more conservative. Since
`lambda_hat <= L_future` implies every registered future certificate obeys its
standardized-drop inequality, the result follows.

At `n=20`, `alpha=0.05`, `k=1`: the selected dose is the minimum calibration
safe dose and the marginal protection lower bound is `20/21 = 0.952381`.
Unlike M1 split-conformal endpoint acceptance, calibration is explicitly
allowed to select the dose. The direction may not be selected or altered using
the same calibration units.

## Claim boundary

The theorem is finite-sample and distribution-free under exchangeability, but
only for first-order protected scores on the fixed line. It is marginal, not
conditional on the selected dose. It does not prove finite-parameter-step
retention, generated-response utility, or exchangeability of a particular data
collection. The M1 panel was opened before this controller was frozen and can
only estimate a development dose; a confirmatory claim requires new protected
units. No optimizer step or training is authorized.
