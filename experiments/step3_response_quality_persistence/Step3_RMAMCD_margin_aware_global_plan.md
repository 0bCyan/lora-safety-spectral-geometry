# Step3 RMA-MCD: Global Protected/Target Margin-Aware MCD

## 1. Motivation and scope

NPT-A-MCD and NPT-LR-MCD both produced strong target gains but failed
out-of-fit TARC protection. Both hard-projected the tangent away from fit
protected gradients. RMA-MCD tests whether that hard nullspace is itself the
wrong inductive bias by optimizing global protected and target margins jointly.

This is a new exploratory protocol on disclosed data. It does not reopen any
preceding result and does not authorize training.

## 2. Frozen split and boundary

The seed-7 12 fit / 20 calibration / 8 diagnostic split, model, eval-mode
one-pair gradients, and all registered numerical thresholds remain unchanged.
Fit builds one candidate, calibration only accepts it or falls back to raw
MCD, and diagnostics are untouched until selection. No optimizer, scheduler,
checkpoint, generation, Core100, or heldout access is allowed;
`optimizer_steps=0` throughout.

## 3. Seven global certificate covectors

For every certificate `q` in the seven-certificate registry, normalize each
fit-unit gradient and average the 12 normalized gradients:

`bar b_q = (1/12) sum_i g_iq / ||g_iq||_2`.

The scoring covector is `b_q = bar b_q / ||bar b_q||_2`; thus the averaging
rule is scale-invariant at the unit level while the final score remains a
cosine-like directional score. This creates four global protected covectors
and three global target covectors. The candidate tangent is constrained only
to be orthogonal to the normalized raw displacement, so its norm can be
matched in an angular construction. It is not claimed to be in the unit-level
protected nullspace.

## 4. Joint margin problem

Set `theta* = acos(0.951) - 1e-6` and let `R` be the fixed score reference
norm. For a target certificate `q`, the fit target margin is

`m_q(u) = raw_q cos(theta*) + sin(theta*) <P_raw b_q,u>/R - 0.05`.

For a protected certificate `p`, the pooled retention margin is

`m_p(u) = cos(theta*) - 0.95
          + sin(theta*) <P_raw b_p,u>/(R raw_p)`.

Here `P_raw` projects only off the raw displacement, and `raw_p` is the pooled
raw protected score. RMA-MCD solves the convex ball problem

`maximize_u min({m_q(u)} union {m_p(u)})`

subject to `||u||_2 <= ||d0||_2`, where `d0` is raw MCD. Its simplex dual is

`min_lambda sum_j lambda_j offset_j
    + ||d0||_2 ||sum_j lambda_j h_j||_2`.

There are seven dual coordinates, and the fixed SLSQP center/vertex
multistart, primal-dual, simplex KKT, and `1e-7` residual checks are frozen.

The candidate is `d* = cos(theta*) d0 + sin(theta*) u*`. The affine solver's
pooled margin and its simplex KKT residual are recorded separately from the
fit audit. Fit-level gates still evaluate all 48 protected facets and all 36
target facets, so a positive pooled margin cannot hide a unit-level failure.
The unit-level retention deviation from `cos(theta*)` is reported as a
diagnostic only; requiring that identity would incorrectly reintroduce the
old hard protected-nullspace assumption.

## 5. Protection and diagnostics

The method makes no exact unit-level protection theorem. Its auditable claim
is only equal-norm/raw-orthogonal geometry, the pooled convex margin/KKT audit,
and the observed unit-level fit margin. The
independent 20-unit split-conformal candidate-or-raw policy remains the safety
certificate, with marginal lower bound `20/21` and no conditional-acceptance
claim. Eight diagnostics compare baseline, raw MCD, and the candidate under
the unchanged exact and finite-step gates.

## 6. Stop rule

Any failure of fit margins, conformal protection, protected diagnostic
noninferiority, finite-step descent, target advantage, or effect strength stops
the attempt. A strong target effect cannot override a protection failure.
