# Step3 NPT-MCD: Nullspace-Protected Target MCD

## 1. Registered precursor and scope

Seed-6 MED-SC-MCD stopped at
`STOP_MEDSCMCD_P0_NO_MINIMUM_EFFECTIVE_DOSE`. Its first positive scalar path
dose increased the fit target floor by only `0.0018856` while reducing the
worst protected retention to `0.9436408`. Every registered dose that met the
target floor and gain gates violated the 95% protected-retention gate. The
bottleneck was a low-margin unit-level TARC facet.

NPT-MCD replaces that one-dimensional endpoint homotopy. It removes the target
correction component along every protected fit covector before selecting an
angular dose. It does not lower any registered effect, retention, calibration,
or diagnostic threshold.

This method is distinct from the stopped seed-2 functional-nullspace family.
That family projected training gradients against activation singular vectors
and failed behavioral redirect gates. NPT-MCD uses one-pair objective gradients
as covectors, constructs one transient fixed displacement, performs no
training, and makes no behavioral claim.

The public prompt pool has already been disclosed by prior exploratory runs.
Seed 7 is therefore a development mechanism test, not an independent
confirmation or paper-level generalization result.

## 2. Frozen three-way split

The 40 prompt clusters are ordered by ascending SHA256 of

`nptmcd-seed7-split-v1|sample_id`.

The first 12 units are fit, the next 20 are calibration, and the final eight
are diagnostics. Risk rows are the first 40 indices of
`torch.randperm(256, seed=7)` in the same role sizes. Every gradient is measured
in eval mode from exactly one prompt/risk pair.

Fit may construct and select one candidate. Calibration may only accept that
candidate or select the raw fallback. Diagnostics may only measure the frozen
candidate and cannot alter it.

## 3. Exact endpoint maps

The 12 fit gradients are averaged coordinatewise. For certificates

`I = {R, U, T, E, C, H, G}`,

the pooled baseline and auxiliary budget are

`G0 = 0.8 g_R + 0.2 g_U` and `B = ||g_U||_2`.

The full-seven MCD direction and the target-only MCD direction over

`Q = {E, H, G}`

are passed through the exact clipped first-step Adam map `M` used by the prior
P0 experiments:

`d0 = M(G0, v_all)` and `dT = M(G0, v_target)`.

The raw endpoint `d0` remains the control. The target proposal is

`c = dT - d0`.

Both source endpoint maps must exercise registered clipping. The NPT candidate
is a post-map fixed displacement and is not claimed to be another ordinary
Adam output.

## 4. Protected-null tangent

For each of 12 fit units and four protected certificates

`P = {R, U, T, C}`,

let `a_ji` be its unit-normalized gradient covector. Stack the 48 rows into
`A`. Add the normalized raw displacement as a forty-ninth constraint:

`A_tilde = [A; d0^T / ||d0||_2]`.

The protected-null target component is

`z = (I - A_tilde^T (A_tilde A_tilde^T)^+ A_tilde) c`.

The pseudoinverse is computed from the 49 by 49 float64 Gram eigensystem with
relative eigenvalue cutoff `1e-10`, followed by two fixed residual-refinement
iterations. The construction requires

`||z||_2 / ||c||_2 >= 0.01`.

If it passes, define the matched-norm tangent

`u = ||d0||_2 z / ||z||_2`.

### Proposition 1: protected-null orthogonal basis

In exact arithmetic,

`A u = 0`, `<d0, u> = 0`, and `||u||_2 = ||d0||_2`.

This follows because `z` is the orthogonal projection onto the nullspace of
all rows of `A_tilde`, and the final scalar normalization does not change
orthogonality.

The implementation separately gates the maximum protected cosine, raw-tangent
cosine, and norm error at `1e-6`.

## 5. Fixed-norm angular family

For angle `theta`, define

`d(theta) = cos(theta) d0 + sin(theta) u`.

### Theorem 1: norm and protected-score identity

Under Proposition 1,

`||d(theta)||_2 = ||d0||_2`

and every fit protected score has retention

`s_ji(theta) / s_ji(0) = cos(theta)`.

The norm identity follows from equal norms and orthogonality. The score
identity follows from `<a_ji,u>=0`. This removes the small-denominator failure
of MED-SC-MCD: every protected facet shares the same analytically controlled
retention.

Finite-precision scoring records the exact identity error. The search is
restricted to

`0 <= theta <= acos(0.951)`.

The design cosine of 95.1% leaves a 0.1 percentage-point numerical margin over
the registered 95% fit gate. The observed minimum retention must still be at
least 95%, and maximum identity error must be at most `1e-5`.

## 6. Exact minimum-effective angle

For every fit target score, let `r_j` and `t_j` be its scores on `d0` and `u`.
Then

`s_j(theta) = r_j cos(theta) + t_j sin(theta)`.

Let the raw fit target floor be `m0`. The required common target floor is

`m_req = max(0.05, m0 + 0.002)`.

Each equality `s_j(theta)=m_req` has an analytic finite set of trigonometric
roots on the registered angular cap. NPT-MCD enumerates all such roots, tests
the sign-constant intervals between adjacent roots, and selects the leftmost
positive jointly feasible angle plus a fixed `1e-8` radian numerical guard.
There is no outcome-dependent grid refinement or target-maximizing scan.

If no angle satisfies all fit target constraints and exact fit protection, the
raw anchor is retained only as a failed fallback measurement.

## 7. Split-conformal accept-or-raw policy

After the fit angle is fixed, calibration unit `k` receives joint protected
nonconformity

`Q_k = max_i (0.95 - s_ki(theta*) / s_ki(0))`.

With 20 calibration units and `alpha=0.05`, the rank is 20 and
`qhat=max_k Q_k`. The registered policy uses the NPT candidate only if a fit
candidate exists and `qhat<=0`; otherwise it uses raw MCD.

### Theorem 2: finite-sample protection of the complete policy

Assume the fit candidate is fixed before calibration and the 20 calibration
plus one future candidate nonconformities are exchangeable. The complete
candidate-or-raw policy has future joint protection probability at least

`20/21 = 95.238%`.

On rejection, raw MCD has protected ratio one. On acceptance, an unsafe output
requires `Q_future>0>=qhat`, which is a subset of split-conformal miscoverage.
That event has probability at most `1/21`.

This is a marginal guarantee over calibration and future draws for the full
policy. It is not coverage conditional on acceptance, not conditional by
category, and not valid under unmodeled distribution shift.

## 8. Independent diagnostics

Eight untouched role-assigned units compare:

1. pooled risk/uniform baseline;
2. full-seven raw MCD;
3. NPT-MCD or its registered raw fallback.

Each arm is applied transiently at learning rate `2e-4`, all seven losses are
measured, and the exact initial LoRA state is restored. The same prospective
diagnostic gates as MED-SC-MCD apply: common exact descent, finite-step descent,
protected noninferiority, target exact and finite-loss advantage, and a
non-weak absolute target floor.

## 9. Stop boundary

The first failed gate fixes the route in this order:

1. integrity;
2. protected-null geometry;
3. minimum-effective angle;
4. conformal accept-or-raw policy;
5. diagnostic transfer;
6. finite-step common descent;
7. protected diagnostic noninferiority;
8. target diagnostic advantage;
9. effect strength.

Failure does not permit changing the split, constraints, pseudoinverse cutoff,
refinement count, retained-component floor, angular cap, angle guard,
calibration rank, or effect thresholds. A pass authorizes only a separately
frozen, training-disjoint proxy-dev data protocol. It does not authorize an
optimizer, checkpoint, generation, or training.

Throughout P0: `optimizer_objects=0`, `scheduler_objects=0`,
`optimizer_steps=0`, `virtual_parameter_evaluations=24`,
`training_authorized=false`, `candidate_generation_authorized=false`,
`core100_opened=false`, and `heldout_opened=false`.
