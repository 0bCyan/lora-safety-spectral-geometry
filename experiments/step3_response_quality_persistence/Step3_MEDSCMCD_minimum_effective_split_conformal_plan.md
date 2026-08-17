# Step3 MED-SC-MCD: Minimum-Effective-Dose Split-Conformal MCD

## 1. Registered precursor

Seed-5 JCR-MCD stopped at
`STOP_JCRMCD_P0_CONSTRUCTION_UNCERTAINTY_INVALID`. Its four aggregated support
hulls admitted no robust-feasible path candidate, and only `6/56` independent
diagnostic gradient vectors were covered. A finite-panel audit proved that,
for every certificate, the radius required to cover all eight diagnostic
vectors exceeded the largest scalar Euclidean radius compatible with positive
raw robust descent. The gap ranged from `0.5296` to `1.1607`.

That result closes isotropic scalar-radius calibration around the same
aggregated support hulls. MED-SC-MCD changes the calibrated object: it controls
only candidate-relevant score ratios at the same one-pair granularity as the
future unit. No full-gradient norm ball, jackknife support mean, radius
multiplier, certificate-role scan, or post-outcome threshold change is used.

This is a seed-6 exploratory mechanism test on reused disclosed data. It is
designed after observing the JCR-MCD failure and therefore cannot serve as an
independent confirmation.

## 2. Frozen three-way split

The 40 public prompt clusters are globally ordered by ascending SHA256 of

`medscmcd-seed6-split-v1|sample_id`.

The first 12 units are the fit set, the next 20 are calibration, and the last
eight are diagnostics. Risk rows use the first 40 indices of
`torch.randperm(256, seed=6)` in the same role sizes. The hash rule is fixed
without category balancing or outcome access. Every gradient is computed in
eval mode from exactly one prompt/risk pair, so fit, calibration, and
diagnostic score distributions use the same granularity and dropout mode.

The roles are disjoint:

- fit may construct endpoints and select one candidate;
- calibration may only certify or reject that already-fixed candidate;
- diagnostics may only evaluate the frozen candidate and cannot alter it.

## 3. Fit-only path construction

The 12 fit-unit gradients are averaged coordinatewise. For the seven
certificates

\[
\mathcal I=\{R,U,T,E,C,H,G\},
\]

the pooled baseline and auxiliary budget remain

\[
G_0=0.8g_R+0.2g_U,\qquad B=\lVert g_U\rVert_2.
\]

The raw endpoint `d_0` is the full-seven MCD direction. The target endpoint
`d_1` is the MCD direction over

\[
\mathcal Q=\{E,H,G\}.
\]

The candidate family is unchanged from JCR-MCD:

\[
d_\lambda=B\frac{(1-\lambda)d_0+\lambda d_1}
{\lVert(1-\lambda)d_0+\lambda d_1\rVert_2},
\qquad
\lambda\in\{0,1/32,\ldots,1\}.
\]

All 12 fit units score the same 33 exact optimizer-map displacements using
their own normalized certificate gradients and one fixed pooled baseline norm.

## 4. Minimum-effective-dose selection

Let `s_ji(lambda)` be the exact normalized score for fit unit `j` and
certificate `i`. The protected set and retention remain

\[
\mathcal P=\{R,U,T,C\},\qquad\rho=0.95.
\]

The fit target floor is

\[
m_Q(\lambda)=\min_{j\in\mathrm{fit},i\in\mathcal Q}s_{ji}(\lambda).
\]

Candidate `lambda>0` is fit-eligible iff:

\[
s_{ji}(\lambda)/s_{ji}(0)\ge\rho
\quad\forall j, i\in\mathcal P,
\]

\[
m_Q(\lambda)\ge0.05,
\qquad
m_Q(\lambda)-m_Q(0)\ge0.002.
\]

MED-SC-MCD selects the smallest eligible positive `lambda`, not the candidate
with maximum target gain. This is a registered minimum-dose rule: once the
material target floor is reached, additional movement toward the target-only
endpoint has no selection benefit. If no positive candidate is eligible, the
raw anchor is retained only as a failed fallback measurement.

### Proposition 1: fit minimality

If the eligible set is nonempty, the selected candidate satisfies all fit
retention and target gates, and no smaller registered positive fraction does.

This follows directly from ascending finite-family selection. It is an exact
statement over the fit panel, not a transfer guarantee.

## 5. Joint split-conformal accept-or-raw policy

After `lambda*` is fixed, calibration unit `k` receives joint protected
nonconformity

\[
Q_k=\max_{i\in\mathcal P}
\left(\rho-\frac{s_{ki}(\lambda^*)}{s_{ki}(0)}\right).
\]

Raw scores must be positive. `Q_k<=0` means all four protected scores retain
at least 95%. With `n=20`, `alpha=0.05`, the one-sided split-conformal rank is

\[
r=\lceil(n+1)(1-\alpha)\rceil=20,
\]

so the threshold is the largest calibration nonconformity

\[
\widehat q=Q_{(20)}=\max_k Q_k.
\]

### Theorem 1: split-conformal predictive bound

Assume the candidate is fixed before calibration and the 20 calibration plus
one future joint nonconformities are exchangeable. Then

\[
\Pr\{Q_{21}\le\widehat q\}\ge\frac{20}{21}=95.238\%.
\]

### Proof

Under exchangeability, the rank of the future nonconformity among 21 values is
uniform up to conservative tie handling. The event that its rank is at most
20 has probability at least `20/21`; on that event it does not exceed the
largest of the 20 calibration values. QED.

The final sentence does **not** imply 95.238% coverage conditional on observing
`qhat<=0`; ordinary split conformal does not provide that conditional claim.

### Corollary 1: finite-sample protection of the complete policy

Define the registered output policy

\[
\widehat d=\begin{cases}
d_{\lambda^*}, & \widehat q\le0,\\
d_0, & \widehat q>0.
\end{cases}
\]

The raw fallback has protected score ratio exactly one and therefore joint
nonconformity `rho-1=-0.05`. Under the assumptions of Theorem 1,

\[
\Pr\{Q_{21}(\widehat d)\le0\}\ge\frac{20}{21}.
\]

### Proof

On rejection, the raw fallback is protected deterministically. On acceptance,
an unsafe output requires both `qhat<=0` and `Q_21>0`, which implies
`Q_21>qhat`. Thus policy failure is a subset of the split-conformal
miscoverage event in Theorem 1 and has probability at most `1/21`. QED.

This is a marginal guarantee for the full accept-or-raw policy over repeated
calibration/future draws. It is neither a conditional-on-acceptance guarantee
nor a per-category guarantee.

The deterministic hash split does not establish exchangeability; it prevents
model-outcome-dependent role assignment. Distribution shift remains an
explicit scope limitation.

## 6. Independent diagnostic finite-step test

The eight diagnostic units compare three fixed arms:

1. pooled risk/uniform baseline;
2. pooled full-seven raw MCD;
3. pooled MED-SC-MCD.

Each unit first measures exact scores, then transiently applies each fixed
first-step displacement at learning rate `2e-4`, evaluates all seven losses,
and restores the exact initial LoRA state. Calibration scores are not used to
change the candidate after certification.

## 7. Prospective gates

### Integrity

- 12 fit, 20 calibration, and eight diagnostic one-pair units with no overlap;
- all gradients finite and nonzero in eval mode;
- exactly 33 path candidates with active clipping and budget error at most
  `1e-6`;
- exact state restoration, RTX 4070 Ti, and at least 10% VRAM headroom;
- no optimizer, scheduler, checkpoint, generation, Core100, heldout, or raw
  gradient serialization.

### Fit minimum-effective dose

- a positive eligible candidate exists and is the smallest eligible fraction;
- minimum fit protected retention is at least 95%;
- fit target score floor is at least `0.05`;
- fit target floor gain over raw is at least `0.002`.

### Split-conformal accept-or-raw policy

- all 20 calibration raw scores are positive;
- rank is 20 and finite-sample marginal coverage lower bound is at least 95%;
- `qhat<=0`, so the observed policy action is the fit-selected candidate rather
  than the protected raw fallback;
- the registered policy-level marginal protection lower bound is at least 95%,
  and no conditional-on-acceptance coverage is claimed;
- candidate selection records that calibration and diagnostic scores were not
  used.

### Diagnostic gates

The unchanged RP-PFRSC thresholds apply:

- every exact score at least `0.02`; worst-score advantage over baseline mean
  at least `0.02`, positive bootstrap lower bound, and sign-test `p<=0.05`;
- every objective positive in mean finite loss, at least 75% positive units,
  and mean actual/predicted ratio in `[0.25,1.75]`;
- protected exact and actual retention mean at least 95%, bootstrap lower
  retention strictly above 92.5%, and every unit at least 85%;
- target bottleneck gain at least `0.002`, target macro finite-loss gain at
  least 2%, positive bootstrap lower bounds, sign-tests `p<=0.05`, target mean
  retention at least 95%, and every target unit at least 80%;
- mean diagnostic target minimum at least `0.05`.

## 8. Stop boundary

The first failed gate fixes the route. Failure does not permit changing the
split, conformal alpha/rank, fit target floor, fit gain, retention threshold,
candidate grid, seed, or calibration rule. The calibration panel may not be
reused to select a replacement candidate.

A pass authorizes only a separately committed, training-disjoint `proxy_dev`
triplet-data protocol. It does not authorize training. Throughout this P0:
`optimizer_objects=0`, `scheduler_objects=0`, `optimizer_steps=0`,
`virtual_parameter_evaluations=24`, `checkpoint_writes=0`,
`training_authorized=false`, `candidate_generation_authorized=false`,
`core100_opened=false`, and `heldout_opened=false`.
