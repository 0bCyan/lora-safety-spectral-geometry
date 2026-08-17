# Step3 Q-E-TARC: Positive Continuation Quota Geometry

## 1. Registered precursor result

CC-E-TARC P0 showed that the complete-path certificate has a strong,
prompt-disjoint route direction, but its binary continuation halfspace was
non-discriminating. The raw E-TARC/continuation dot was already positive, so
the projection was inactive. Both E-TARC and TARC were continuation-compatible
on 32/32 prompts even though the earlier TARC free-generation gate had shown a
redirect deficit. Binary non-conflict is therefore too weak a mechanism.

Q-E-TARC replaces the zero threshold with a positive angular quota. It does
not alter the complete-path certificate, reuse the failed halfspace control,
or relax any CC-E-TARC gate.

## 2. Fixed-budget constrained direction

Let

\[
u=\frac{g_E}{\|g_E\|_2},\qquad
c=\frac{g_C}{\|g_C\|_2},\qquad a=\langle u,c\rangle,
\]

where g_E is the complete-path E-TARC gradient and g_C is post-route semantic
continuation CE. For auxiliary budget B > 0 and fixed quota gamma in (0,1),
solve

\[
\max_d\ \langle u,d\rangle
\quad\text{subject to}\quad
\|d\|_2=B,\quad \langle c,d\rangle\ge B\gamma.
\]

The experiment prospectively fixes `gamma=0.5` and
`B=0.25 ||g_uniform||`. The quota is absolute, not selected from the P0
outcome or fitted to validation prompts.

## 3. Closed-form optimum

### Theorem 1

If a >= gamma, the unique route-optimal direction is d* = B u. If a < gamma
and |a| < 1, define

\[
q=\frac{u-ac}{\sqrt{1-a^2}}.
\]

Then

\[
d^*=B\left(\gamma c+\sqrt{1-\gamma^2}\,q\right).
\]

### Proof

Any unit feasible direction can be decomposed as

\[
v=\beta c+\sqrt{1-\beta^2}\,w,
\]

where beta >= gamma, w is orthogonal to c, and ||w||=1. For fixed beta,
Cauchy-Schwarz maximizes the route dot at w=q, giving

\[
f(\beta)=a\beta+\sqrt{1-a^2}\sqrt{1-\beta^2}.
\]

Its unconstrained maximum occurs at beta=a. If a>=gamma, v=u is feasible and
attains the global upper bound 1. If a<gamma, f decreases for beta>a, so the
feasible maximum is at beta=gamma and yields the stated expression. Scaling
by B proves the theorem.

### Corollary 1: continuation lower bound

For an update theta' = theta - eta d*,

\[
C(\theta')-C(\theta)
=-\eta\langle g_C,d^*\rangle+O(\eta^2)
\le-\eta B\gamma\|g_C\|_2+O(\eta^2).
\]

Unlike binary non-conflict, Q-E-TARC guarantees a nonzero first-order
continuation allocation whenever g_C is nonzero.

### Corollary 2: maximal route retention

When the quota is active, normalized route retention is

\[
\frac{\langle g_E,d^*\rangle}{B\|g_E\|_2}
=a\gamma+\sqrt{1-a^2}\sqrt{1-\gamma^2}.
\]

No other direction with the same norm and quota can retain more local E-TARC
descent.

## 4. Prospective zero-update gate

The development P0 reuses the disclosed eight calibration and 32 validation
prompt clusters from CC-E-TARC. These prompts are no longer fresh and can
support only mechanism development. Core100 and heldout remain sealed.

RAMP, TARC, raw E-TARC, and Q-E-TARC are compared at exactly the same
auxiliary norm. A pass requires:

- achieved calibration continuation cosine `0.5 +/- 1e-6` and exact norm
  matching within relative tolerance `1e-6`;
- Q-E-TARC calibration route retention cosine at least `0.85`;
- validation route dot at least `0.80` of raw E-TARC and `1.20` of TARC;
- positive route dots on at least `90%` of validation prompts;
- validation continuation cosine at least `0.40` and at least `0.25` above
  TARC;
- validation continuation dot at least `50%` of RAMP's equal-budget positive
  control;
- paired continuation-dot advantage over TARC with bootstrap lower bound
  above zero and exact sign-test p <= `0.05`;
- positive branch-entropy and negative-gap dots, meaning the descent direction
  retains the registered strong-path branch signature;
- exact state restoration and at least `10%` VRAM headroom.

Failure closes Q-E-TARC. A pass authorizes only a separately frozen matched
smoke protocol. It does not establish generated safety, utility, or a paper
claim.

## 5. Boundaries

P0 creates no optimizer, scheduler, checkpoint, adapter, or candidate output.
No full gradient is serialized. `optimizer_steps=0`,
`training_authorized=false`, `core100_opened=false`, and
`heldout_opened=false`.
