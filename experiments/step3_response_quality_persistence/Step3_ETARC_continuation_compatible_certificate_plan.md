# Step3 E-TARC: Continuation-Compatible Existential Route Certificates

## 1. Motivation and registered failure

TARC produced a strong prompt-disjoint four-token route effect, but its frozen
generation gate failed because the gain did not survive as a sufficiently
strong safe redirection effect. Relative to RAMP, TARC improved exact route
entry by 0.125 while reducing the redirect marker rate by 0.125 and response
uniqueness by 0.10. The original TARC soft minimum requires every registered
child at a trie branch to outrank every unregistered token. Its derivative is
largest on the weakest allowed branch. This is a stronger condition than
greedy decoding needs and can spend gradient on mutually exclusive routes.

E-TARC tests a weaker sufficient certificate. CC-E-TARC then adds an exact
first-order constraint against damaging the natural continuation after the
registered route. This is a new exploratory method, not a reinterpretation of
the failed TARC gate.

## 2. Existential complete-path certificate

Let P(x) contain the registered token paths from prompt x to safe-route trie
leaves. For token t on path p, define the exact greedy margin

\[
d_{p,t}=z_{p_t}(x,p_{<t})-\max_{v\ne p_t}z_v(x,p_{<t}).
\]

The competitor set includes tokens on other registered branches. Thus every
positive token margin certifies the particular path token, rather than merely
membership in the union of legal children. Define a conservative within-path
bottleneck and an existential across-path score:

\[
b_\tau(p)=-\tau\log\sum_t\exp(-d_{p,t}/\tau),
\]

\[
r_{\tau,\sigma}(x)=\sigma\log\left(
\frac{1}{|P(x)|}\sum_{p\in P(x)}\exp(b_\tau(p)/\sigma)
\right).
\]

The E-TARC loss is

\[
C_\exists(x)=\operatorname{softplus}(-r_{\tau,\sigma}(x)).
\]

### Lemma 1: nested lower bound

For every path p and finite nonempty path set P,

\[
b_\tau(p)\le\min_t d_{p,t},
\qquad
r_{\tau,\sigma}(x)\le\max_{p\in P(x)}b_\tau(p).
\]

The first inequality follows because the exponential sum is at least the term
for the minimum margin. For the second, every exponential in the mean is at
most exp(max b / sigma). Applying the scaled logarithms proves both bounds.

### Theorem 1: greedy route certificate

If r(tau,sigma)(x) > 0, greedy decoding from x emits one complete registered
route.

By Lemma 1, a positive outer score implies that some path p has b(p) > 0,
which implies min_t d(p,t) > 0. At every prefix on p, its next token therefore
strictly outranks every other vocabulary token. Induction over the finite path
forces greedy decoding to emit p and reach its registered leaf. Strict margins
exclude ties. The theorem says nothing about text after the leaf or semantic
safety.

The two gradient levels implement the intended division of labor:

\[
\frac{\partial r}{\partial b(p)}
=\operatorname{softmax}_p(b(p)/\sigma),
\qquad
\frac{\partial b(p)}{\partial d_{p,t}}
=\operatorname{softmax}_t(-d_{p,t}/\tau).
\]

The outer derivative selects the currently strongest complete registered
path; the inner derivative repairs that path's weakest token. Original TARC
instead reinforces every legal branch at every reached trie node. This nested
strong-path/weak-token ordering is the registered mechanism distinction.

## 3. Continuation-compatible direction

Let g_E be the E-TARC gradient and g_C the cross-entropy gradient on semantic
tokens strictly after each registered route leaf. Define the auxiliary
direction as the Euclidean projection

\[
d^*=\arg\min_d\frac12\|d-g_E\|_2^2
\quad\text{subject to}\quad\langle d,g_C\rangle\ge0.
\]

The Karush-Kuhn-Tucker conditions give the closed form

\[
d^*=\begin{cases}
g_E,&\langle g_E,g_C\rangle\ge0,\\
g_E-\dfrac{\langle g_E,g_C\rangle}{\|g_C\|_2^2}g_C,
&\langle g_E,g_C\rangle<0.
\end{cases}
\]

For an update theta' = theta - eta d*, the first-order continuation change is
-eta <g_C,d*> <= 0. If projection is active, route descent is retained because

\[
\langle g_E,d^*\rangle=\|g_E\|_2^2
\left(1-\cos^2(g_E,g_C)\right)\ge0,
\]

with strict inequality unless the gradients are collinear. Positive rescaling
to a fixed auxiliary norm preserves both inequalities. The guarantee is local
and calibration-batch conditional; prompt-disjoint transfer must be measured.

## 4. Prospective P0

P0 creates a fresh seed-1 LoRA policy and no optimizer. It reuses the fixed
eight-prompt TARC calibration split and assigns all other 32 approved prompt
clusters to validation. This split was available before E-TARC and is
exploratory rather than confirmatory.

The real-model comparison computes gradients for uniform full-response CE,
RAMP16, TARC, E-TARC, and post-route continuation CE. RAMP, TARC, raw E-TARC,
and CC-E-TARC are independently normalized to 0.25 of the calibration uniform
gradient norm. Validation additionally measures branch entropy and the
strongest-versus-second legal-token gap. Per-prompt validation gradients are
retained only long enough to compute scalar dots, cosines, paired bootstrap
intervals, and sign tests; full gradients are never serialized.

CC-E-TARC can authorize a separate matched smoke only if all integrity checks
pass and all of the following prospective requirements hold:

- calibration-to-validation E-TARC cosine is at least 0.10;
- at least 80% of validation prompts have a positive CC-E-TARC route dot;
- its aggregate existential-route dot retains at least 80% of raw E-TARC and
  at least 80% of equal-budget TARC;
- at least 75% of validation prompts have a nonnegative continuation dot;
- continuation-compatible prompt rate exceeds TARC by at least 0.15;
- the paired bootstrap lower bound for the continuation-dot advantage over
  TARC is nonnegative;
- exact state restoration and at least 10% VRAM headroom hold.

Failure closes this formulation. Thresholds, temperature, prompt allocation,
and gradient budget may not be adjusted after P0 output is observed.

## 5. Evidence and claim boundary

A P0 pass establishes a prompt-disjoint first-order mechanism result only. A
separately frozen matched smoke, development run, free generation gate, and
finally blinded safety/utility review are still required. Core100 and heldout
remain sealed. No generated output, human label, or optimizer step belongs to
P0.
