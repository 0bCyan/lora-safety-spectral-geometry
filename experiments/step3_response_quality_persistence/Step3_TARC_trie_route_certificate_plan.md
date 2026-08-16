# Step3 TARC: Trie-Aware Routing Certificates

## 1. Evidence-Motivated Problem

RAMP improved the prompt-disjoint full-answer redirect marker rate by 0.30, but
its first-16-token refusal gain was only 0.05 versus uniform and 0.075 versus a
suffix placebo. Both paired intervals crossed zero, and procedural markers
increased. The registered RAMP gate therefore stopped. This pattern suggests
that a fixed 16-token exact-target objective spends substantial auxiliary
budget after the refusal decision and can transfer prompt-specific action
language. TARC tests a narrower mechanism rather than increasing RAMP's weight
or training duration.

## 2. Safe-Route Trie

Each frozen safety-training prompt has two independently approved natural
responses. For each response, tokenization stops at the first completion of a
registered refusal phrase such as `I can't help`, `I won't provide`, or
`I won't assist`. The two token prefixes form a prompt-specific trie
\(\mathcal T_x\). At node \(u\), let \(A(u)\) be its registered child-token
set and \(z_v(x,u)\) the next-token logit.

For temperature \(\tau>0\), define the conservative allowed score and node
margin

\[
s_\tau(A)=-\tau\log\sum_{a\in A}\exp(-z_a/\tau),
\qquad
m(x,u)=s_\tau(A(u))-\max_{v\notin A(u)}z_v.
\]

Because \(s_\tau(A)\le\min_{a\in A}z_a\), a positive node margin implies
every registered child outranks every unregistered token. The auxiliary loss is

\[
C_{\mathrm{TARC}}(x)=\frac1{|\mathcal N_x|}
\sum_{u\in\mathcal N_x}\operatorname{softplus}(-m(x,u)).
\]

## 3. Greedy Certificate Theorem

**Theorem.** If \(m(x,u)>0\) for every internal node reachable from the root
of \(\mathcal T_x\), greedy decoding from prompt \(x\) emits one complete
registered refusal route.

**Proof.** At the root, positivity and the conservative-score inequality imply
that every allowed child logit is strictly greater than every disallowed logit,
so the greedy token is an allowed child. Assume the greedy prefix reaches an
internal trie node \(u\). The same inequality makes its next greedy token a
child in \(A(u)\). Induction keeps the path in the finite trie until a registered
leaf is reached. Strict inequalities avoid tie ambiguity. QED.

The theorem certifies membership in a registered refusal-route language. It
does not certify the semantics of text after the leaf, prompt-level safety, or
utility.

## 4. P0 Mechanism Test

Before any optimizer exists, P0 uses prompt-disjoint calibration and validation
prompt pairs at the fresh seed-1 LoRA state. Five source categories have at
least two prompt clusters and are represented on both sides; three categories
are singletons and therefore cannot be represented in both disjoint splits.
A deterministic maximum-coverage rule assigns those singletons and fills each
split to eight prompt pairs while minimizing within-split category reuse. P0
computes aggregate gradients for
uniform response CE, the prior 16-token exact-target RAMP loss, and TARC.
RAMP and TARC auxiliary gradients are independently scaled to exactly 0.25 of
the calibration uniform-gradient norm.

P0 passes only if:

- all 80 approved targets complete a registered route within 16 semantic
  tokens and all 40 two-target tries are structurally valid;
- every aggregate gradient covers every trainable tensor, is finite and
  nonzero, and the exact trainable-state hash is restored;
- both auxiliary budgets match within relative tolerance `1e-6`;
- the TARC calibration gradient aligns positively with the prompt-disjoint
  TARC validation gradient;
- equal-budget TARC has a prospectively fixed directional advantage over RAMP
  on the TARC validation gradient;
- at least 10% VRAM headroom remains.

P0 creates no optimizer, scheduler, checkpoint, generation, or candidate text.
A pass can authorize only a separately frozen matched uniform/RAMP/TARC smoke
protocol. Core100, proxy-dev outputs, and heldout remain outside P0.

## 5. Claim Boundary

TARC is a mechanism hypothesis with an exact conditional certificate. A P0
pass establishes only first-order cross-prompt direction at initialization.
Training, free generation, blinded semantic safety, and utility each require
separate prospective gates. A failed gate is retained as a negative result and
stops the corresponding path.
