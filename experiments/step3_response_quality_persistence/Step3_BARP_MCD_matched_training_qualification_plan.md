# Step3 BARP-MCD Q1: Matched Two-Step Training Qualification

## 1. Question

BARP-MCD P0 passed its frozen same-batch, zero-optimizer-step mechanism gates.
Q1 asks the narrower implementation question required before any longer run:

> Can BARP-MCD be recomputed at every optimizer batch, scored through the
> current AdamW state, and applied for two consecutive matched updates without
> violating its registered local protection or numerical contracts?

Q1 is an exploratory qualification on disclosed seed-4 source blocks. It is
not fresh confirmation and cannot establish generated safety or utility.

## 2. Matched arms and budget

Two fresh Qwen3-4B LoRA models start from the same initialization:

- `raw_mcd`: recompute the full-seven raw MCD anchor on each optimizer batch;
- `barp_mcd`: recompute the complete BARP-MCD finite family and select under
  the frozen four-coordinate protection rule.

Both arms use the same two registered optimizer blocks, each containing four
risk rows and four safety prompt pairs. The optimizer, learning rate, clipping,
model mode, order, and objective implementation are identical. Each arm is
allowed exactly two AdamW steps, for four optimizer steps total. No scheduler,
checkpoint, generation, Core100, or heldout access is allowed.

## 3. Current-state optimizer map

P0 used the exact first-step Adam map. At training step `t`, Q1 instead scores
every candidate with the current optimizer moments. For a globally clipped
candidate gradient `g_t`, the positive displacement before learning-rate
scaling is

```text
m_t = beta1 m_(t-1) + (1-beta1) g_t
v_t = beta2 v_(t-1) + (1-beta2) g_t^2
d_t = (m_t / (1-beta1^t)) /
      (sqrt(v_t / (1-beta2^t)) + epsilon).
```

Weight decay is frozen to zero. Q1 verifies after every real optimizer step
that the observed parameter displacement equals `learning_rate * d_t` within
the registered floating-point tolerance. Candidate scoring must not mutate
the optimizer state.

## 4. Frozen BARP selection

The seven certificates and finite family are unchanged from P0. The protected
set is `{risk, uniform, tarc, continuation}` and the target set is
`{etarc, branch_entropy, branch_negative_gap}`. Every optimizer batch uses all
127 nonempty MCD subset endpoints, the frozen Top-16 screen, and the nine-point
raw-anchored homotopy.

Let `x_0,t` be the current-state AdamW displacement of full-seven raw MCD and
let `H_t` be the 129-candidate final family. BARP selects

```text
F_t = {x in H_t: s_p(x) >= 0.95 s_p(x_0,t) for every protected p}
x_t* in argmax over F_t of min over target q of s_q(x).
```

The raw anchor remains an explicit member, so the finite-family feasibility
and protection theorem applies separately at every current optimizer state.

## 5. Qualification gates

Q1 passes only if:

1. both arms execute exactly two matched updates from identical initial state;
2. all gradients, losses, moments, and parameter updates remain finite;
3. the analytic current-state AdamW displacement matches every real update;
4. every raw anchor has strict seven-coordinate common descent;
5. every BARP selection retains at least 95% on all four protected scores and
   has target bottleneck no worse than its current raw anchor;
6. BARP selects a nonzero homotopy candidate on at least one of two steps;
7. all seven same-batch finite-step losses decrease after each selected update;
8. peak reserved VRAM leaves at least 10% device headroom; and
9. no checkpoint, generated output, Core100, or heldout artifact is produced.

The excluded eight-category source block is evaluated before and after each
arm only as a descriptive diagnostic. It cannot select a route or tune a
threshold in Q1.

## 6. Decision boundary

A pass authorizes only a separately frozen longer matched-training protocol.
A failure stops before longer training and is reported by the first failed
registered gate. Q1 does not authorize candidate generation or paper-level
behavioral claims.
