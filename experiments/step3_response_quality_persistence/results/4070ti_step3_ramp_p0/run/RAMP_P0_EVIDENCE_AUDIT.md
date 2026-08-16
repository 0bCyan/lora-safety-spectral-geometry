# RAMP-P0 Evidence Audit

## 1. Status

The frozen route is `PASS_RAMP_P0_AUTHORIZE_MATCHED_SMOKE_PROTOCOL`. This audit
was written after the machine decision and was not an input to that decision.
The seven machine-generated artifacts and their reproduction hashes are
unchanged.

P0 establishes a nontrivial, prompt-disjoint first-order direction for the
RAMP prefix certificate. It does not establish generated safety, response
quality, utility, or publication readiness.

## 2. Mathematical Link to Greedy Decoding

For target token `y_t`, define

\[
m_t=z_t(y_t)-\max_{v\ne y_t}z_t(v).
\]

If all first-16 margins are strictly positive, induction over greedy decoding
proves that the generated sequence begins with those exact 16 target tokens.
RAMP minimizes the smooth certificate deficit

\[
C_{pre}=\frac{1}{16}\sum_{t=1}^{16}\operatorname{softplus}(-m_t).
\]

The suffix placebo uses the identical loss on the final 16 semantic target
tokens. Both arms use the same rows, target text, term count, strongest-
competitor rule, and barrier. Their auxiliary calibration gradients are scaled
to the same norm, `0.25 ||g_uniform||`.

## 3. Structural Evidence

| Quantity | Result | Frozen gate |
| --- | ---: | ---: |
| approved target rows | 80 | exact |
| semantic target length | 22 to 37 | at least 16 |
| prefix completes first sentence | 0.9875 | at least 0.95 |
| prefix reaches routing onset | 0.8875 | at least 0.85 |
| normalized prefix uniqueness | 1.0000 | at least 0.99 |
| mean prefix/suffix overlap | 3.9625 tokens | descriptive |

The target prefixes are therefore not one repeated generic string, although
structural uniqueness is not a behavioral quality judgment.

## 4. Baseline Difficulty

At the fresh seed-1 step-0 policy, calibration prefix mean margin was
`-4.94597`; validation prefix mean margin was `-4.57634`. Only `0.296875` and
`0.375000` of the selected target tokens, respectively, were already greedy
winners. No calibration or validation row had all 16 margins positive.

Thus the full certificate was unsatisfied on every P0 row. The direction test
did not pass by operating on an already solved prefix.

## 5. Equal-Budget Directional Result

| Quantity | Result |
| --- | ---: |
| uniform calibration gradient norm | 9.590803 |
| target auxiliary norm | 2.397701 |
| prefix coefficient | 0.262232 |
| suffix coefficient | 0.239193 |
| maximum relative budget error | 1.53e-8 |
| calibration prefix/suffix cosine | 0.707916 |
| calibration/validation prefix cosine | 0.906324 |
| suffix-calibration/prefix-validation cosine | 0.662709 |
| registered directional cosine gap | 0.243614 |

Because the auxiliary norms are equal, the cosine gap directly compares their
normalized first-order alignment to the validation prefix objective. The RAMP
auxiliary validation dot product was `19.506339`, versus `14.263157` for the
suffix placebo, a `36.8%` larger incremental first-order decrease. Including
the shared uniform gradient, the corresponding dots were `83.957795` and
`78.714613`, a `6.7%` difference.

These are local Taylor coefficients, not observed optimizer-step effects. A
matched smoke must still demonstrate that the implementation realizes the
registered assignments without numerical or exposure differences.

## 6. Integrity and Claim Boundary

- Initial and final trainable-state SHA-256 were both
  `4e4b6ec33469d81c3cc02d1527073151178880cfa252c08b16e59e460efd1924`.
- All six aggregate gradient objects were finite, nonzero, and covered every
  trainable LoRA tensor; structural fresh-LoRA zero values were retained as
  records rather than treated as missing.
- Peak reserved VRAM was `4,930,404,352` bytes, leaving `61.7%` headroom.
- Runtime was about `42.0` seconds.
- `optimizer_objects=0`, `scheduler_objects=0`, `optimizer_steps=0`, and
  checkpoint writes were zero.
- No generation, proxy-dev, Core100, heldout, or reviewer data was opened.

The only authorized next action is a separately frozen matched two-step smoke
for uniform CE, RAMP, and suffix placebo. No behavioral or paper claim is
authorized by P0.
