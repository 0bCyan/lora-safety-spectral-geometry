# FOACPO P0S Segment-Robust Anchor Protocol

P0S addresses the P0R mechanism failure without weakening protection. It
constructs one local exact AdamW max-min anchor on the train protected block and
one on the calibration protected block. On their displacement segment,

```text
d_R(w) = (1-w)d_train + w d_cal,  0 <= w <= 1,
```

each of the eight block-certificate scores is affine in `w`. Their minimum is a
concave piecewise-linear function. A global segment maximum occurs at an
endpoint or an intersection of two affine score functions, so P0S enumerates
exactly that finite set and selects the largest minimum score. The robust anchor
must have strictly positive score above `1e-8` for all eight certificates.

For each frozen alpha, the behavior endpoint is mixed with this robust anchor
using the D3 closed form over all eight train-plus-calibration certificates.
Every candidate therefore retains at least 95% of the robust anchor on both
blocks. Candidate attribution remains the P0R rule: nonzero normalized
increment and strict calibration quality/safety Pareto improvement over the
robust anchor. The fixed winner must pass the unchanged 3/5 and 6/9 calibration
pair gates before mechanism-dev may open.

Mechanism-dev then requires positive aggregate quality and safety descent, at
least 4/6 and 6/9 positive pairs, and 95% protected retention relative to the
robust anchor (or no worsening when a mechanism anchor score is nonpositive).
Passing authorizes only a separately frozen finite-radius verification. No
optimizer step, generation, checkpoint write, Core100, heldout, or training is
authorized.
