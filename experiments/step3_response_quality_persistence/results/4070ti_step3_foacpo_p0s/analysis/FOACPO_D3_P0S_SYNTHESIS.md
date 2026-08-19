# FOACPO-S Method and Evidence Synthesis

## Method retained

The surviving candidate is **FOACPO-S**, factorized optimizer-anchored
contrastive preference optimization with a segment-robust anchor.

Its first mathematical component is the factorized objective

```text
L(theta) = alpha E_Q[l_quality(theta)]
         + (1-alpha) E_S[l_safety(theta)].
```

Linearity of expectation makes independent qualified quality and safety pairs
an unbiased gradient estimator; a full three-response prompt intersection is
not required. D3 therefore preserves 28 quality pairs and 32 safety pairs rather
than discarding them to the 13-prompt triplet intersection.

Its second component constructs exact AdamW local max-min anchors on train and
calibration protected blocks. Eight certificate scores are affine on the segment
between those anchors. Their lower envelope is concave piecewise linear, so
enumerating endpoints and pairwise intersections gives the exact segment
max-min anchor. The behavior endpoint is then injected by a second closed-form
line search that retains at least 95% of all eight protected scores.

## What passed

The robust anchor was strictly feasible, with calibration-anchor weight
`0.47748`. The selected `alpha=0.5` route had an 8.05% normalized displacement
from the anchor and exactly retained the frozen 95% floor on all train and
calibration protected certificates. It strictly improved calibration quality
and safety cosine over the anchor by `0.01253` and `0.01200`; calibration pair
sign gates passed at `3/5` and `9/9`.

On the prospectively sealed mechanism-dev split, every quality pair (`6/6`) and
eight of nine safety pairs had positive directional transfer. Aggregate quality
and safety cosines were `0.02775` and `0.02240`. These are not a weak or
zero-direction result: the behavior mechanism transferred across the frozen
split.

## What failed

The full method gate still failed. Mechanism risk improved despite a negative
anchor score; uniform retained `0.96474` and continuation retained `0.97212`.
TARC retained `0.9498136`, missing `0.95` by `0.0001864`. The valid route is
`STOP_FOACPO_P0S_MECHANISM_TRANSFER`; it cannot be rounded into a pass.

This leaves a useful but limited conclusion: factorization solved the data-yield
problem and produced a transferable behavior direction; segment robustification
solved train-to-calibration protection; a small residual protected transport
gap remains on unseen mechanism data.

## Evidence boundary and next study

The opened mechanism block is now descriptive evidence and cannot be used to
tune and reconfirm the same method. A credible next study needs fresh blindly
qualified quality/safety pairs and a fresh protected validation block. Before
opening either, it should derive and freeze a transport buffer, for example a
`0.96` fit-retention requirement evaluated against the unchanged `0.95` test
floor. That value is only a proposed design choice here, not a passed result.

Only a fresh zero-step conjunction pass should authorize finite-radius virtual
parameter verification. Training remains later still. Current evidence does
not support generated-response improvement or training efficacy.

All runs restored the model state and used `optimizer_steps=0`;
`training_authorized=false`. Core100 and heldout remained closed.
