# FOACPO P0R Anchor-Relative Prospective Protocol

P0R preserves the P0 data, checkpoint, alpha registry, exact AdamW anchor,
95% protection floor, per-pair calibration gates, and unopened mechanism-dev.
It does not reclassify the valid P0 stop.

P0's absolute `lambda>=0.05` rule is replaced because an interpolation
coordinate is not an invariant effect size. P0R registers a stronger method
attribution rule before recomputing any anchor-relative score. A candidate must:

1. have normalized displacement from the anchor greater than `1e-6`, used only
   as a numerical nonidentity check;
2. retain at least 95% of every positive train and calibration protected anchor
   certificate, and not worsen a nonpositive calibration anchor certificate;
3. have positive calibration descent for both factorized objectives; and
4. strictly improve both calibration quality and calibration safety cosine
   relative to the optimizer-space anchor.

Among strict Pareto candidates, selection maximizes the smaller anchor-relative
gain, then total gain, then proximity to `alpha=0.5`. Only that fixed route is
checked on individual calibration pairs using the unchanged 3/5 quality and
6/9 safety gates. Only if they pass may mechanism-dev open, with the unchanged
positive aggregate, 4/6 quality, 6/9 safety, and protected-transfer gates.

Passing authorizes only a separately frozen finite-radius verification. No
optimizer step, generation, checkpoint write, Core100 access, heldout access,
or training is authorized.
