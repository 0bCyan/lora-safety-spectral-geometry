# FOACPO P0 Prospective Zero-Step Protocol

P0 tests the factorized optimizer-anchored method on the frozen Qwen3-4B
constant-20 LoRA checkpoint without changing parameters. Quality and safety
preference gradients are estimated from their independent D3 pools using
length-normalized response log probability. The registered mixture weights are
`alpha in {0.25, 0.50, 0.75}`.

For each mixture, the exact next-step AdamW behavior displacement is computed
without mutating optimizer state. A protected AdamW anchor is independently
constructed from risk, uniform, TARC, and continuation certificates. The
closed-form D3 interpolation chooses the largest behavior fraction retaining
at least 95% of every train-block anchor certificate. A route must inject at
least 5% behavior displacement and improve both aggregate calibration
objectives. The best route maximizes the worse calibration cosine, then their
sum, then proximity to `alpha=0.5`.

Only the selected route receives per-pair calibration checks. It requires at
least 3/5 positive quality pairs, 6/9 positive safety pairs, and complete
protected transfer. Only after that conjunction passes may mechanism-dev open.
The fixed route then requires positive aggregate quality and safety cosines,
at least 4/6 and 6/9 positive pairs respectively, and complete protected
transfer. Support certificates with a positive anchor score retain at least
95%; a nonpositive support anchor may not be worsened.

These are first-order optimizer-space mechanism gates, not behavioral or
finite-training claims. Passing authorizes only a separately frozen
finite-radius one-step verification. P0 creates one optimizer object but calls
no optimizer step, writes no checkpoint, generates no response, and keeps
Core100 and heldout closed. `optimizer_steps=0` and
`training_authorized=false` in every route.
