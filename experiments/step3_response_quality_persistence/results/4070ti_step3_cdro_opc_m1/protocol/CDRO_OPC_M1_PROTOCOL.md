# CDR-OPC M1 Protocol

M1 is a zero-optimizer-step test of the CDR-OPC controller frozen in M0. It
constructs at most five orthonormal realized-update directions from two local
FOACPO anchors and three factorized behavior proposals. Six historical
two-pair protected blocks fit sign-safe standardized score ellipsoids. The
controller is solved once, and its coordinate digest is recorded before any M1
protected calibration gradient is computed.

Twenty safety/risk units selected outside the 12 historical FOACPO protected
units form the method-specific calibration panel. With `alpha=0.05`, the frozen
split-conformal rank is 20 and the complete candidate-or-fallback policy has a
`20/21` marginal protection lower bound under exchangeability. FOACPO-S is
replayed on the same panel as a frozen ablation.

This is not a fresh-data confirmation: the source pool predates the method and
other Step3 studies may have used parts of it. Behavior calibration and
mechanism-dev are already-open descriptive gates. A pass establishes only
zero-step geometric feasibility and method-specific transport evidence. It
does not establish finite-step or generated-response efficacy.

The run must preserve `optimizer_steps=0`, `training_authorized=false`, and keep
Core100 and heldout closed. A failure stops the method; it cannot authorize
training.
