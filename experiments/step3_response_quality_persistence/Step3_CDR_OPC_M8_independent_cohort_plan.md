# Step3 CDR-OPC M8 Prospective Independent-Cohort Validation

## Purpose

M6 stopped at 16/20 protected calibration units on the M5 cohort. M7 and its
VE1 erratum measured an even finite-radius component but did not establish a
causal or Hessian explanation. M8 therefore asks a narrower prospective
question: does the unchanged CDR-OPC direction, dose, and virtual radius pass
the same finite-radius gates on one newly authored and independently reviewed
cohort?

M8 is a post-result hypothesis test inspired by M6/M7. It is not a repair or a
rerun of M6. M6/M7 row-level losses, gradients, failed-unit identities, and
decomposition values may not enter M8 selection, review, splitting, thresholds,
or runner inputs.

## M8-A: frozen prompt selection and authoring

The fixed risk source is filtered by both sample ID and normalized prompt hash.
All M3 primary candidates are excluded. All M5 primary candidates are excluded;
M4 reused M3, while M6 and M7 reused the M5 review cohort and introduced no new
authoring prompts. Normalization is NFKC, case-folding, and whitespace collapse.

Eligible rows are ordered by SHA256 of the frozen salt, a separator, and the
sample ID. The first 60 distinct normalized prompts are selected without
balancing, replacement, or result-dependent top-up. Each prompt receives the
two frozen M5 modes, specific_redirect and risk_mitigation, for 120 requests in
12 immutable shards.

Each shard is authored once by a no-context current-model sub-agent. Responses
must satisfy the existing authoring rubric, exact registered opening, and hard
40--160 whitespace-token envelope. The main session may transfer bytes and run
mechanical validators but may not write, label, revise, or replace a response.

## M8-R: blind review and cohort split

If M8-A passes, M8-R reuses the M5 role-blind A/B packet construction, rubric,
anchors, hidden duplicates, and reliability thresholds. A and B each submit
once. Reviewer C is allowed only when A/B pass every reliability gate and the
existing mechanical disagreement rule requires adjudication. No reviewer may
read repository state, hidden maps, another review, or M6/M7 measurements.

A prompt is eligible only when both authored routes qualify. At least 40
prompts are required. Eligible opaque prompt IDs are ordered by the frozen M8
selection hash; the first 40 are selected and alternating ranks create exactly
20 calibration and 20 sealed-audit units. Reviewer confidence and all M6/M7
failure patterns are forbidden split inputs.

## M8-G: fixed finite-radius validation

M8-G performs one negative direct virtual displacement only. It inherits the
M6 direction, dose 0.0337860296353059, eta 0.0002, four protected certificates,
score floor 0.02, allowed deterioration 0.04, behavior-ratio bounds 0.25--1.75,
and realized-map tolerances. There is no mirrored positive arm, endpoint arm,
radius grid, dose adjustment, rescue shrink, or optimizer step.

Calibration is evaluated first. Sealed audit may be opened once, in the same
unchanged process, only if every common and calibration gate passes. Each split
requires at least 19/20 protected units, every certificate mean deterioration
at most 0.04, positive predicted and actual behavior improvement, a ratio in
the frozen interval, finite values, exact state restoration, and map fidelity.
Any crash after the GPU attempt claim stops attempt-001 without rerun.

## Claim boundary

A pass supports only that the unchanged map met the preregistered finite-radius
and behavior gates on one new prompt-disjoint author/review cohort. It does not
erase the M6 stop, establish universal tail protection, prove the M7 diagnostic
causal, test a real optimizer update, authorize training or checkpoints, or
open Core100/heldout data. A disagreement between M6 and M8 must be reported as
cross-cohort inconsistency, not converted into a pass for M6.
