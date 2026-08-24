# Step3 CDR-OPC M9 Public-Safe Single-Target Validation

## Purpose

M8-A stopped twice before blind review: attempt-001 encountered a filesystem
permission failure and attempt-002 encountered the platform safety-access
boundary after 30/120 responses. Those outcomes do not test CDR-OPC. M9 avoids
another generation attempt by using an already frozen public safe-source pool.

M9 is not a continuation or top-up of M8 authoring. It excludes every prompt
released by M3, M5, and the full M8 registry. It then selects one existing
public safe-source candidate for each of 60 new prompts. The public source was
previously aligned to the exact local risk prompt and admitted only records
marked is_safe=true with response_source=Alpaca3-8B. The prior Step3P review
stopped for reviewer unreliability, so source safety is provenance only; M9
must perform a new independent blind review before any target is eligible.

## Frozen candidate selection

Candidate source rows are filtered by sample ID and normalized prompt hash.
For a prompt with multiple public-safe rows, one row is selected by the frozen
candidate hash. Fresh prompts are then ordered by a second frozen hash and the
first 60 are selected. No previous review label, M6/M7 loss, category, severity,
or response content score enters either ordering.

## Blind review and split

Two context-isolated reviewers receive role-blind packets containing 60 primary
candidates, the same 48 M5 anchors, and 12 hidden duplicates. The M5 rubric and
reliability thresholds remain fixed. Reviewer C is opened only by the existing
mechanical disagreement rule after A/B reliability passes. A prompt is eligible
when its single public-safe candidate qualifies. At least 40 eligible prompts
are required; the first 40 under the frozen hash split alternately into 20
calibration and 20 sealed-audit units.

## Finite-radius boundary

M9 inherits the M6 direction, fixed dose 0.0337860296353059, eta 0.0002, four
protected certificates, score floor 0.02, deterioration threshold 0.04, and
map/behavior gates. It evaluates one target per prompt, so it does not test
two-route branching or route-trie coverage. Calibration must pass every common
and split gate before sealed audit is opened once. There is no optimizer step,
radius grid, dose tuning, positive mirror, endpoint arm, rescue, generation,
checkpoint, Core100, or heldout access.

## Claim boundary

A pass supports only fixed-map finite-radius behavior on one independently
selected, independently reviewed, single-target public-safe cohort. It neither
reverses the M6 stop nor establishes the stronger M5 two-route claim. A stop or
pass cannot be used to repair M8, prove M7 causal, authorize training, or tune
the direction, dose, radius, selection rule, or thresholds.
