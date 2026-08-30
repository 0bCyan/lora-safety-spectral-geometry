# P4 — Human Validity Study

Status: prospective.

## Purpose

Test whether the final component rubric has validity beyond internally consistent model-review sessions.

## Sampling

The human subset is selected before model-review outcomes are opened. Selection uses a frozen hash over prompt ID, seed, checkpoint, and opaque response ID, stratified by arm and harm category. Outcome-dependent sampling is forbidden.

Target size:

- preferred: all 100 common safety prompts for the final checkpoint and principal arms;
- minimum: a preregistered stratified subset with at least 200 arm-response judgments and adequate positive/negative prevalence for each component.

## Reviewers

- two independent primary human annotators;
- one adjudicator for registered disagreements;
- no annotator sees arm or seed identity;
- annotators complete training examples and a qualification packet before target access.

## Registered outputs

- per-component raw agreement;
- Gwet AC1/AC2 or another prevalence-robust coefficient selected before labels;
- strict-boundary agreement for the derived score;
- model-human agreement by component;
- disagreement taxonomy;
- time and exclusion log.

## Minimum validity gate

The final numeric minima are stored in `configs/human_validation_registry.json` and frozen before packet release. Failure does not trigger rubric tuning on the same target set; it lowers the claim tier and is reported as a measurement limitation.

## Data protection

Only de-identified IDs and labels are committed. Annotator names, contact information, and unredacted notes are not stored in the repository.
