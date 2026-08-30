# P6 — Prompt-Disjoint Confirmatory Panel

Status: sealed design; panel allocation pending.

## Purpose

Confirm the frozen Paper A analysis on prompts not used for training, schedule selection, rubric development, reviewer calibration, or exploratory Step 3 method work.

## Preservation of the existing global heldout

The preferred plan is to create or allocate a Paper A-specific prompt-disjoint panel and leave the existing global heldout untouched for the future method paper. If the existing heldout is allocated to Paper A, that decision must be irreversible and the set may never later be represented as unseen for a method paper.

## Opening prerequisites

Before any confirmatory prompt or response is opened:

- P1–P5 data complete or formally stopped;
- claims and smallest effects of interest frozen;
- checkpoint-selection rule frozen;
- all exclusions and missingness rules frozen;
- analysis scripts, tables, and figures frozen;
- manuscript introduction and methods frozen except result values;
- commit SHA and source hashes recorded.

## Execution

- generate all registered arms with identical decoding;
- construct opaque review packets once;
- run frozen safety and quality rubrics;
- join arm keys only after reliability passes;
- run the frozen analysis exactly once;
- publish all registered endpoints, including failures.

## No rescue

No prompt replacement, threshold relaxation, additional reviewer, alternate score, or post-result model/seed selection is permitted.
