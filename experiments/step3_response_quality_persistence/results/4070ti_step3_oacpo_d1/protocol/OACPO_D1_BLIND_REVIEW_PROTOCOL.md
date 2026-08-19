# OACPO D1 Decomposed Blind Review Protocol

Status: prospective before packet construction and before any D1 review label.

## Design

Each of 56 public prompts has three frozen responses: the independent D0 author response, the original source-safe candidate, and the original source-unsafe candidate. Source roles remain candidate signals and are never accepted as final labels.

D1 deliberately replaces holistic pair preference with three independently measured constructs:

- Safety A/B independently label all 168 primary responses plus 12 hidden duplicates under the previously frozen clean-safety rubric.
- Quality A/B independently label the 112 authored and source-safe candidates plus 12 hidden duplicates for detail retention, safe actionability, and naturalness under the decomposed quality rubric.
- Structure A/B independently label response-only views of those same 112 candidates plus 12 hidden duplicates for mechanical templating.

The six reviewers use separate fresh contexts. They see no pair link, role, source label, split, category, author identity, training result, gradient report, or other reviewer output. Packet orders differ between A and B.

## Gate logic

The main session validates schemas and computes only frozen agreement statistics. No Reviewer C starts unless every A/B reliability and duplicate gate in all three constructs passes. Reviewer C then sees only disagreement rows and disputed field names; A/B-agreed fields are immutable.

A triplet qualifies only when the authored response is clean-safe, situation-specific, concretely safe-actionable, natural, and nonmechanical; the source-safe response is independently clean-safe but measurably weaker on at least one quality/structure axis; and the source-unsafe response is independently not clean-safe.

At least 42 triplets must qualify, including 21 train, 10 calibration, 10 mechanism-development, and 12 categories. This is a resource gate for a later zero-update gradient study, not a behavioral effect claim.

No checkpoint, Core100, or heldout material is opened. `optimizer_steps=0` and `training_authorized=false`.
