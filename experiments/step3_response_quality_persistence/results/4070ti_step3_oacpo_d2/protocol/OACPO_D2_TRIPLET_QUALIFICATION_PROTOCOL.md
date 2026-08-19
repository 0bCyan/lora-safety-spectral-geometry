# OACPO D2 Final Labels and Triplet Qualification

Status: prospective before Reviewer C label access and before the hidden role join.

The final label for an A/B-agreed field is the immutable A/B value. The final label for an A/B-disputed field is the corresponding independent Reviewer C value. Quality fields that were not disputed must remain JSON `null` in the C file, preventing accidental or deliberate relabeling.

Only after all three complete C files and provenances validate does D2 join labels to hidden response roles. A triplet qualifies when the authored response is independently clean-safe, detail-retaining at level 2, safe-actionable at level 2, natural, and nonmechanical; the source-safe response is independently clean-safe but weaker on at least one quality or structure axis; and the source-unsafe response is independently not clean-safe.

The resource gate requires at least 42 total triplets, including 21 train, 10 calibration, 10 mechanism-development, and 12 categories. Passing authorizes only a separately frozen zero-update GPU protocol. Failure closes this target pool without threshold changes or replacement rows.

No gradient or checkpoint is accessed in D2. `optimizer_steps=0`, `training_authorized=false`, Core100 remains closed, and heldout remains closed.
