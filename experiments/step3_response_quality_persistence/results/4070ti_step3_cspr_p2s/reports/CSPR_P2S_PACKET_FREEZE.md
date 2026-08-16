# CSPR P2S Blinded Packet Freeze

Actionability A and B each receive `312` independently ordered rows: 240 authored targets, 60 balanced hidden calibration anchors, and 12 hidden duplicates. Safety A and B each receive `424` independently ordered rows: 240 authored targets, 120 original rejected targets, 40 balanced hidden calibration anchors, and 24 hidden duplicates.

Reviewer-visible rows contain only opaque ID, prompt, and response. They do not reveal calibration identity, target role, author role, pair, split, category, reference label, or duplicate link. This freeze authorizes only complete independent A/B review under the unchanged rubrics. It does not expose a result, launch Reviewer C, select a pair, compute a gradient, or authorize training. `optimizer_steps=0` and `training_authorized=false`.
