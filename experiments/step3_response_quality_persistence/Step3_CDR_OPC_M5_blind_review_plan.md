# Step3 CDR-OPC M5 Independent Blind Review

## Purpose

M5 tests target qualification on the prompt-disjoint authoring cohort sealed at
commit `6c6f3b7`. This stage measures whether the two independently authored safe
responses for each of sixty new risk prompts satisfy the frozen M3 V2 target
qualification rubric. It does not calculate gradients, tune a dose, train a
model, or inspect Core100 or heldout data.

## Frozen packet

Two no-context reviewers independently receive 180 role-blind rows:

- 120 M5 primary candidates;
- 12 positive component anchors;
- 12 deterministic generic negative anchors;
- 12 deterministic cross-category mismatch anchors;
- 12 clean-safety negative anchors;
- 12 hidden primary duplicates.

The anchor sources, label semantics, reliability statistics, and thresholds are
identical to M3 blind review V3. Packet IDs and order are newly namespaced by
the M5 protocol lock, authoring seal, and registry hash. Reviewer A and Reviewer
B may read only their packet, the copied rubric, and the copied finalizer. They
must not read the repository, hidden maps, another review, prior labels,
training results, gradient results, or aggregate outcomes. Each reviewer gets
one immutable submission; replacement and post-submission revision are
forbidden.

Reviewer C is created only when both A and B pass all anchor, duplicate, and
agreement gates and their mechanically derived qualification labels disagree
for at least one primary candidate. C sees only the disputed prompt-response
pairs and the two short reasons.

The review uses the single frozen external root
`D:/Desktop/step3-cdro-opc-m5-blind-review-attempt-001`. The protocol commit and
its one-child lock commit receive separate tags. Every later command requires
that exact lock-tagged HEAD and a clean worktree, reconstructs the base packet
from the frozen authoring inputs, and byte-compares it with the external copy.
A/B and final outputs are also re-derived from the immutable review submissions
before Reviewer C authorization, finalization, or repository import. Exact
partial phase writes may be resumed; a different byte sequence is rejected.

## Frozen decision

Qualification is derived mechanically from the six rubric fields. A prompt is
eligible only when both of its authored candidates are finally qualified. At
least forty eligible prompts are required. A deterministic prospective order
selects forty prompts and alternates them into twenty calibration and twenty
sealed-audit units.

Passing this review authorizes only a separately frozen zero-step GPU protocol.
That later protocol must replay the already chosen conservative dose
`0.0337860296353059`; it may not tune the dose from M5 labels or gradients.

```text
optimizer_steps=0
training_authorized=false
core100_opened=false
heldout_opened=false
```
