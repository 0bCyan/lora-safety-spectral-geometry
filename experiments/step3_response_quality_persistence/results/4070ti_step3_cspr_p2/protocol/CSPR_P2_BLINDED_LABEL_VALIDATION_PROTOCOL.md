# CSPR P2 Blinded Label-Validation Protocol

Status: prospective before constructing any CSPR P2 reviewer packet.

P1R passed only source integrity and pair feasibility. BeaverTails labels are not final labels. P2 separates the two validated constructs so that the previously unreliable holistic pair-preference judgment is not reused.

Safety reviewers A and B independently review all 240 response items, with pair membership and chosen/rejected role hidden, using the unchanged Step3O clean-safety rubric. Each reviewer receives 24 hidden duplicates, 12 from each hidden source role. Actionability reviewers A and B independently review only the 120 candidate responses using the unchanged Step3O component rubric. Each receives 12 hidden duplicates. Reviewer-specific row orders are deterministic and different.

Safety A/B reliability requires raw agreement at least `0.90`, Gwet AC1 at least `0.80`, positive and negative agreement at least `0.75`, and at least 23 of 24 exact duplicate labels for each reviewer. Actionability uses the already validated Step3O component, derived ordinal, score-2 boundary, duplicate, and bootstrap gates without weakening. Every A/B disagreement requires a new blinded Reviewer C who sees only the disagreement packet and rubric for that construct.

A pair qualifies only after reliable review and adjudication when its candidate response is `clean_safe=yes`, `safe_route=yes`, `executable_step=yes`, and `prompt_binding=yes`, while its rejected response is `clean_safe=no`. Source labels, source roles, categories, and prospective splits remain hidden until all final labels freeze.

Before any new label is opened, the fixed-pool yield gate is 24 of 64 training candidates, 12 of 28 calibration candidates, 12 of 28 mechanism-development candidates, at least 48 of 120 overall, and at least two qualified pairs in at least 10 of 14 categories. These are resource and diversity requirements for a later zero-update mechanism study, not a power calculation or population-rate claim. Reviewed pairs cannot be replaced.

P2 packet freeze does not launch reviewers or inspect labels. It does not compute model gradients or authorize training. Passing review and yield can authorize only a separately frozen real-model zero-update mechanism protocol.
