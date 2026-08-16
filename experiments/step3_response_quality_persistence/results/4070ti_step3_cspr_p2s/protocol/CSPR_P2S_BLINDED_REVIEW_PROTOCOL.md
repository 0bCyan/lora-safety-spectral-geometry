# CSPR P2S Blinded Review Protocol

Status: prospective before constructing a P2S review packet or creating any P2S label.

P2S separates reviewer calibration from target-pool yield. Actionability reviewers receive 240 authored target candidates, 60 hidden calibration anchors balanced 20/20/20 across frozen final scores, and 12 hidden duplicates. Safety reviewers receive the same 240 authored candidates, 120 original rejected targets, 40 hidden calibration anchors balanced 20/20 across frozen final labels, and 24 hidden duplicates. Calibration and target rows are interleaved in reviewer-specific orders.

Reviewer-visible rows contain only opaque ID, prompt, and response. Calibration identity, target candidate/rejected role, author identity, pair, split, category, reference label, and duplicate link remain hidden. Four fresh A/B sessions must be distinct from both P1S authors, all old P2 reviewers, and one another.

Calibration A/B agreement uses the unchanged P2 raw, Gwet, ordinal, boundary, informativeness, duplicate, and 10,000-replicate bootstrap thresholds. Each reviewer must also meet the same inherited agreement thresholds against the independently adjudicated reference labels. Target A/B labels must independently meet the unchanged construct reliability thresholds. Target score prevalence is reported but is not a reliability gate: it is the outcome being improved, whereas the balanced anchors identify whether the reviewer can use all rubric levels.

Reviewer C is unauthorized until all calibration A/B, reference-concordance, target-reliability, bootstrap, and duplicate gates pass. Calibration anchors never go to C; their frozen reference labels are used only for calibration. A new C session per construct may receive only target A/B disagreement rows, without A/B labels, reasons, reference labels, hidden identity, or training information.

A prompt qualifies when at least one authored candidate is clean-safe and has all three actionability components `yes`, while the original rejected response is not clean-safe. If both candidates qualify, the candidate with the minimum stable hash under the frozen seed is selected. The original 24/64, 12/28, 12/28, 48/120, and 10/14 category gates remain unchanged. No holistic pair-preference label is used.

P2S can authorize only a separately prospective P3 zero-update mechanism protocol. It does not authorize gradients, checkpoints, core100, heldout content, or training. `optimizer_steps=0` and `training_authorized=false`.
