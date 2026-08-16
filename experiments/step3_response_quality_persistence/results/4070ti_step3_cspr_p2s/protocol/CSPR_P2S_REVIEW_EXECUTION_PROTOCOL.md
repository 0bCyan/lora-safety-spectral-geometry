# CSPR P2S Review Execution Protocol

Status: prospective before any CSPR P2S review label is created or opened.

This layer freezes execution of the calibration-separated P2S review. It does not change the authored candidate pool, packets, rubrics, hidden anchors, duplicate placement, reliability thresholds, bootstrap settings, pair qualification rule, fixed-pool yield gates, or any training rule. The execution script and its synthetic tests are hash-bound before reviewer work begins.

## Independent A/B Review

Four distinct fresh no-context agents must complete the full packets: Safety A and B each receive 424 rows; Actionability A and B each receive 312 rows. An agent may read only its assigned visible packet, blank template, and construct rubric. Each role must have a unique agent ID and session ID, and no role may reuse either P1S author or any old P2 reviewer listed in the JSON protocol.

Hidden keys, calibration identity, reference labels, target role, author role, split, category, another reviewer's output, old labels or conclusions, training or gradient results, target generation scripts, and git history are forbidden. The main session validates structure and provenance and computes frozen metrics; it does not judge, repair, infer, or replace any label.

Calibration agreement and each reviewer's concordance against frozen balanced references are computed only on calibration primaries. Target A/B reliability is computed only on target primaries. Hidden duplicates are evaluated separately. Target score prevalence is descriptive and is never a reviewer reliability gate. Actionability calibration uncertainty uses 10,000 frozen bootstrap replicates.

## Reviewer C

Reviewer C is authorized only after all calibration, reference-concordance, target-reliability, bootstrap, and duplicate gates pass. A new agent and session are required for each construct. C receives only visible prompt/response rows from target A/B disagreements and the unchanged rubric; calibration disagreements are resolved by the frozen reference and are never sent to C. C sees neither A/B labels nor their reasons. Only fields disputed by A and B may use C's label; agreed fields are immutable.

## Stop Rules

A safety-policy block or any inability to complete an assigned packet records `STOP_CSPR_P2S_REVIEW_INCONCLUSIVE`. Partial labels, local heuristics, silent replacement reviewers, threshold changes, and replacement candidates are forbidden. Invalid artifacts stop at `STOP_CSPR_P2S_REVIEW_INVALID`; calibration or duplicate failure stops at `STOP_CSPR_P2S_CALIBRATION_UNRELIABLE`; target reliability failure stops at `STOP_CSPR_P2S_TARGET_UNRELIABLE`; fixed-pool yield failure stops at `STOP_CSPR_P2S_PAIR_YIELD`.

Finalization emits six frozen result artifacts: target safety labels, target actionability labels, the 120-prompt candidate-selection ledger, final analysis, route decision, and a readable report. A pass can authorize only a separately frozen P3 zero-update mechanism protocol. It is not evidence that training works and does not authorize a model gradient, checkpoint experiment, or optimizer step. Throughout P2S, `optimizer_steps=0`, `training_authorized=false`, and core100 and heldout remain closed.
