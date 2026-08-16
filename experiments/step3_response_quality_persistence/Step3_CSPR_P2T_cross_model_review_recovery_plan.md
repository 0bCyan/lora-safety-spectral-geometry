# Step3 CSPR P2T Cross-Model Review Recovery Plan

## 1. Decision

CSPR P2S stopped at `8f80ccb` because Actionability Reviewer B was interrupted by a model-side biological-risk safety filter after writing 276 of 312 rows. Actionability A completed, but the complete A output and partial B output are excluded from all P2T reliability, adjudication, qualification, yield, and route calculations. P2S is an instrument-access failure, not evidence for or against CSPR.

P2T is one prospective recovery attempt modeled on the successful Step3C-T5H recovery. It changes reviewer qualification and model diversity only. It does not change, split, sanitize, redact, paraphrase, reorder, regenerate, or selectively omit any packet row. It leaves the P2S rubrics, hidden anchors, duplicate allocation, target pool, thresholds, bootstrap settings, pair qualification rule, fixed-pool yield gates, and hidden keys byte-identical.

## 2. Frozen Evidence

P2T binds the following before any new reviewer is launched:

- the P2S inconclusive incident and its `STOP_CSPR_P2S_REVIEW_INCONCLUSIVE` route;
- the P2S review protocol and packet manifest;
- all eight visible packet/template files and two hidden keys registered by that manifest;
- the P2S review analyzer and tests;
- the completed Actionability A labels/provenance and partial Actionability B labels as excluded audit evidence.

The excluded P2S labels remain preserved but may never be copied, resumed, pooled, compared, used for calibration, or joined to a hidden key in P2T. Every P2T reviewer uses a new agent and session.

## 3. P0: Recovery Freeze

P0 freezes this plan, the machine-readable protocol, the zero-access qualification implementation, and its tests. P0 preflight must verify every frozen input hash, every packet-manifest artifact, the P2S stop route, the explicit exclusion of old labels, Git ancestry, and the zero-training boundary.

The only P0 pass route is `PASS_CSPR_P2T_P0_PREFLIGHT`. It authorizes zero-access Safety reviewer qualification only. It does not authorize a reviewer to open a packet.

## 4. P1: Safety Reviewer Qualification

Two new no-context sub-agents are prospectively assigned:

- Safety A: requested runtime `gpt-5.5`;
- Safety B: requested runtime `gpt-5.6-sol`.

Before P1 is committed and pushed, neither session may receive a repository path, packet excerpt, rubric, prompt, response, hidden key, report, or study row. Each receives only its role, requested runtime, row-count commitment, complete-or-abort contract, independence constraints, and a request for model identity and assent.

Qualification requires distinct agent and session IDs, `fork_context=false`, zero repository material opened, zero study rows accessed, no prior involvement, no reviewer contact, and agreement to complete all 424 rows without heuristic substitution or partial acceptance. The two requested models must differ. All P1 identities are added to later forbidden-reuse sets.

P1 success is `P2T1_TWO_SAFETY_REVIEWERS_QUALIFIED`. P1 must also freeze and push a hash-bound Safety execution protocol before either reviewer receives study material. Qualification failure is `STOP_CSPR_P2T_QUALIFICATION` and launches no replacement.

## 5. P2: Safety A/B First

Only after P1 and its execution protocol are pushed may each qualified Safety reviewer read exactly its role-specific unchanged packet, blank template, and unchanged safety rubric. Each reviewer receives one complete attempt. An incomplete, blocked, malformed, reordered, mixed-session, or unprovenanced file is rejected in full and selects `STOP_CSPR_P2T_REVIEW_INCONCLUSIVE`.

Both complete 424-row outputs and provenance records must freeze before A/B comparison. No Actionability reviewer is qualified or launched before the Safety A/B freeze is committed and pushed. Main-session labeling, continuation from partial output, retry shopping, replacement reviewers, packet transformation, and heuristics are forbidden.

## 6. P3-P4: Actionability Qualification and A/B

After complete Safety A/B freeze, two additional fresh no-context cross-model sessions are qualified under the same zero-access contract for Actionability A and B. Their requested runtimes are again `gpt-5.5` and `gpt-5.6-sol`, with new agent and session IDs distinct from all prior authors and reviewers.

After the Actionability qualification and execution package is pushed, each reviewer receives exactly its unchanged 312-row role-specific packet, blank template, and unchanged actionability rubric. Any access, completion, provenance, or policy failure stops P2T without replacement.

## 7. P5: Frozen A/B Analysis and Reviewer C

Only four complete A/B outputs authorize the unchanged P2S analysis definitions:

- calibration A/B reliability on calibration primaries only;
- per-reviewer concordance against frozen balanced references;
- target A/B reliability on target primaries only;
- 10,000-replicate frozen bootstrap intervals;
- hidden duplicate consistency;
- target score prevalence as descriptive only, never a gate.

Every calibration, reference-concordance, target, bootstrap, and duplicate gate must pass before Reviewer C. Calibration disagreements use frozen references. Reviewer C receives only visible target A/B disagreement rows, never A/B labels or reasons. A new cross-model-independent session is required per construct, and only disputed fields may use C.

## 8. P6: Fixed-Pool Decision

After complete C coverage, final labels qualify the unchanged 120-prompt pool. If both authored candidates qualify, the existing stable-hash rule selects one. The unchanged split and category yield gates determine the route.

Pass authorizes only `AUTHORIZE_CSPR_P3_ZERO_UPDATE_MECHANISM_PROTOCOL_ONLY`. Failure preserves the original threshold failure. A policy block remains inconclusive. No route in P2T authorizes a model gradient, optimizer update, direct training, heldout access, or core100 access.

## 9. Single Recovery Boundary

P2T is the only automated recovery attempt authorized after P2S. If any post-qualification reviewer is blocked, automated model review stops. Future continuation requires an approved external human or compliant evaluation endpoint under another prospective protocol. P2T may not be weakened, split, or retried to obtain a favorable completion.

Throughout P2T, `optimizer_steps=0`, `training_authorized=false`, core100 remains closed, and heldout remains closed.
