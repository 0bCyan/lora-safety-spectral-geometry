# Experiment Machine Task: Step3C-T5R Review Recovery and Mechanism Completion

## Authorization

Pull `exp/step3-response-quality-persistence` and follow only:

```text
experiments/step3_response_quality_persistence/Step3C_T5R_independent_review_recovery_and_mechanism_completion_plan.md
experiments/step3_response_quality_persistence/configs/step3ct_t5r_review_recovery_registry.json
```

Step3C-T3 training and T4 generation are complete. T5 produced zero accepted labels because Safety Reviewer A was blocked by a platform safety gateway. This is an inconclusive reviewer-access incident, not a candidate failure.

The next work has two tracks:

1. recover independent blind review on the unchanged T4 packets and then run T6;
2. complete the missing representation and grouped-gradient diagnostics from the frozen T3 states.

No fresh candidate training or generation is authorized.

## Global Prohibitions

- Do not overwrite T3, T4, or T5 artifacts.
- Do not modify or regenerate any T4 packet, hidden key, duplicate map, template, rubric, row order, or opaque ID.
- Do not reuse failed T5 sessions or any partial T5 output.
- Do not sanitize, redact, paraphrase, split, truncate, or selectively omit packet rows to bypass a gateway.
- Do not let the main executor, a heuristic, a keyword rule, an embedding rule, or a local classifier assign outcome labels.
- Do not open heldout.
- Do not tune eta, temperature, queue, targets, hard multiplier, projection, decoding, or behavior thresholds.
- Do not run C4, C5, seeds 2/3, a second backbone, SpectraSafe, or another candidate.
- Do not commit adapters, checkpoints, caches, full gradient tensors, raw API caches, or unrelated paper files.

## Package P1: Entry and Implementation Freeze

### Entry audit

Record pulled HEAD, environment, GPU, CUDA, torch, transformers, PEFT, and available VRAM. Verify:

- commits `b3375d2`, `1ea3bf6`, and `fb23b7e` are ancestors of HEAD;
- all registry hashes match;
- T4 contains exactly 12 generation manifests and 1,200 generated rows;
- packet rows are safety `630`, quality `330`, and utility `330`;
- T5 accepted files and accepted rows are both zero;
- no partial T5 output is present or reused;
- heldout is closed and no new training/generation occurred.

Hash mismatch or provenance failure selects `T5R0_integrity_or_provenance_stop`.

### Implement before review

Implement and test dedicated tools for:

- T5R entry/hash validation;
- reviewer-output schema and exact-ID validation;
- complete-output freeze manifests and reviewer provenance;
- A/B decision-field comparison without arm-key join;
- disagreement-only C packet construction with no A/B labels or reasons;
- Gwet AC1/AC2, weighted agreement, raw agreement, duplicate consistency, and zero-variance handling;
- adjudication and T6 statistics;
- mechanism representation probes and exact grouped-gradient reconstruction.

Tests must use synthetic fixtures only. Do not inspect study responses while developing tests. Push P1 before a study reviewer or model checkpoint is launched.

## Package P2: Safety A/B Gateway

Preferred route: two independent external human safety reviewers. A model-review route is acceptable only in a lawful policy-compliant environment that can inspect the complete unchanged packet.

For Safety A and B:

- use new reviewer IDs and fresh contexts;
- expose only the safety packet, blank role-specific template, and frozen safety rubric;
- expose no git history, arm identity, training result, hidden key, duplicate link, prior label, or other reviewer output;
- require exactly 630 unique output IDs and the complete schema;
- record exact reviewer/session provenance and input/output hashes;
- freeze both complete files before comparison.

If either reviewer cannot complete, accept zero labels from that failed role, record an incident, select `T5R1_reviewer_access_stop`, and stop. Do not start quality or utility under that launch.

## Package P3: Quality and Utility A/B

Only after complete Safety A/B freeze, launch separate fresh sessions for:

```text
Quality A / Quality B: 330 rows each
Utility A / Utility B: 330 rows each
```

Apply the same independence, complete-coverage, schema, provenance, and freeze requirements. Safety, quality, and utility reviewers must remain separate. Push all six complete A/B outputs before comparison.

## Package P4: Reviewer C and Reliability

After all A/B files are frozen:

1. compare decision fields without joining hidden arm/checkpoint keys;
2. build one disagreement-only C packet per domain;
3. include every row with at least one A/B decision-field disagreement;
4. omit A/B labels, reasons, identities, vote counts, arm/checkpoint fields, and duplicate links;
5. freeze C packet hashes before launch;
6. obtain complete fresh Reviewer C coverage;
7. compute reliability on primary rows and duplicate consistency separately.

Apply every threshold in the plan and registry. Zero-variance chance-corrected agreement is `NA/undefined`, with raw agreement `>=0.95`. Do not join the outcome key if any reliability gate fails.

Select exactly one T5R route:

```text
T5R0_integrity_or_provenance_stop
T5R1_reviewer_access_stop
T5R2_review_completion_or_schema_stop
T5R3_review_reliability_stop
T5R4_complete_labels_authorize_T6
```

Only `T5R4` authorizes T6.

## Package P5: Mechanism Implementation Freeze

This package may proceed while external review is pending, but reviewers must not see its results.

Inventory immutable source states for C0-R, C3a, and C3b at steps 0/16/32/48/64, including adapter, optimizer, scheduler, scaler, queue, target-position, RNG, projection, config, trace, and manifest hashes.

Implement a new output root and no-update representation probes for steps 0/16/32/64. Reuse the existing tested retrieval and effective-rank definitions. Add tests for:

- source-checkpoint immutability;
- exact 40-prompt/80-target coverage;
- cross-arm order equality;
- deterministic rerun equality;
- RNG restoration and disabled diagnostic dropout;
- retrieval and effective-rank fixtures;
- attention/MLP parameter grouping;
- grouped cosine fixtures;
- no full-gradient serialization.

Push implementation and tests before executing a model checkpoint.

## Package P6: Mechanism Execution

### Representation trajectories

For all three arms at steps 0/16/32/64, report:

- top1, top5, MRR, and multi-positive recall@5;
- positive, global-negative, and same-category-hard-negative cosine distributions;
- positive-minus-global and positive-minus-hard prompt-cluster gaps;
- 10,000 bootstrap intervals with seed `2026072405`;
- 100 permutation controls with seed `2026072406`;
- prompt/target effective rank and top singular-value share;
- opaque per-prompt metrics for later behavior association.

### Grouped-gradient completion

- labels 0/16/32: load the matching state and run the exact registered four-microstep diagnostic unit without an update;
- label64: load checkpoint48 and deterministically replay completed steps 49-63 in a disposable directory before capturing the pre-update step64 unit;
- optionally apply disposable update64 only to verify exact frozen checkpoint64 hashes.

Before accepting grouped results, reproduce original global norms, ratios, cosines, hard-active counts, energy shares, draw hashes, queue hashes, traces, and final hashes within frozen tolerances.

Report attention-only, MLP-only, and per-layer binding-safety cosine, plus C3a-C3b direction/residual results. Mechanism results are explanatory and cannot select or rescue a candidate.

Select one mechanism route:

```text
M0_source_state_incomplete
M1_reconstruction_integrity_failure
M2_mechanism_package_complete
```

## Package P7: T6

Only after `T5R4`, join frozen adjudicated labels to hidden arm/checkpoint keys and execute the parent-plan statistics with:

```text
paired unit: prompt ID
bootstrap repeats: 10,000
bootstrap seed: 2026072404
tests: exact McNemar
primary multiplicity: Holm across two C3-vs-C0 comparisons
```

Apply all control, safety, quality, utility, and selection gates exactly as frozen. Report all endpoints and trajectories regardless of direction.

Select exactly one:

```text
R0_integrity_control_or_review_failure
R1_no_c3_candidate_passes
R2_select_C3a_as_Q_star
R3_select_C3b_as_Q_star
```

`R1` closes the current C3 recipe without scans. `R2` or `R3` authorizes only a request for a separate C4/Q*/C5 seed1 plan.

## Commit Boundaries

1. P1 entry and implementation freeze.
2. P2 Safety A/B freeze or access-stop incident.
3. P3 Quality/Utility A/B freeze.
4. P4 C labels, reliability, and T5R route.
5. P5 mechanism implementation freeze.
6. P6 mechanism evidence and route.
7. P7 T6 gate table, route, and next authorization.

Push every boundary before its dependent package.

## Required Return State

Each package must include:

- entry/result commit and exact commands;
- environment or reviewer provenance;
- all source/input/output/code/config/rubric hashes;
- complete gate table and denominators;
- accepted/rejected/partial artifact counts;
- checkpoint/source immutability;
- core100 and heldout flags;
- one route and one authorized next action;
- confirmation that no unrelated paper file or large runtime artifact was committed.
