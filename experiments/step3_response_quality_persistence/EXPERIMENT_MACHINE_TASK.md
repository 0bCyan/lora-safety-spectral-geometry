# Experiment Machine Task: Step3Q API Authoring And P2 Review Flow

## Current State

There is no GPU or optimizer task. The active branch is
`codex/step3q-calibrated-api-pool` and the route is
`step3q_calibrated_api_pool_attempt_001`.

Historical boundaries remain unchanged:

```text
Step3C-T6:      R1_no_c3_candidate_passes
Track B:        M1_reconstruction_integrity_failure
Step3N-R1:      STOP before tuple writing or training
Step3O O3:      PASS external actionability validation
Step3O-O4/O4R:  both closed at STOP_AUTHORING
Step3P:         STOP_REVIEW_UNRELIABLE before C, target selection, or training
Step3Q:         prospective calibrated API pool; optimizer_steps=0
```

P1 has been deterministically materialized from P0 commit
`510c1a186c214655273a05826acaba2a9e1a2264` and passed exact reconstruction
under namespace
`f2fa3257d5664b1c5d6747904e301968676fbad95ea0724c76a0589861866356`.
No external API request has been sent.

P2 receipt import, role-blind review packet construction, calibration-first
analysis, disagreement-only Reviewer C, and mode-blind 48-target selection are
now prospectively frozen under tag
`step3q-calibrated-api-attempt-001-p2-protocol`. The tag must be visible on the
remote before dispatch.

Step3Q does not rerun Step3P and does not test a method. It asks whether a
calibrated blind-review design plus controlled structured-API authoring can
produce a fixed 48-target training pool. Even a final pool pass leaves
`training_authorized=false`.

## Release Gates

Do not start an external API call until all of the following are true:

1. tag `step3q-calibrated-api-attempt-001-p0` exists on the remote;
2. tag `step3q-calibrated-api-attempt-001-p1` exists on the remote;
3. tag `step3q-calibrated-api-attempt-001-p2-protocol` exists on the remote;
4. the branch is clean at the one-child P2 registry-lock commit, whose frozen
   `protocol_commit_sha1` exactly matches the P2 protocol tag;
5. `verify-prepared` and the P2 preflight pass exact deterministic reconstruction;
6. the designated machine has `OPENAI_API_KEY` in its process environment;
7. the receipt spool is outside the Git repository and does not already
   contain a sealed result from another attempt.

If any gate fails, stop without network access. Do not repair, replace, top up,
or regenerate prompts or requests.

## Pull And Verify

Use a clean checkout and fetch both branch and tags:

```text
git fetch origin codex/step3q-calibrated-api-pool --tags
git switch codex/step3q-calibrated-api-pool
git pull --ff-only origin codex/step3q-calibrated-api-pool
git status --short
git rev-parse step3q-calibrated-api-attempt-001-p0^{commit}
git rev-parse step3q-calibrated-api-attempt-001-p1^{commit}
git rev-parse step3q-calibrated-api-attempt-001-p2-protocol^{commit}
git rev-parse HEAD
git rev-list --count step3q-calibrated-api-attempt-001-p2-protocol..HEAD
git rev-list --parents -n 1 HEAD
git diff --name-only step3q-calibrated-api-attempt-001-p2-protocol HEAD
```

Run the frozen checks before reading or using any request packet:

```text
python -m py_compile experiments/step3_response_quality_persistence/scripts/step3q_calibrated_api_pool.py
python -m py_compile experiments/step3_response_quality_persistence/scripts/step3q_p2_review.py
python -m unittest discover experiments/step3_response_quality_persistence/scripts -p test_step3q_calibrated_api_pool.py -v
python -m unittest discover experiments/step3_response_quality_persistence/scripts -p test_step3q_p2_review.py -v
python experiments/step3_response_quality_persistence/scripts/step3q_calibrated_api_pool.py preflight
python experiments/step3_response_quality_persistence/scripts/step3q_calibrated_api_pool.py verify-prepared
python experiments/step3_response_quality_persistence/scripts/step3q_p2_review.py preflight
```

Expected fixed P1 denominators are:

```text
fresh prompts:        60
API requests:         120 (two modes per prompt)
calibration rows:     96
component anchors:    48
prompt mismatches:    24
clean-safety anchors: 24
per-reviewer sentinels: 8 paired component + 8 mismatch + 8 clean-safety (4 yes/4 no)
optimizer steps:      0
```

`verify-prepared` must report 60 prompts, 120 requests, 96 calibration rows,
and no external call. A missing, added, or one-byte-modified P1 file is a hard
stop. The final `git diff --name-only` above must list exactly
`experiments/step3_response_quality_persistence/configs/step3q_p2_review_registry.json`;
the count must be `1`, and HEAD's sole parent must be the tagged protocol
commit. That child writes the tagged implementation SHA into the registry and
changes no field other than `protocol_commit_sha1` and
`protocol_lock_status`.

## External Dispatch

The spool path must be a new durable directory outside this repository. Do not
put the API key in the command, a file, shell history, Git config, or a commit.

```text
python experiments/step3_response_quality_persistence/scripts/step3q_calibrated_api_pool.py dispatch --spool-root <outside-repository-step3q-spool>
```

The dispatcher is fixed to:

- OpenAI Responses API;
- requested model `gpt-5.6-sol`;
- medium reasoning, strict JSON schema, `store=false`, and no tools;
- 60 fixed prompts and two independent modes per prompt;
- at most four concurrent calls;
- at most three attempts only for registered transport/HTTP failures;
- no semantic retry, prompt replacement, or top-up;
- immutable per-request receipts and deterministic global response deduplication.

An HTTP-200 body that is malformed, refused, out of the 40-160-word envelope,
or otherwise semantically invalid is one terminal denominator failure. It is
not regenerated. The returned model must be the exact alias or a dated
`gpt-5.6-sol-YYYY-MM-DD` snapshot, and all valid candidates must use one
effective model.

## Authoring Decision

The fixed dispatch passes only if all conditions hold:

```text
terminal receipts:             120 / 120
valid globally unique targets: >= 96
represented prompts:           >= 54 / 60
effective model identities:    exactly 1
protocol/model failures:        0
```

Otherwise the route is `STOP_API_YIELD`. Do not add calls, change word limits,
discard failed receipts, or start review.

After completion, preserve the entire external spool for every later P2 stage.
Do not manually edit or commit it. Report the printed dispatch summary.

## P2 Import And Packet Freeze

Run this only when dispatch printed `PASS_API_AUTHORING`, the repository is
still clean at the P2 registry-lock commit, and the P2 output root does not exist:

```text
python experiments/step3_response_quality_persistence/scripts/step3q_p2_review.py import-dispatch --spool-root <outside-repository-step3q-spool>
```

The importer revalidates all 120 receipts, reconstructs the three sealed files
in a temporary directory, and requires byte equality. It commits no raw receipt
body. Expected output is
`results/step3q_calibrated_api_pool_attempt_001_p2_review/` containing the
verified sealed package, hidden primary metadata, private ID maps, reviewer
packets, templates, and an exact manifest.

After a successful import, run the P2 tests again, confirm that no path outside
that result root changed, then commit and push:

```text
git add experiments/step3_response_quality_persistence/results/step3q_calibrated_api_pool_attempt_001_p2_review
git commit -m "数据：冻结Step3Q P2 API导入与盲审包"
git tag step3q-calibrated-api-attempt-001-p2-packets
git push origin codex/step3q-calibrated-api-pool
git push origin step3q-calibrated-api-attempt-001-p2-packets
```

Do not start A/B analysis from an uncommitted import bundle.

## Independent A/B Review

Reviewer A and Reviewer B must be fresh, distinct, context-isolated sessions.
Give each reviewer only its own visible packet, review template, provenance
template, and the frozen rubric. Do not give either reviewer the repository,
the other packet, any hidden map, API mode, category/severity key, or prior
review.

After valid A/B files are delivered outside the repository, use a clean
checkout containing the committed packet tag:

```text
python experiments/step3_response_quality_persistence/scripts/step3q_p2_review.py analyze-ab --spool-root <outside-repository-step3q-spool> --review-a <reviewer-a.jsonl> --provenance-a <reviewer-a-provenance.json> --review-b <reviewer-b.jsonl> --provenance-b <reviewer-b-provenance.json>
```

This command reruns receipt and import reconstruction before reading review
inputs. If either calibration or 11/12 duplicate gate fails, it writes
`STOP_REVIEW_CALIBRATION` and no primary metrics. If calibration passes but any
primary reliability gate fails, it writes `STOP_REVIEW_UNRELIABLE`. Only
`PASS_AB_REVIEW` creates a Reviewer C packet, and that packet contains every
and only derived qualification disagreement.

Commit and push the immutable `ab_outcome/` before any C review or finalization.
A suggested tag is
`step3q-calibrated-api-attempt-001-p2-ab-outcome`.

## Reviewer C And Finalization

For a non-empty C packet, use a third fresh context-isolated reviewer and give
it only the C packet, template, provenance template, and rubric. Then run:

```text
python experiments/step3_response_quality_persistence/scripts/step3q_p2_review.py finalize --spool-root <outside-repository-step3q-spool> --review-c <reviewer-c.jsonl> --provenance-c <reviewer-c-provenance.json>
```

For an empty C packet, omit both C arguments. Finalization reruns the receipt,
import, and complete A/B outcome reconstructions before adjudication. A pass
selects exactly 48 unique prompts with all 19 pre-assigned categories and
severity minima 5/28/5. The selector never receives API mode. Even
`PASS_CALIBRATED_API_POOL` leaves training closed.

## Hard Prohibitions

- No GPU training, generation, scoring, gradient extraction, or optimizer step.
- No Core100 or heldout access.
- No Step3P Reviewer C, hidden-yield analysis, or final flow.
- No reuse of O4/O4R outputs or Step3P candidates.
- No prompt replacement, third authoring mode, semantic retry, or post-result
  threshold change.
- No manual candidate labels from a session that has seen API mode or hidden
  calibration roles.
- No trusting a self-consistent import or A/B manifest without the preserved
  raw spool and deterministic reconstruction.
- No API key, raw spool, checkpoint, adapter, cache, or unrelated paper file in
  Git.

The experiment machine remains idle until both P0 and P1 tags are visible.
