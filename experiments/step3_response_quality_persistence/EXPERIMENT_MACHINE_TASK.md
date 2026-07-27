# Experiment Machine Task: Step3Q Controlled API Authoring

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

Step3Q does not rerun Step3P and does not test a method. It asks whether a
calibrated blind-review design plus controlled structured-API authoring can
produce a fixed 48-target training pool. Even a final pool pass leaves
`training_authorized=false`.

## Release Gates

Do not start an external API call until all of the following are true:

1. tag `step3q-calibrated-api-attempt-001-p0` exists on the remote;
2. tag `step3q-calibrated-api-attempt-001-p1` exists on the remote;
3. the branch is clean and contains the P1 tag commit;
4. `verify-prepared` passes exact deterministic reconstruction;
5. the designated machine has `OPENAI_API_KEY` in its process environment;
6. the receipt spool is outside the Git repository and does not already
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
```

Run the frozen checks before reading or using any request packet:

```text
python -m py_compile experiments/step3_response_quality_persistence/scripts/step3q_calibrated_api_pool.py
python -m unittest discover experiments/step3_response_quality_persistence/scripts -p test_step3q_calibrated_api_pool.py -v
python experiments/step3_response_quality_persistence/scripts/step3q_calibrated_api_pool.py preflight
python experiments/step3_response_quality_persistence/scripts/step3q_calibrated_api_pool.py verify-prepared
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
stop.

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

After completion, preserve the entire external spool. Do not manually edit or
commit it. Report the printed dispatch summary and keep the receipt directory
available for the separately frozen P2 import/review-packet step. P2 must
verify the receipt manifest before moving any candidate into a reviewer packet.

## Hard Prohibitions

- No GPU training, generation, scoring, gradient extraction, or optimizer step.
- No Core100 or heldout access.
- No Step3P Reviewer C, hidden-yield analysis, or final flow.
- No reuse of O4/O4R outputs or Step3P candidates.
- No prompt replacement, third authoring mode, semantic retry, or post-result
  threshold change.
- No manual candidate labels from a session that has seen API mode or hidden
  calibration roles.
- No API key, raw spool, checkpoint, adapter, cache, or unrelated paper file in
  Git.

The experiment machine remains idle until both P0 and P1 tags are visible.
