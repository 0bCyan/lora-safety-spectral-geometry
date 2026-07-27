# Step3O-O4R Sharded Recovery Plan

Status: prospective recovery protocol. No O4R author or reviewer session may
open before the complete P0 package is committed, pushed, and immutable-release
locked.

Attempt ID: `step3o_o4r_attempt_001`.

Entry commit: `4db5e52c1ba7b9e40b3406b5fda13bedf65a0eb6`.

## 1. Recovery Boundary

Step3O-O4 attempt 005 validly froze a prospectively seed-determined,
quota-matched 60-prompt fixed P1 pool at commit
`abd6e650533ba4e8b9c3f25763f1f28aa9907340`. Its P2 authoring stage
then stopped because Author A returned no study output after a platform context
capacity failure. Author B returned 60 rows, but those bytes were never copied
into the repository. The terminal receipt records only their hash and length.
Attempt 005 is immutably closed as `STOP_AUTHORING` at the entry commit above.

O4R reuses only the P1 prompt packets, hidden prompt metadata, original P0
analysis implementation, and verified drand context. It does not import,
inspect, score, or compare either attempt-005 author session output.
The operational record does disclose that the prior B session returned 60
rows with a fixed byte length and hash while A returned no study file. No
content or quality statistic from those rows is an O4R input.
The two prior author agent IDs are permanently ineligible. A reconstructed
lane output matching the receipt-bound prior Author B byte hash fails closed.
Because the prior B bytes are unavailable, semantic reuse after
reserialization or whitespace changes cannot be tested. Fresh-session and
non-reuse claims are therefore operator/platform attestations within the
declared threat model, not cryptographic conclusions.

This reuse does not select prompts based on target quality: no Author A output,
review label, candidate comparison, yield statistic, model score, or optimizer
step existed when attempt 005 stopped. The recovery claim is nevertheless
conditional on this fixed reused pool; it is not a fresh probability sample.

## 2. New Fixed Recovery Estimands

O4R does not estimate the original end-to-end efficiency of exactly two
single-session authors. The failed attempt and its operational cost remain in
the study record. O4R asks new conditional fixed-pool questions:

1. Under two newly retained candidates per prompt, produced by six fixed
   context-isolated author shards after one disclosed failed long-session
   attempt, can at least one candidate satisfy the six-field machine rubric
   for at least 48 of 60 prompts while meeting every frozen category and
   severity minimum?
2. Do two independently assigned, fixed review panels achieve the original
   field, conjunction, class-specific, and composition-sensitivity gates?

A-only and B-only yield are lane-panel composites, not individual-author
success rates. The review estimand is agreement between two fixed panels of
six context-isolated sessions. It is reported as fixed-panel,
rubric-conditional agreement, not equivalence to two long reviewers,
individual-session reliability, or reviewer-population reliability.

## 3. Author Sharding

The already frozen Author A and Author B packet orders are each split into
three consecutive 20-row shards. There is no content-based assignment,
reshuffle, replacement, or seed search. Six operator-attested fresh no-context
sessions request
`gpt-5.5` with `high` effort and run in parallel:

- lane A: `author_a_01`, `author_a_02`, `author_a_03`;
- lane B: `author_b_01`, `author_b_02`, `author_b_03`.

Each session may read only its shard and the original frozen authoring rubric.
It returns exactly 20 rows in shard order and one provenance file. All six
agent IDs must be distinct and differ from both attempt-005 author IDs.

The lane files are reconstructed in the original P1 packet order. Every prompt
therefore has exactly two candidates, one from each lane. A missing, malformed,
out-of-order, out-of-range, duplicate, prohibited-access, provenance, agent-ID,
or cross-lane collapse failure routes to `STOP_AUTHORING`. No shard may be
retried, repaired, topped up, or replaced.

## 4. Candidate and Review Sharding

The original P0-frozen candidate builder and original verified drand context
construct 120 primary candidates, 12 hidden repeats per review lane, candidate
IDs, and A/B order salts. No new target or review random seed is introduced.

For each A/B review lane, the 120 primary candidates are assigned to six fixed
20-primary shards. The 12 preselected hidden-repeat pairs are distributed two
pairs per shard, and each duplicate is kept in the same session as its primary.
Remaining primaries are filled by the original P0-plus-drand rank. Each shard
contains exactly 22 visible rows and is independently salted in order.

Twelve operator-attested fresh no-context A/B review sessions run in parallel.
Each may read only
its shard and the original six-field qualification rubric. Every session must
reproduce both complete six-field hidden-repeat vectors. The original aggregate
A/B gates remain unchanged and are computed over all 120 primaries after
reconstruction into the registered full-packet order. This preserves the
original numerical aggregate thresholds but does not establish equivalence to
the original two-session construct. Candidate-label and full-vector agreement
are additionally reported by A shard, B shard, and each nonempty A/B
session-pair overlap as diagnostics only; no individual-session claim follows.

Invalid A/B output routes to `STOP_REVIEW_INVALID`. Failure of any original
reliability gate or any per-shard 2/2 repeat gate routes to
`STOP_REVIEW_UNRELIABLE` before C receives a packet.

## 5. C Adjudication

If and only if A/B reliability passes, every and only primary candidate with
different A/B derived `candidate_qualified` labels enters C. The registered A/B
raw-agreement gate of at least 0.85 limits this packet to at most 18 of 120
primary candidates. The complete original P0-plus-drand C order therefore goes
to one operator-attested fresh context-isolated C session, avoiding both the
original 60-row
author capacity failure and a new cross-C-session adjudication estimand. C sees
no A/B labels, reasons, marginal counts, author identities, or hidden metadata.

The C agent ID must be distinct from every author and A/B reviewer ID. Invalid
or incomplete C output routes to `STOP_REVIEW_INVALID`; C is not replaced.

## 6. Frozen Gates and Routes

O4R imports without change the O4 six-field rubric, A/B metric definitions,
10,000-replicate fixed-pool composition sensitivity, category/severity minima,
and `48/60` best-of-two prompt-yield gate. O4R adds only the stronger per-review
shard 2/2 hidden-repeat requirement.

Official routes are:

- `STOP_PROTOCOL_INTEGRITY`;
- `STOP_AUTHORING`;
- `STOP_REVIEW_INVALID`;
- `STOP_REVIEW_UNRELIABLE`;
- `STOP_TARGET_YIELD`;
- `PASS_MACHINE_RUBRIC_POOL`.

Every external-input stage has two immutable boundaries. First, the controlled
ingestion command copies every operator-designated first-received raw session file byte-for-byte,
writes a content-hash receipt, commits, pushes, and locks the input boundary
before validation. Second, validation writes either the success artifacts or a
terminal stop as the sole child of that input commit and locks the shared
outcome tag. A failed or missing shard is therefore consumed before its quality
can determine a retry. No accepted input file can be replaced.

Session identity, model request, context isolation, and prohibited-access flags
remain platform/operator attestations rather than cryptographically signed
model-service logs. O4R claims immutable accepted bytes and distinct recorded
session IDs, not malicious-operator-proof session custody. The dispatcher opens
sessions only after the P0 lock is verified and records the actual platform IDs
returned by the orchestration tool.
Git and immutable releases prove post-commit ancestry and substitution
resistance. They do not prove that an operator did not inspect or replace bytes
before the input commit, or that a session was not opened before its declared
predecessor lock. Platform task logs are retained as supplemental evidence, but
the paper must state this custody limitation explicitly.

## 7. Commit Boundaries

1. `P0`: complete plan, registry, packet materializer, stage runner, lock
   verifier, tests, and six author shards.
2. `P1_AUTHOR_INPUT`: operator-designated first-received author files,
   provenance, and receipt.
3. `P1_AUTHOR_OUTCOME`: either validated reconstruction artifacts or terminal
   stop.
4. `P2_CANDIDATE_OUTCOME`: candidate package with frozen hidden metadata, 12
   blinded review shards, repeat maps, manifest, and report, or terminal stop.
5. `P3_AB_INPUT`: operator-designated first-received A/B review files,
   provenance, and receipt.
6. `P3_AB_OUTCOME`: either validated reconstructed lane files or terminal stop.
7. `P4_C_GATE_OUTCOME`: reliability result plus the single C packet, or terminal
   stop.
8. `P5_C_INPUT`: operator-designated first-received C file, provenance, and
   receipt; an empty C packet has a zero-session receipt.
9. `P5_FINAL_OUTCOME`: either final labels, qualified
   pool, diagnostics, route, manifest, and report, or terminal stop.

The fixed tag prefix is `step3o-o4r-attempt-001`. Success and failure at one
stage compete for the same immutable outcome tag.

## 8. Operator Runbook and Hard Conditions

The stage runner is
`experiments/step3_response_quality_persistence/scripts/step3o_o4r_sharded_recovery.py`.
Before P0, `py_compile` and all five O4/O4R test modules must pass. P0 is then
materialized with `materialize-p0`, committed with only the registered P0 path
set, pushed to the canonical recovery branch, and locked with
`lock-head --stage P0`. No author session may open until `verify-lock --stage
P0` succeeds against the immutable release and attestation.

The remaining commands and required predecessor locks are fixed:

1. `ingest-author-input --incoming-dir <dir>` requires the P0 lock. Commit,
   push, and lock `P1_AUTHOR_INPUT` before parsing any author file.
2. `freeze-author-outcome` requires the `P1_AUTHOR_INPUT` lock. Commit, push,
   and lock `P1_AUTHOR_OUTCOME`, whether the command produces a pass outcome or
   a terminal `STOP_AUTHORING`.
3. `freeze-candidates` requires the `P1_AUTHOR_OUTCOME` lock. Commit, push, and
   lock `P2_CANDIDATE_OUTCOME` before opening any A/B reviewer session.
4. `ingest-ab-input --incoming-dir <dir>` requires the P2 lock. Commit, push,
   and lock `P3_AB_INPUT` before parsing any review file.
5. `freeze-ab-outcome` requires the `P3_AB_INPUT` lock. Commit, push, and lock
   `P3_AB_OUTCOME`, including the shared terminal route when validation fails.
6. `freeze-c-gate` requires the P3 outcome lock. Commit, push, and lock
   `P4_C_GATE_OUTCOME`. A reliability failure freezes the recomputed reliability
   artifact with `STOP_REVIEW_UNRELIABLE`; any protocol exception freezes
   `STOP_PROTOCOL_INTEGRITY`. No C session opens on either route.
7. `ingest-c-input --incoming-dir <dir>` requires the passing P4 lock. Commit,
   push, and lock `P5_C_INPUT`. When the C packet has zero rows, the frozen
   receipt contains zero roles and no C session file.
8. `freeze-final-outcome` requires the P5 input lock. Commit, push, and lock
   `P5_FINAL_OUTCOME` for `PASS_MACHINE_RUBRIC_POOL`, `STOP_TARGET_YIELD`, or a
   terminal invalid-review route.

Every input command copies only operator-designated first-received bytes and may run once. Every
outcome command reconstructs its receipt, provenance, packets, metrics,
diagnostics, yield, and route from frozen ancestors. A nonzero validation exit
that writes a terminal stop is an outcome to commit and lock, not permission to
repair inputs or rerun the command. `verify-terminal-stop` must pass before a
terminal outcome is locked. That verifier deterministically replays the frozen
stage: P1/P2/P3/P5 must reproduce the same exception type and message hash,
while P4 must reproduce either the complete unreliable-gate bytes or the same
protocol exception. A successful replay rejects a manually fabricated stop.
`lock-head` recursively verifies every predecessor
lock and rejects a skipped stage. Historical locks remain valid only when their
commits are ancestors of the current canonical branch tip. Stage commits use
Chinese messages naming the exact boundary and contain no unrelated files.
Multi-file stages are prepared in same-directory temporary files, published
with exclusive hard links, and rolled back completely if publication fails.
Concurrent writers therefore have one winner and cannot overwrite accepted
bytes. A publication error does not create or
consume a scientific terminal route. Wrong HEAD, a dirty worktree, a missing
predecessor lock, or pre-existing stage paths are operator precondition errors;
they occur before the scientific stage opens and must not create a terminal
study route. A single terminal artifact is likewise written through a
same-directory temporary file, flushed, and exclusively hard-linked, so an
interrupted write cannot leave a truncated stop at the registered path.

## 9. Claim and Training Boundary

`PASS_MACHINE_RUBRIC_POOL` remains only a machine-rubric-qualified artifact.
It does not authorize optimizer steps, establish human validity, prove a
geometric mechanism, or show that PCSR improves behavior. Training still
requires tokenizer/loss-mask checks, leakage and template-concentration audits,
an independent human or genuinely heterogeneous-model suitability audit, and
a separately frozen method-and-training protocol.
