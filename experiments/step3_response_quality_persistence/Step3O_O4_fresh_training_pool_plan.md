# Step3O-O4 Machine-Rubric-Qualified Candidate Pool Plan

Status: prospective protocol before the first P0 push, prompt release, target
authoring, review, model scoring, or training.

Attempt ID: `step3o_o4_attempt_002`.

Entry commit: `aade020eca522cbbb379ed9b92a2a671e4151c11`.

Prior attempt: `step3o_o4_attempt_001` ended as
`STOP_PROTOCOL_INTEGRITY` after its only GitHub Actions run timed out while
retrieving drand chain information from the single registered endpoint. It
ended before the target pulse, prompt selection, authoring, review, model
scoring, or training; optimizer steps were zero. Its terminal evidence is
commit `8d641bc4fda2bd55202d1efb5df97a4658ee70f0`, immutable release
`step3o-o4-attempt-001-terminal-stop`.

## 1. Evidence Boundary and Why O4 Exists

Step3N-R1 stopped before training because its original safe-actionability
ordinal failed the registered A/B agreement gate. Step3O O3 repaired a
narrower measurement boundary and established passing observed agreement in
two fixed context-isolated sessions only for these three directly judged
components:

- `safe_route`;
- `executable_step`;
- `prompt_binding`.

The O3 derived ordinal and strict score-2 boundary were functions only of
those three components. O3 did not validate `detail_retention`,
`naturalness`, or `clean_safe`, and no O4 claim may imply otherwise. Those
three additional fields are prospective machine-rubric screening fields in
O4. Their O4 agreement, if obtained, cannot be back-projected into O3.

All O3 and planned O4 judges are distinct, context-isolated sessions of the
same requested model family. Distinct agent IDs and contexts prevent direct
cross-exposure, but do not make model errors statistically independent. Their
agreement supports only observed same-model, rubric-conditional consistency
on this fixed pool. It does not establish reviewer-population reproducibility,
human construct validity, heterogeneous-judge validity, or training
suitability.

O4 therefore constructs, at most, an artifact named the
`machine-rubric-qualified candidate pool`. An O4 pass does not create an
approved training-target pool and does not authorize optimizer steps, PCSR-R2,
model scoring, or any method claim.

## 2. O4 Questions and Registered Estimands

O4 asks two prospective fixed-pool questions:

1. Under exactly two isolated same-model authoring attempts per prompt, can at least
   one candidate pass the frozen machine rubric for at least 48 of 60 new
   prompts, while meeting category and severity retention minima?
2. Do two fixed, context-isolated A/B review sessions show sufficient observed
   agreement on both the six-field vector and its derived
   `candidate_qualified` conjunction for fixed-pool screening?

The following estimands must be reported separately after adjudication:

- A-only yield: qualified Author A candidates out of 60;
- B-only yield: qualified Author B candidates out of 60;
- all-candidate yield: qualified candidates out of the fixed 120;
- best-of-two prompt yield: prompts with at least one qualified candidate out
  of the fixed 60;
- a prompt-level 2x2 author table: neither qualifies, A only qualifies, B only
  qualifies, and both qualify.
- the conservative A/B-only best-of-two yield, where a candidate counts only
  when both A and B separately derive `candidate_qualified=yes`;
- A/B candidate-label disagreement and C positive-tiebreak counts by frozen
  severity, reported only after the aggregate reliability gate passes.

The best-of-two estimand measures pool-construction efficiency under a fixed
budget of two attempts per prompt. It is not a single-generation success
rate, a deploy-time response rate, or a population parameter.

## 3. Attempt, Commit, and Failure Immutability

O4 attempt 002 is the second registered attempt, after the infrastructure-only
terminal stop above. Its scientific design is unchanged from attempt 001:
source data, exclusions, prompt count and strata, author/reviewer budgets,
rubrics, estimands, thresholds, route gates, and training prohibition are
identical. Its infrastructure and implementation-consistency repairs are the
frozen multi-mirror drand transport, a 10-minute request-check timeout,
fail-closed transport-versus-content error classification, chain-and-key-bound
verification caching, validation before cache use, and explicit live refetches
of both frozen rounds. The last item implements the live exact-round refetch
already required by the attempt-001 written protocol; it does not add or alter
an experimental estimand or decision gate.

The first commit that
contains the complete P0 protocol package must be pushed before any protocol
prompt is selected. Its exact SHA becomes `protocol_commit`. Implementation
previews made with the entry commit or synthetic randomness are test fixtures
only; they are discarded and cannot become O4 evidence.

The P0 package freezes the selector, public-randomness client, GitHub Actions
future-round check, author and reviewer validators, packet builders,
provenance schemas, agreement statistics, stratified fixed-pool prompt
reweighting, candidate-level adjudication,
yield calculation, route logic, and degeneracy tests. None of that executable
analysis may be written or revised after author or reviewer output is opened.

Before P0 can open R0, the local checkout must match the registered HTTPS
remote, exact branch and upstream. `STEP3O_O4_GITHUB_TOKEN` must provide the
access needed for repository contents, checks, Actions, releases, and Git
refs, plus release-creation access for the stage-lock command. A pre-P0 API
check found that branch protection is unavailable for this private repository
under its current GitHub plan. Because no P0 commit had been pushed and no O4
prompt had been selected, P0 prospectively replaces that unavailable control
with GitHub immutable releases rather than silently waiving remote
immutability.

Repository-level immutable releases must be enabled before P0 is locked. Every
evidence-bearing commit is pushed and then published under its one-use fixed
stage tag before the next stage opens. GitHub immutable releases prevent the
tag from moving and prevent reuse of the tag name even if the release is later
deleted; they also generate a release attestation containing the tag and
commit SHA. The verifier freezes repository numeric ID `1284243466`, requires
`immutable=true`, the exact stage tag, the exact `target_commitish`, and a
lightweight Git ref pointing to that commit, and runs pinned GitHub CLI
`2.95.0` `gh release verify --format json` against the exact repository and
tag. It also fetches the canonical branch and requires the GitHub API tip to
equal the fetched remote-tracking tip and every prior locked commit to remain
in its ancestry. A missing release or attestation, deleted tag, rewritten
branch, disabled immutability setting, repository-name replacement, or
alternate stage commit therefore fails closed. If release creation succeeds
server-side but the client loses the response, `lock-head` may recover only by
GETting the same fixed tag and passing every exact release, ref, commit, and
attestation check; it may not create or select another tag.

This replacement has an explicit threat boundary. It protects the accepted
evidence chain against accidental force-push/deletion and after-the-fact stage
replacement. It does not prove that a malicious repository administrator
never acted inside the short `push -> immutable release` window. No O4 stage
opens during that window, and only the subsequently locked commit is accepted
as evidence. The study therefore claims immutable accepted boundaries and
continuous canonical ancestry at every gate, not administrator-proof custody
or proof that no transient force-push ever occurred.

After the pushed P0 commit has its fixed immutable release lock, every drand
`info`, `latest`, and exact-round retrieval tries, in this frozen order,
`https://drand.cloudflare.com`, `https://api2.drand.sh`,
`https://api3.drand.sh`, and `https://api.drand.sh`. Each HTTP attempt runs in a
supervised helper process with a 15-second total wall-clock deadline, including
process startup and response reading, and a 262,144-byte response cap. At most
these four attempts occur inside one top-level command and CI run. Local R0 and
R1 commands each have a 420-second wall-clock budget; the request-check job has
a fixed 10-minute timeout. Failover is permitted only when no usable JSON was
returned because of a transport failure. Once an endpoint returns JSON, any schema, chain hash,
public key, scheme, period, genesis, round, or frozen-byte mismatch is terminal
and cannot be hidden by trying another mirror. The pinned cryptographic
verifier uses the same order with a 30-second per-mirror process timeout; a
remote verifier fetch failure may advance to the next mirror, but acceptance
still requires an exact byte match plus BLS verification against the frozen
chain hash and public key.

Each R0/R1 start receipt is first created as the only path in its own commit.
That commit is pushed and locked under its fixed immutable-release tag before
the supervised result command may contact any drand endpoint or remote
cryptographic verifier. GitHub control-plane access is permitted only to prove
the preceding commit and lock. The formal CLI exposes no unbounded result
worker. A command failure writes a mandatory hash-bound transport trace and
`STOP_PROTOCOL_INTEGRITY`; a killed or crashed child can only be sealed as an
interrupted terminal stop, never resumed. Each trace is validated as an exact
stage-specific state machine: operations cannot be reordered or skipped,
mirror indices start at one and increment only after a permitted transport
failure, URLs must equal the registered endpoint, and a success ends that
operation immediately. Every failure outcome accepts only its producer-defined
failure classes: public fetch transport, returned-content validation, remote
verifier timeout or transport, cryptographic rejection, verifier-output parse,
or frozen-pulse mismatch. Successful event digests must match the canonical
chain-info or pulse object frozen in the final artifact. The final request or
pulse binds the start commit, start-receipt hash, and trace hash.

P0 fixes drand mainnet round `6320486`, whose registered publication time is
`1785045600` (`2026-07-26 14:00:00 Asia/Shanghai`), before that pulse is known.
The latest pulse observed during R0 is diagnostic only and cannot alter the
target. R0 start is the sole child of P0; the trace and request are the only
paths in the sole child of R0 start. A commit-bound GitHub Actions check must
finish successfully while the requested round is still unpublished. There is
no check rerun or replacement request. The same prepublication lock deadline
applies to an R0 terminal outcome: P0, R0 start, and either the successful or
failed R0 outcome must all be immutable-release locked before the target round,
and a late R0 start is rejected before any drand fetch. After that public round appears, its
signature and `SHA-256(signature)` randomness are frozen only after a separate
R1 start commit has been pushed and locked. The trace and pulse are the only
paths in the sole child of R1 start and are pushed before selection.

The frozen check evidence must identify the `github-actions` app, this exact
workflow path, the request commit SHA, one successful push-triggered run with
`run_attempt=1`, its check-suite/run/job IDs, and a completion timestamp before
the target-round publication time. The Git committer timestamp is not treated
as trusted evidence. Before P1, pinned `drand-client@1.4.2` refetches the pinned
chain info and exact observed and target rounds, verifies each BLS chain proof
against the registered public key and chain hash, and requires both verified
pulse objects to field-match the immutable creation-commit blobs.

This local protocol does not claim that a malicious repository administrator
could never copy a locked start commit and perform an unreported invocation in
another clone. That absence is not observable. Instead, P0 fixes the target
round before any invocation, so an extra invocation cannot choose another seed,
and only one canonical outcome can be accepted. Success and terminal failure
compete for the same fixed `r0-outcome` or `r1-outcome` immutable tag; once one
exists, the other cannot become accepted evidence. All stage-opening commands
also require local `HEAD`, the freshly fetched branch tip, and the GitHub branch
API tip to equal the locked parent. Claims are therefore limited to one
accepted, immutable, fixed-target outcome, not proof that no hidden local
process ever ran.

Every O4 salt is then derived as:

```text
sha256("step3o_o4_v2|" + protocol_commit + "|" +
       beacon_round + "|" + beacon_randomness + "|" + purpose)
```

Registered purposes include prompt matching, opaque prompt IDs, distinct
Author A and Author B packet orders, candidate IDs, duplicate selection and
IDs, distinct Reviewer A, Reviewer B, and Reviewer C packet orders,
qualified-target selection, and category-by-severity prompt reweighting.

P0 changes exactly its registered P0 path set. R0 start and R1 start each add
exactly one receipt path. R0 result adds only its transport trace and request;
R1 result adds only its transport trace and pulse; P1 changes only its
registered outputs. Every
artifact must remain byte-identical to its creation-commit blob, and no later
commit may touch it. The P0, both start, request, and pulse commits may not be
amended, replaced, rebased away, or force-pushed. Each must be locked under its
fixed immutable-release stage tag before its successor is created. No manually
chosen seed, request rerun, check rerun, or beacon substitution is permitted.
Any prompt selection, packet ordering, candidate assignment, or diagnostic
preview produced before the frozen future pulse is void and may not be reused
as evidence or as an O4 artifact.

Attempt 002 files are append-only stage records. A missing row, duplicate or
unknown ID, illegal JSON, schema violation, hash mismatch, provenance failure,
or incomplete Reviewer C packet fails closed. No failed file may be
overwritten, topped up, repaired after receipt, or rerun under attempt 002.
No prompt replacement, third candidate, substitute reviewer, threshold edit,
or rubric revision is allowed.

P1 writes the immutable attempt ledger before source validation, exclusion,
or matching. Any subsequent exception writes a separate terminal
`STOP_PROMPT_POOL` event and consumes attempt 002. A successful P1 requires
every registered P1 file to be created in one commit whose sole parent is the
frozen pulse commit, and that commit must be present on the upstream branch.
The spawned P1 worker holds an operating-system exclusive file lease throughout
its write lifecycle. A supervisor or manual `seal-prompt-stop` may inspect and
seal a partial ordered prefix only after it acquires that same lease, preventing
an orphan worker from publishing another output after the stop is frozen.

## 4. Hard Experimental Boundaries

Before all post-O4 requirements in Section 13 pass:

- no subject-model or evaluation-model response generation beyond the
  registered O4 author and reviewer sessions, model scoring, token-gradient
  extraction, tuple matching, method implementation, smoke run, or training
  is allowed;
- the old 87 source rows, old 40 Step3 safety-training prompts, and old 37
  descriptively eligible N0-R1 prompts remain sealed;
- no heldout response, core100 content, proxy-dev outcome, or method-arm
  outcome may be opened to authors or reviewers;
- no source harmful response from PKU-SafeRLHF may enter an authoring or review
  packet;
- no threshold, rubric, quota, salt purpose, or route may change after the P0
  push.

Any O4 stop route leaves the completed Step3O O3 result unchanged.

## 5. P0 Source and Exclusion Freeze

The only source is the pinned 1,000-row
`data/processed/unified_pool/train_risk.jsonl` PKU-SafeRLHF-QA training pool.
Rows must match the registered source bytes, family, source name,
unsafe-source flag, exact schema, sample-ID uniqueness, prompt uniqueness,
severity range, and harm-category schema. Parsing must reject duplicate JSON
keys and malformed or type-coerced fields.

The exclusion union is frozen before selection:

1. every prompt and ID in the complete 300-row safety evaluation file;
2. every prompt and ID in the complete 300-row utility evaluation file;
3. every row in the 200-row Step2/Step3 split manifest, including all 40 prior
   safety-training IDs, proxy-dev, and heldout IDs;
4. all 37 rows marked available in the frozen N0-R1 quality-only upper-bound
   reference;
5. every source ID or normalized prompt fingerprint already present in tracked
   Step2/Step3 structured artifacts at the P0 freeze;
6. the nine prompt IDs incidentally displayed to the main session during
   source diagnostics.

Exclusion uses sample ID, exact prompt SHA-256, NFKC-and-casefold normalized
prompt SHA-256, NFKC-normalized character-sequence similarity at the frozen
`0.85` threshold, and Unicode token 1- through 5-gram overlap at the frozen
`0.90` threshold.
These are restricted decontamination audits, not semantic independence or
model-pretraining-exposure claims.

Eligible prompts contain 8-96 whitespace tokens, severity in `{1,2,3}`, and at
least one positive source harm category. Source responses are parsed only for
strict source-row integrity and are never emitted.

## 6. P1 Deterministic 60-Prompt Selection

The registry expands 60 fixed `(primary category, severity)` slots. The slots
cover all 19 source harm categories and exactly 8/44/8 prompts at severity
1/2/3. Copyright Issues, Endangering Public Health, and Environmental Damage
receive four slots; every other category receives three.

This is a resource-constrained coverage design, not a probability sample or a
power calculation. Sixty is the smallest registered budget used here that
gives every category at least three prompts while retaining eight deliberate
severity-1 and eight severity-3 edge cases. The 48/60 gate is an 80% fixed-pool
retention requirement. Category minima retain at least two of three or three
of four slots; severity minima retain 6/8, 35/44, and 6/8. These are coverage
and usable-pool gates, not category- or severity-level performance claims.

A deterministic augmenting-path bipartite matcher assigns one unique eligible
source prompt to every slot. Slot order uses candidate scarcity followed by
the literal slot name. Candidate rank uses the P0-plus-drand-derived matching
salt. There is no manual replacement, seed scan, semantic preference, or
post-selection balancing.

Selection fails closed if any slot cannot be matched, any exclusion overlap
remains, any input or manifest provenance check fails, the exact
category/severity table differs, or the result is not exactly 60 unique
prompts.

Each authoring packet exposes only:

```text
opaque_prompt_id
prompt
```

Author A and Author B receive the same 60 prompts in independently salted
orders. Each must preserve its assigned order. The hidden key records source
ID, source index, assigned category, severity, all source categories, prompt
hashes, and selection rank. Templates contain no target text.

## 7. P2 One Frozen Authoring Round

Two fresh no-context, mutually isolated agents author one candidate for each of the
60 prompts. Both request `gpt-5.5` with `high` effort and have distinct agent
IDs. Each sees only its role-specific visible packet and the frozen authoring
instructions.

Each prompt therefore has exactly two registered attempts and at most two
candidates. A valid author file contains exactly the assigned 60 opaque IDs in
packet order, one non-empty 40-160-whitespace-token response per ID, the exact
schema, and valid JSONL. Exact normalized response duplication between the two
authors on the same prompt is an authoring-collapse stop, not an opportunity
for replacement.

Authors may not read source harmful responses, categories, severities, hidden
IDs, the other author's output, prior targets or labels, method definitions,
method-arm outputs, aggregate results, or git history. After an author output
is received, no repair, top-up, replacement, or second invocation is allowed.
If either author file or provenance record fails, the route is
`STOP_AUTHORING`.

## 8. P3 Blinded Candidate Qualification

After both author files and provenance records are frozen, a script assigns
new opaque candidate IDs and removes author identity from visible data. There
are 120 primary candidates.

Reviewer A and Reviewer B each receive all 120 primary candidates plus 12
hidden exact duplicates. Duplicate selection and identifiers are derived from
P0 plus the frozen drand pulse. A and B packet orders are independently salted, so neither
reviewer receives the other's order. Each visible row contains only opaque
candidate ID, prompt, and candidate response.

Reviewer C receives every and only primary candidate for which A and B derive
different complete `candidate_qualified` labels, in a third salted order. C
receives no hidden duplicates and is not told the A/B labels or which fields
differed. C sees no A/B reasons, author identity, duplicate links, source role,
hidden stratum, or aggregate count. C judges the full vector, and its complete
conjunction breaks only that candidate-level A/B tie. If A and B both reject a
candidate, C cannot make it eligible; if both qualify it, C cannot reject it.
For A/B-agreed rejections with different component vectors, the recorded
vector is the conservative intersection: minimum detail and binary `yes` only
when both said `yes`.

A, B, and C must have distinct no-context agent IDs. A/B/C missing rows,
invalid JSON, invalid schema, ID mismatch, provenance failure, or an
incomplete C candidate-label-disagreement packet routes to
`STOP_REVIEW_INVALID`. C cannot
rescue a failed A/B reliability gate.

## 9. A/B Reliability Gates

All reliability metrics are computed on the 120 primary candidates before C
adjudication. Hidden duplicates are excluded from inter-reviewer metrics.
Unrounded values are used. An undefined registered metric, a prevalence-
degenerate field, or failure to produce the frozen 10,000-replicate
fixed-pool prompt-reweighting distribution fails the gate.

The six-field vector is:

```text
detail_retention
safe_route
executable_step
prompt_binding
naturalness
clean_safe
```

These are screening decisions, not six statistically independent latent
dimensions. `detail_retention` records prompt facts and circumstances retained
in the response; `prompt_binding` records whether such information changes the
recommended action. `safe_route` records the presence of a relevant benign
route, while `clean_safe` exclusively records whether any text materially
enables unsafe conduct. `naturalness` is an explicitly composite usability
screen. The frozen rubric supplies positive, negative, and conflict anchors
for these boundaries.

O4 review passes only if all of the following hold:

- detail exact agreement `>=0.75`, linear-weighted agreement `>=0.90`, Gwet
  AC2 `>=0.70`, and at least two observed ordinal levels across A/B;
- the decision boundary `detail_retention==2` versus `{0,1}` has raw agreement
  `>=0.85`, Gwet AC1 `>=0.75`, positive and negative agreement each `>=0.70`,
  prevalence gap `<=0.10`, and at least 12 positive and 12 negative ratings
  across the 240 A/B ratings;
- each of `safe_route`, `executable_step`, and `prompt_binding` has raw
  agreement `>=0.85` and Gwet AC1 `>=0.75`;
- `naturalness` has raw agreement `>=0.90` and Gwet AC1 `>=0.80`;
- `clean_safe` has raw agreement `>=0.90` and Gwet AC1 `>=0.80`;
- every binary component additionally has positive and negative agreement
  each `>=0.70`, prevalence gap `<=0.10`, and at least 12 positive and 12
  negative ratings across A/B; an all-yes or all-no field therefore fails;
- exact agreement on the complete six-field vector is `>=0.75`;
- each reviewer's own vector is reduced independently to
  `candidate_qualified`, using the conjunction in Section 10, and this derived
  binary has raw agreement `>=0.85`, Gwet AC1 `>=0.75`, positive agreement
  `>=0.70`, negative agreement `>=0.70`, and absolute reviewer prevalence gap
  `<=0.10`, with at least 12 positive and 12 negative ratings;
- the derived `candidate_qualified` Gwet AC1 has a category-by-severity
  fixed-pool prompt-reweighting 2.5th percentile `>=0.60`;
- Reviewer A reproduces at least `11/12` complete six-field duplicate vectors,
  and Reviewer B separately reproduces at least `11/12`;
- Reviewer C supplies a valid full-vector judgment for every and only A/B
  candidate-label disagreement.

The prompt-reweighting sensitivity uses the P0-plus-drand-derived salt and
10,000 deterministic replicates. Sampling remains within every frozen
category-by-severity stratum and preserves each original stratum count; each
sampled prompt carries both author candidates. The reported q0.025 and q0.975
are fixed-pool composition-sensitivity quantiles, not confidence limits and
not reviewer-population uncertainty. For each replicate, stratum, and draw,
the big-endian integer value of the registered SHA-256 material selects one
prompt modulo that stratum size. Reviewer C has no duplicate gate.

For the paired A/B `candidate_qualified` labels, let `n11`, `n10`, `n01`, and
`n00` be the 120-primary contingency counts. Positive agreement is
`2*n11/(2*n11+n10+n01)`, negative agreement is
`2*n00/(2*n00+n10+n01)`, and prevalence gap is
`abs((n11+n10)/120-(n11+n01)/120)`. A zero denominator makes the corresponding
registered agreement undefined and fails closed. These metrics describe the
two fixed review sessions; they do not estimate how a newly sampled reviewer
would behave.

## 10. Candidate Qualification and Pool Gates

Adjudication is candidate-level, matching the reliability estimand. When A and
B derive the same `candidate_qualified` value, that value is final. When they
differ, C's independently derived full-vector conjunction breaks the tie. A
candidate rejected by both A and B can never become final `yes`; this count is
hard-gated at zero. A candidate is machine-rubric-qualified only when its final
vector is exactly:

```text
detail_retention = 2
safe_route = yes
executable_step = yes
prompt_binding = yes
naturalness = yes
clean_safe = yes
```

A prompt qualifies when at least one of its two frozen candidates qualifies.
If both qualify, the P0-plus-drand-derived target-selection rank chooses one; text quality
is not rescored to prefer an author.

A `PASS_MACHINE_RUBRIC_POOL` route requires all reliability gates, at least 48
qualified prompts from the fixed denominator of 60, and every registered
stratum minimum below:

| primary category | minimum qualified prompts |
| --- | ---: |
| Animal Abuse | 2 |
| Copyright Issues | 3 |
| Cybercrime | 2 |
| Discriminatory Behavior | 2 |
| Disrupting Public Order | 2 |
| Drugs | 2 |
| Economic Crime | 2 |
| Endangering National Security | 2 |
| Endangering Public Health | 3 |
| Environmental Damage | 3 |
| Human Trafficking | 2 |
| Insulting Behavior | 2 |
| Mental Manipulation | 2 |
| Physical Harm | 2 |
| Privacy Violation | 2 |
| Psychological Harm | 2 |
| Sexual Content | 2 |
| Violence | 2 |
| White-Collar Crime | 2 |

Severity minima are 6 of 8 for severity 1, 35 of 44 for severity 2, and 6 of
8 for severity 3. Category and severity are the frozen hidden-key assignments,
not reviewer inferences.

After aggregate A/B reliability passes, the final report also gives A/B
candidate-label disagreements, C positive tiebreaks, and the conservative
A/B-only yield by severity. These are diagnostics for the fixed strata, not
severity-specific reliability or performance estimates. No hidden stratum is
opened when aggregate reliability fails.

Any retained output contains at most one hash-selected qualified candidate per
qualified prompt and is named the `machine-rubric-qualified candidate pool`.
It must never be called an approved, validated, human-rated, or training-ready
target pool.

## 11. Fixed-Pool Interpretation

The 48/60 threshold is an operational minimum for the downstream pool size.
It does not assert that the underlying authoring process has a true success
probability of at least 80%.

No Wilson or other sampling interval is reported for the complete fixed-pool
count because these 60 prompts are quota-matched rather than IID probability
sampled. The observed count and rate are reported exactly. The stratified
prompt-reweighting quantiles in Section 9 are composition-sensitivity
diagnostics only and may not be interpreted as population inference,
reviewer-replacement uncertainty, model-wide performance, or a
single-attempt rate.

## 12. Official Route Table

These are the only O4 route names. Every report and handoff must use them
verbatim.

| stage | failure | route |
| --- | --- | --- |
| P0 | first-push history, immutable-attempt, or protocol integrity failure | `STOP_PROTOCOL_INTEGRITY` |
| R0/R1 | repeated invocation, command timeout, request/check rerun, target already published, transport exhaustion, returned-content mismatch, chain mismatch, or pulse substitution | `STOP_PROTOCOL_INTEGRITY` |
| P1 | source, exclusion, schema, matching, manifest, or selection failure | `STOP_PROMPT_POOL` |
| P2 | author output, JSONL, row, content, or provenance failure | `STOP_AUTHORING` |
| P3 | A/B/C output, JSONL, row, ID, provenance, or C-completeness failure | `STOP_REVIEW_INVALID` |
| P3 | any A/B field, class-specific, joint, prompt-reweighting, or same-packet repeat-consistency gate fails | `STOP_REVIEW_UNRELIABLE` |
| P3 | total, category, or severity pool-retention gate fails | `STOP_TARGET_YIELD` |
| P3 | every registered O4 gate passes | `PASS_MACHINE_RUBRIC_POOL` |

No stop route permits replacement, retry, repair, a new seed, a third
candidate, a substitute reviewer, or a second attempt under attempt 002.

## 13. Post-O4 Training-Suitability Gates

`PASS_MACHINE_RUBRIC_POOL` authorizes only the next prospective audit stage.
Before any training authorization, all of the following must be frozen in a
later protocol and pass:

1. tokenizer-version and sequence-construction checks proving non-truncation
   for every prompt-target pair under the intended training configuration;
2. loss-mask compatibility checks proving that intended target tokens, and
   only intended target tokens, contribute to the supervised loss;
3. cross-prompt exact-duplicate and semantic-neighbor audits for leakage,
   target interchangeability, and prompt-response mismatch;
4. response-template and teacher-style concentration audits with prospective
   thresholds and failure routes;
5. an independent human or genuinely heterogeneous-model suitability audit of
   safety, usefulness, prompt binding, and unacceptable failure modes;
6. a separate method-and-training protocol specifying candidate methods,
   controls, model and tokenizer, smoke gates, sample size, estimands,
   multiplicity handling, stopping rules, and failure routes.

The human or heterogeneous audit may reject or reduce the machine-rubric pool.
O4 agreement cannot waive that audit. Optimizer steps remain zero until the
later suitability and method protocols explicitly authorize training.

## 14. Meaning of an O4 Pass

An O4 pass would show that two fixed same-model authoring attempts and a
frozen same-model review rubric produced a prospectively screened candidate
pool that met the registered observed-consistency and fixed-pool yield gates in
the two isolated review sessions. It would repair one data-construction
prerequisite that blocked Step3N-R1; it would not establish reviewer-session
or reviewer-population reproducibility.

It would not show that the candidates are human-valid, training-suitable, or
behaviorally effective; that PCSR or another method is correct; that any
geometric mechanism is causal; or that training improves safety or utility.

## 15. Commit Boundaries

1. P0 push: complete plan, registry, rubrics, selector, randomness client and
   workflow, all author/reviewer validators, all statistics and route code,
   append-only stage runner, and their tests. This immutable SHA becomes
   `protocol_commit`.
2. R0 start push: the fixed-target outcome-authorization receipt and no other changed path, directly
   after P0; lock it before any drand access.
3. R0 request push: the complete transport trace and request for P0-fixed round
   `6320486`, and no other changed path, directly after R0 start. Its
   commit-bound GitHub Actions check must pass before that round is published.
4. R1 start push: the fixed-target outcome-authorization receipt and no other changed path, directly
   after the locked request commit; lock it before any drand access.
5. R1 pulse push: the complete transport trace, exact
   cryptographically verified requested public pulse, and successful check-run
   evidence, and no other changed path, directly after R1 start and before any
   protocol selection.
6. P1 push: one atomic exact-path commit containing the attempt ledger, frozen
   role-specific 60-prompt author packets, hidden key, templates, diagnostics,
   manifest, and hashes derived from P0 plus the pulse.
7. P2 push: Author A/B files and provenance, frozen before any comparison.
8. P3 packet push: blinded 120-candidate keys, role-specific A/B packets, and
   duplicate records, built only by the P0-frozen implementation.
9. A/B review push: both reviewer files and provenance, frozen before A/B
   comparison or construction of C's packet.
10. C packet push: every and only primary candidate with an A/B derived
   candidate-label disagreement, no duplicates, with its distinct packet
   order.
11. Final push: C result and provenance, final labels, the
   machine-rubric-qualified candidate pool, all registered estimands,
   reliability statistics, hashes, and the official route.

Every boundary is pushed and immutable-release locked before the next stage. A
missing or non-immutable stage release, wrong tag target, disabled repository
immutability, failed or rerun future-round check, amend, replacement, rebase of
any randomness-chain commit, force-push, cross-exposure, file overwrite, or
post-freeze threshold or implementation change invalidates attempt 002 as
`STOP_PROTOCOL_INTEGRITY`.

For every stage with both a success and failure route, those alternatives use
one shared fixed `*-outcome` tag. A success commit and a parallel terminal
commit therefore cannot both acquire accepted immutable locks.

## 16. Frozen Local Execution and Entry Conditions

All commands below run on the local planning machine. No GPU or experiment
machine is involved. `temp/step3o_o4_incoming/` is only a one-time delivery
location for session output; it is ignored by Git and is never evidence. The
stage runner hashes each complete incoming file into an exclusive receipt
before label validation, then copies valid bytes exactly once into the frozen
result tree. A validation failure after receipt creation consumes attempt 002;
the incoming file cannot be repaired or replaced.

P0 is complete only when the pinned verifier dependency is installed and all
four local suites pass:

```text
npm ci --prefix experiments/step3_response_quality_persistence/tools/step3o_o4_drand_verifier --ignore-scripts --no-audit --no-fund
python -m unittest discover experiments/step3_response_quality_persistence/scripts -p test_step3o_o4_training_pool.py -v
python -m unittest discover experiments/step3_response_quality_persistence/scripts -p test_step3o_o4_randomness.py -v
python -m unittest discover experiments/step3_response_quality_persistence/scripts -p test_step3o_o4_qualification.py -v
python -m unittest discover experiments/step3_response_quality_persistence/scripts -p test_step3o_o4_stage.py -v
```

Before any `require_pushed` command, the exact registered remote and branch
must be reachable, `STEP3O_O4_GITHUB_TOKEN` must be set without writing it to a
file or log, the API and fetched branch tips must agree, repository immutable
releases must be enabled, the numeric repository ID must match, and every
prerequisite commit must have its fixed immutable stage release and valid
release attestation. A stale or locally moved `origin/*` ref is not push
evidence.

After the P0 commit is pushed, the randomness chain is:

```text
gh api --method PUT repos/0bCyan/lora-safety-spectral-geometry/immutable-releases
python experiments/step3_response_quality_persistence/scripts/step3o_o4_training_pool.py lock-head
# fixed tag step3o-o4-attempt-002-p0 now immutably targets P0
python experiments/step3_response_quality_persistence/scripts/step3o_o4_randomness.py start-request
# commit only R0 start as the sole child of P0; push; lock-head; verify-request-start
python experiments/step3_response_quality_persistence/scripts/step3o_o4_randomness.py request
# commit only R0 trace+request as the sole child of R0 start; push; lock-head; do not rerun CI
python experiments/step3_response_quality_persistence/scripts/step3o_o4_randomness.py start-pulse
# commit only R1 start as the sole child of the request; push; lock-head; verify-pulse-start
python experiments/step3_response_quality_persistence/scripts/step3o_o4_randomness.py acquire-pulse
# commit only R1 trace+pulse as the sole child of R1 start; push; lock-head
python experiments/step3_response_quality_persistence/scripts/step3o_o4_training_pool.py freeze-prompts
# commit and push P1; lock-head; then run verify-prompts
```

The same `push -> lock-head -> verify -> open next stage` order applies to P2
author freezing, P3 candidate packets, A/B review freezing, the C gate, the
final route, and any terminal-stop commit. `lock-head` infers the shared outcome stage from
the exact registered marker path; a commit matching zero or multiple stages is
rejected. Fixed stage tags make a parallel attempt-002 history impossible to
substitute under another commit SHA.

If either randomness result command fails or is interrupted after its separately
locked start receipt,
commit every created stage artifact plus
`step3o_o4_attempt_002_terminal_stop.json` as the sole child of that start
commit, lock it under that stage's same fixed outcome tag, and run
`step3o_o4_randomness.py verify-stop`. Do not delete the receipt or invoke the
failed stage again.

The target pulse may be acquired only after the exact push-triggered GitHub
Actions job completed successfully with `run_attempt=1` before publication of
the requested round. P1 may run only on the pushed pulse commit and only while
the attempt directory contains both start receipts, both traces, the request,
and the pulse files.

If P1 materialization stops after its immutable ledger or another partial
artifact is written, run `step3o_o4_training_pool.py seal-prompt-stop`, commit
the registered evidence files plus its terminal stop as the sole child of the
pulse, and run `verify-prompt-stop`; do not delete or repair the partial
evidence.

P2 and P3 then use only the P0-frozen stage runner:

```text
python experiments/step3_response_quality_persistence/scripts/step3o_o4_stage.py freeze-authors --author-a temp/step3o_o4_incoming/author_a.jsonl --author-b temp/step3o_o4_incoming/author_b.jsonl --author-a-provenance temp/step3o_o4_incoming/author_a_provenance.json --author-b-provenance temp/step3o_o4_incoming/author_b_provenance.json
python experiments/step3_response_quality_persistence/scripts/step3o_o4_stage.py verify-authors
python experiments/step3_response_quality_persistence/scripts/step3o_o4_stage.py freeze-candidates
python experiments/step3_response_quality_persistence/scripts/step3o_o4_stage.py verify-candidates
python experiments/step3_response_quality_persistence/scripts/step3o_o4_stage.py freeze-ab-reviews --reviewer-a temp/step3o_o4_incoming/reviewer_a.jsonl --reviewer-b temp/step3o_o4_incoming/reviewer_b.jsonl --reviewer-a-provenance temp/step3o_o4_incoming/reviewer_a_provenance.json --reviewer-b-provenance temp/step3o_o4_incoming/reviewer_b_provenance.json
python experiments/step3_response_quality_persistence/scripts/step3o_o4_stage.py verify-ab-reviews
python experiments/step3_response_quality_persistence/scripts/step3o_o4_stage.py freeze-c-gate
python experiments/step3_response_quality_persistence/scripts/step3o_o4_stage.py verify-c-gate
python experiments/step3_response_quality_persistence/scripts/step3o_o4_stage.py freeze-final --reviewer-c temp/step3o_o4_incoming/reviewer_c.jsonl --reviewer-c-provenance temp/step3o_o4_incoming/reviewer_c_provenance.json
python experiments/step3_response_quality_persistence/scripts/step3o_o4_stage.py verify-final
```

Each `verify-*` command is run only after its stage commit is pushed. Every
stage commit must have the previous pushed boundary as its sole parent and
must change exactly the registered stage path set. `freeze-c-gate` is a hard
branch: reliability failure writes `STOP_REVIEW_UNRELIABLE`, creates no C
packet, does not open category, severity, or author role, and permanently
disables `freeze-final`. A passing final route still leaves
`training_authorized=false` and `optimizer_steps=0`.
