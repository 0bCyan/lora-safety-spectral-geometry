# Step3P Independent Reviewer Handoff

Route: `step3p_public_paired_pool_attempt_001`.

No reviewer session may start before the P0 protocol commit is pushed and the
P1 candidate packet is generated, committed, and pushed from that exact P0
commit. The main research session is the operator and is not an eligible
reviewer.

## 1. Reviewer A and Reviewer B

Use two different fresh no-context sessions. They may use the same registered
model family, but their session IDs must differ and neither may inherit the
main-session conversation.

Each session receives only:

```text
review_rubrics/step3p_public_target_qualification_rubric.md
results/step3p_public_paired_pool_attempt_001/packets/reviewer_<a|b>_packet.jsonl
```

Do not provide the plan, registry, hidden candidates, duplicate links, source
dataset name, category or severity metadata, the other packet, prior O4/O4R
outputs, method definitions, aggregate results, or git history.

The reviewer returns exactly one UTF-8 LF JSONL file in packet order, using the
schema in the rubric. Missing rows, extra rows, reordered IDs, invalid JSON,
invalid labels, or an empty reason are terminal `STOP_REVIEW_INVALID`; do not
repair, top up, or replace a reviewer inside attempt 001.

Complete the corresponding provenance template with:

- the fresh reviewer session ID;
- the exact model identifier;
- the review-file SHA-256;
- the frozen packet and rubric hashes already present in the template;
- `context_isolated=true` and `prohibited_access=false` only when true.

The operator then runs:

```text
python experiments/step3_response_quality_persistence/scripts/step3p_public_paired_pool.py analyze-ab --review-a <reviewer_a.jsonl> --provenance-a <reviewer_a_provenance.json> --review-b <reviewer_b.jsonl> --provenance-b <reviewer_b_provenance.json>
```

This command freezes the received A/B bytes and either records
`STOP_REVIEW_UNRELIABLE` or creates the exact Reviewer C packet. It must not be
rerun with replacement inputs.

## 2. Reviewer C

Reviewer C is a third fresh no-context session with a session ID distinct from
A and B. It may start only when `analyze-ab` records an A/B reliability pass.

It receives only:

```text
review_rubrics/step3p_public_target_qualification_rubric.md
results/step3p_public_paired_pool_attempt_001/reviews/ab/reviewer_c_packet.jsonl
```

The C packet contains every and only primary candidate whose A/B derived
qualification labels disagree. C sees no A/B component labels, reasons,
marginal counts, source metadata, category, or severity. C returns the same
review schema and a completed C provenance record. If the frozen packet is
empty, no C session is opened.

The operator finalizes with the exact A/B files already frozen by `analyze-ab`:

```text
python experiments/step3_response_quality_persistence/scripts/step3p_public_paired_pool.py finalize --review-a <frozen_reviewer_a.jsonl> --provenance-a <frozen_a_provenance.json> --review-b <frozen_reviewer_b.jsonl> --provenance-b <frozen_b_provenance.json> --review-c <reviewer_c.jsonl> --provenance-c <reviewer_c_provenance.json>
```

Omit the final two arguments only when the frozen C packet has zero rows.

## 3. Interpretation Boundary

`PASS_PUBLIC_MACHINE_RUBRIC_POOL` does not authorize training. It only opens a
separate tokenizer, loss-mask, leakage, template-concentration, heterogeneous
suitability, and method-training protocol. Every stop route retains
`training_authorized=false` and `optimizer_steps=0`.

Reviewer isolation and prohibited-access fields are attestations within the
declared workflow, not cryptographic proof of unseen context.
