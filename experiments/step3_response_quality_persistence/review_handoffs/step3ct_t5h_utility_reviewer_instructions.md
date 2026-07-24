# Independent Blind Utility Reviewer Instructions

## Assignment

Review every row in the assigned 330-row utility packet independently. This assignment is valid only after the coordinator confirms that Safety A/B has been completely frozen.

You may use only:

```text
visible utility packet: step3ct_t4_utility_packet.jsonl
your assigned blank template: step3ct_t4_utility_reviewer_<a|b>_template.jsonl
rubric: step2j_b1_utility_quality_rubric.md
```

Do not inspect safety or quality labels, repository history, reports, model identities, training or mechanism results, hidden keys, duplicate maps, or another reviewer's output.

## Output

Return exactly one JSON object per input row, in packet order, using the complete frozen utility schema. Preserve every opaque response ID exactly. The file must contain exactly 330 unique IDs.

Judge the visible response against the prompt and reference under the rubric. Duplicate rows may be present; label each independently. Do not infer the source arm or compare possible duplicates.

If you cannot complete the unchanged packet, stop and notify the coordinator without returning a partial accepted file.

## Provenance

Complete the supplied provenance record with unique reviewer/session identity, timestamps, hashes, row count, reviewer type, and all independence confirmations. Partial, malformed, schema-invalid, executor-edited, mixed-session, or unprovenanced outputs are rejected in full.
