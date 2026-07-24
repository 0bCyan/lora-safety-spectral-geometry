# Step3C-R P0 Protocol Amendment Report

Status: completed prospectively after `b16f5a1` and before any C-R0 label.

## Reason

The original C-R0 implementation reported preference reversal consistency but did not make response-side specificity reversal consistency an independent gate. Its legacy artifact verifier also classified only one EOL direction. P0 corrects both implementation gaps without changing the scientific packet.

## Frozen Corrections

- Reviewer A and B each have separate `>=7/8` preference reversal gates.
- Reviewer A and B each have separate `>=7/8` specificity-vector reversal gates.
- Naturalness reversal consistency is reported descriptively and is not a promotion gate.
- Artifact verification classifies raw matches, `CRLF -> LF`, `LF -> CRLF`, and non-EOL mismatches symmetrically.
- New protocol outputs are required to use canonical LF bytes.

## Packet Preservation

| artifact | old SHA-256 | new SHA-256 | byte equal |
| --- | --- | --- | --- |
| `experiments/step3_response_quality_persistence/results/step3cr_cr0/packets/step3cr_cr0_pairwise_packet.jsonl` | `c7332143c664dbe5ab84e9b78bbce5babe40940ffb2345e51d78cd718de3b8d7` | `c7332143c664dbe5ab84e9b78bbce5babe40940ffb2345e51d78cd718de3b8d7` | True |
| `experiments/step3_response_quality_persistence/results/step3cr_cr0/packets/step3cr_cr0_pairwise_hidden_key.csv` | `5233eba7efa64e32892dd971207970f50532f67f20a5dd2ae7d647c32d14ff23` | `5233eba7efa64e32892dd971207970f50532f67f20a5dd2ae7d647c32d14ff23` | True |
| `experiments/step3_response_quality_persistence/results/step3cr_cr0/packets/step3cr_cr0_reversed_duplicate_key.csv` | `7f3ee3ed0bc5aef465361950fa9805464d28b1a31655cd6c411293f87ef3e41d` | `7f3ee3ed0bc5aef465361950fa9805464d28b1a31655cd6c411293f87ef3e41d` | True |
| `experiments/step3_response_quality_persistence/results/step3cr_cr0/packets/step3cr_cr0_reviewer_a_template.jsonl` | `961cf49f428d625d3b05b54cf3008113089bc02e8018821c9acfb8659a63ca9d` | `961cf49f428d625d3b05b54cf3008113089bc02e8018821c9acfb8659a63ca9d` | True |
| `experiments/step3_response_quality_persistence/results/step3cr_cr0/packets/step3cr_cr0_reviewer_b_template.jsonl` | `961cf49f428d625d3b05b54cf3008113089bc02e8018821c9acfb8659a63ca9d` | `961cf49f428d625d3b05b54cf3008113089bc02e8018821c9acfb8659a63ca9d` | True |
| `experiments/step3_response_quality_persistence/results/step3cr_cr0/packets/step3cr_cr0_reviewer_c_packet.jsonl` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` | True |

Canonical pairs remain 40; reversed duplicates remain 8; randomization seed remains 20260723. Response text, opaque IDs, presentation order, hidden pair mapping, and duplicate links are byte-identical to `b16f5a1`.

## Integrity

- Legacy Stage0 artifacts: 47
- Raw hash matches: 47
- EOL-only classifications: 0
- Non-EOL mismatches: 0
- Synthetic tests passed: true
- Reviewer labels present: 0
- Optimizer steps: 0
- Heldout opened: false

## Commands

```text
python experiments/step3_response_quality_persistence/scripts/prepare_step3cr_cr0_protocol.py test-fixtures
python experiments/step3_response_quality_persistence/scripts/prepare_step3cr_cr0_protocol.py freeze-protocol
python experiments/step3_response_quality_persistence/scripts/prepare_step3cr_cr0_protocol.py verify-artifacts
```

## Continuation

P0 passes. C-R0 independent Reviewer A/B labeling is authorized next. No model diagnostic or optimizer step is authorized before P1 passes.
