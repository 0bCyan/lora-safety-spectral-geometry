# CDR-OPC M3 Authoring Attempt 002 Plan

Status: prospective before Attempt 002 prompt release or author launch.

Attempt 001 is immutably closed at `STOP_M3_AUTHORING_TRANSPORT`. None of its
responses may be reused. Attempt 002 keeps the M3 scientific protocol byte
unchanged and changes only serialization before immutable submission.

Each of the same twelve deterministic ten-row shards is assigned once to a
new no-context current-model subagent. A worker writes exactly ten UTF-8 plain
text drafts named `01.txt` through `10.txt`. Before any immutable submission
exists, the worker may correct a draft that fails the frozen mechanical
validator. The frozen finalizer then checks file set, route opening, length,
thinking tags, and within-shard deduplication and atomically JSON-encodes the
ten rows. Once the JSONL output exists it cannot be overwritten, repaired, or
resubmitted.

This is pre-submission serialization validation, not semantic regeneration or
post-result retry. A missing or invalid final output still selects
`STOP_M3_AUTHORING_TRANSPORT`. No prompt replacement, reassignment, top-up,
threshold change, or Attempt 001 response recovery is allowed.

The Step3Q source bytes, two author modes, registered openings, 40-160 token
envelope, blind qualification requirement, deterministic forty-unit cohort,
20/20 calibration/audit split, M1 direction, M2 conformal-dose rule, and all
confirmation gates remain unchanged.

```text
optimizer_steps=0
training_authorized=false
core100_opened=false
heldout_opened=false
```
