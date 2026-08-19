# CDR-OPC M3 Blind Review V3 Transport Repair Plan

Status: prospective after V2 stopped at the platform safety gateway and before
V3 packet construction.

V3 changes exactly one V2 packet component. The twelve raw PKU source-unsafe
anchors are replaced by twelve byte-frozen Step3Q clean-safety calibration
anchors whose hidden reference is `clean_safe=no`. These anchors were designed
for safe blind transport and were previously deliverable through the platform.

All other V2 elements remain unchanged:

- the same sealed 120 primary candidates;
- twelve positive component anchors;
- twelve deterministic generic negative anchors;
- twelve deterministic cross-prompt mismatch anchors;
- twelve hidden primary duplicates;
- 180 rows per reviewer;
- the same qualification semantics and TSV finalizer;
- the same anchor, duplicate, A/B agreement, AC1/AC2, and bootstrap gates;
- Reviewer C only after a complete A/B pass;
- seed `380021`, forty eligible prompts, and the 20/20 calibration/audit split.

Two new no-context Reviewers F/G are required. V1 A/B labels, V2 Reviewer E
labels, all prior short reasons, and all prior hidden maps are forbidden inputs.
No retry or reviewer replacement is allowed after launch.

V3 is a transport repair, not a fresh candidate replication. If either
reviewer remains unavailable, record `STOP_M3_REVIEW_V3_INCONCLUSIVE`. If
calibration or reliability fails, record `STOP_M3_REVIEW_V3_UNRELIABLE`.

```text
optimizer_steps=0
training_authorized=false
core100_opened=false
heldout_opened=false
```
