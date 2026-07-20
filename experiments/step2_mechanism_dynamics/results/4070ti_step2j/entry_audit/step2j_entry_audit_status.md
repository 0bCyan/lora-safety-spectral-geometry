# Step2J-A Entry Audit Status

## Independent Target Review Packet

Prepared blinded review packet:

`experiments\step2_mechanism_dynamics\results\4070ti_step2j\entry_audit\step2j_natural_v1_independent_target_review_blinded.jsonl`

Rows: 80

Status: pending external independent agent or human review. The existing v1 review remains a local preflight, not an independent approval.

## Evidence Integrity

| check | result |
| --- | --- |
| manifest rows | 80 |
| exposure rows | 80 |
| total anchor draws | 256 |
| exposure min/max | 3 / 4 |
| exposure mean | 3.200 |
| target hash order matches manifest | True |
| manifest hash matches freeze | True |
| review hash matches freeze | True |
| tracked evidence hashes match run source | True |
| formal decay-to-10 control usable | True |

Text manifest and preflight-review hashes are computed after normalizing CRLF to LF for freeze comparisons.
Tracked reproducibility artifacts are copied to `experiments\step2_mechanism_dynamics\results\4070ti_step2j\entry_audit\evidence` and hashes are recorded in `step2j_natural_v1_evidence_integrity.json`.
