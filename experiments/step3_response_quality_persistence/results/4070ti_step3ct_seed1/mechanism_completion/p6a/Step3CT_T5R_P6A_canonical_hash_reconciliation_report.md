# Step3C-T5R P6A Canonical Hash Reconciliation Report

Route: `P6A1_eol_reconciled_content_unchanged`

## Result

The legacy P6 manifest tracked 21 artifacts. 2 already match committed LF bytes, and 19 match exactly after canonical LF bytes are re-expanded to CRLF. Unexplained artifacts: 0. HEAD blob mismatches: 0.

The discrepancy is therefore an end-of-line freeze defect rather than a numerical or semantic content change. This addendum records SHA-256 over committed UTF-8/LF bytes and the corresponding Git blob OID.

## Boundary

No P6 file was overwritten. The mechanism route remains `M1_reconstruction_integrity_failure`; label64 grouped metrics remain unregistered, Track A remains without behavior labels, and no candidate or new training is authorized.
