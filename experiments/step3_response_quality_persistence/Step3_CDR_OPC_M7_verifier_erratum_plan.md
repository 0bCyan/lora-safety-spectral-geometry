# Step3 CDR-OPC M7 Verifier Erratum VE1

## Registration status

VE1 is a retrospective, post-result verifier-only erratum.  It is registered
after the M7 measurement, after the original `verify-run` failed, and after an
exploratory in-memory check had already shown that a key-order adapter lets the
frozen full verifier pass.  VE1 is therefore neither prospective nor blinded.

## Defect

The frozen M7 writer serializes JSON with `sort_keys=True`.  Reloading the
diagnostics consequently presents each flattened-table object in alphabetical
key order.  The frozen verifier nevertheless requires
`tuple(flat) == TABLE_FIELDS`, which treats JSON object member order as part of
the schema.  All 80 rows have the exact registered key set and their raw and
flattened numerical values agree exactly; the first failure is
`m5_cal_02 / risk`, whose first reloaded key is `central_deterioration` rather
than `unit_id`.

## Single authorized correction

VE1 may only do the following in memory:

1. require every diagnostics table row to have exactly the member-name set in
   the frozen M7 `TABLE_FIELDS` tuple;
2. construct a fresh mapping with those same values in `TABLE_FIELDS` order;
3. delegate to the byte-identical frozen M7 full verifier; and
4. retain the frozen exact-order check for the CSV header.

The adapter must prove that mapping equality and canonical JSON SHA-256 are
unchanged.  Missing or extra keys, changed values, row-order changes, duplicate
cells, CSV-order changes, hash drift, or failure of any other frozen M7
predicate remain errors.

## Claim boundary

VE1 only determines whether the byte-identical M7 artifact set satisfies every
pre-existing M7 verification predicate after replacing the invalid JSON-object
member-order predicate with exact member-name-set equality.  It performs no
fresh model measurement and changes no artifact, formula, tolerance, gate,
route, dataset, unit, or interpretation.  A VE1 PASS confirms only the original
M7 validity-only route for the unchanged artifact set.  It does not claim that
the original verifier passed and does not expand the M7 scientific claim.

The original M7 route remains
`PASS_CDRO_OPC_M7_VALIDITY_ONLY_SYMMETRIC_DIAGNOSTIC`; the original verifier
outcome remains `ERROR_CDRO_OPC_M7_VERIFY_RUN_IMPLEMENTATION_DEFECT`; and the M6
calibration route remains `STOP_CDRO_OPC_M6_CALIBRATION` with 16/20 covered
units and the sealed audit unopened.

## Prohibited actions

- no rewrite or reserialization of an original M7 artifact;
- no modification of the frozen M7 runner, tests, protocol, manifest, formula,
  tolerance, gate, route, field set, CSV column order, or interpretation;
- no M6 or M7 rerun, model load, CUDA use, loss/gradient evaluation, optimizer
  step, checkpoint, generation, negative-arm measurement, sealed audit,
  behavior panel, radius change, dose change, or result-dependent rescue;
- no new scientific conclusion from the verifier correction.

VE1 writes only its own receipt, report, and reproduction manifest in the
dedicated verifier-erratum result root.  Immediately before the one official
audit it atomically and exclusively writes an immutable attempt claim.  A PASS
or caught failure publishes a terminal receipt; an abrupt process failure still
leaves the claim and blocks any second attempt.  The only terminal erratum
routes are `PASS_CDRO_OPC_M7_VE1_UNCHANGED_ARTIFACTS_VERIFIED`,
`STOP_CDRO_OPC_M7_VE1_TARGET_DRIFT`,
`STOP_CDRO_OPC_M7_VE1_SCOPE_INTEGRITY`, and
`STOP_CDRO_OPC_M7_VE1_FULL_VERIFY`.  VE1 hashes every frozen source before and
after the read-only verification.
