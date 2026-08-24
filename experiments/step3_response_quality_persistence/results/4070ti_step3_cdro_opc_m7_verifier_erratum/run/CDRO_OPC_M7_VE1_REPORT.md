# CDR-OPC M7 Verifier Erratum VE1 Report

VE1 route: `PASS_CDRO_OPC_M7_VE1_UNCHANGED_ARTIFACTS_VERIFIED`. Underlying frozen M7 route: `PASS_CDRO_OPC_M7_VALIDITY_ONLY_SYMMETRIC_DIAGNOSTIC`.

VE1 is retrospective, post-result, and non-blinded. The original failure and an exploratory in-memory normalized PASS were both known before VE1 registration; the exploratory PASS was not an official erratum attempt.

The original M7 verifier outcome remains `ERROR_CDRO_OPC_M7_VERIFY_RUN_IMPLEMENTATION_DEFECT` with `M7 flattened table does not reproduce`. VE1 required exact member-name sets for 80 JSON rows, rebuilt only their in-memory mapping order, and delegated every remaining predicate to the frozen full M7 verifier. Mapping equality and canonical SHA-256 were unchanged; the exact CSV header-order predicate was retained.

All frozen source SHA-256 values were identical before and after. VE1 performed zero original-artifact writes, model measurements, loss writes, gradient evaluations, optimizer steps, parameter writes, checkpoints, generations, or sealed-audit access. It does not expand the M7 scientific claim, and the M6 16/20 calibration STOP remains in force.
