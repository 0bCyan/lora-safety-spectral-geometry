# CSPR P1 v1 Source-Schema Stop Report

Route: `STOP_CSPR_P1_DATA_INSUFFICIENT`.

The exact frozen BeaverTails file was acquired after the earlier transport stop. Its 4,948,694 bytes matched the registered SHA256 `2e24b85dd599224eeb7379da297223bad50e11fea9fee6a7fdb4bd26aedec315`.

The unchanged P1 v1 audit then stopped during source-schema validation because source line 23,155 has an empty response. The frozen v1 parser rejects every empty response before grouping. No response text is recorded in this report. No audit JSON or pair ledger was created, and no freshness, supply, category, or split gate was evaluated.

P1 v1 remains closed and will not be overwritten. A separate repair attempt may only change empty-response handling from whole-source rejection to conservative exclusion from pair eligibility. The source revision and SHA256, exclusion references, lexical threshold, selection seed, category allocation, and `64/28/28` split remain frozen. This repair cannot turn the v1 stop into a pass.

No model gradient was computed. `optimizer_steps=0`, `training_authorized=false`, `core100_opened=false`, and `heldout_content_opened=false`.
