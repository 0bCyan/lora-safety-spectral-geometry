# CSPR P1R Empty-Response Repair Protocol

Status: prospective after the P1 v1 source-schema stop and before any completed source grouping, freshness statistic, category supply, or pair selection.

P1 v1 is permanently closed at `STOP_CSPR_P1_DATA_INSUFFICIENT`. It rejected the exact frozen source when it encountered an empty response at source line 23,155. No v1 result or ledger was created.

P1R changes one rule only: a source row with a string response that is empty after trimming remains part of the exact 27,186-row source count but is excluded from all response maps and therefore cannot establish either side of a same-prompt pair. Empty prompts, non-string responses, category-schema changes, and non-boolean labels remain fatal. Cross-label identical nonempty responses are still removed from the rejected side.

This repair is conservative for pair construction: it removes information and cannot turn an empty response into a chosen or rejected response. The base protocol is inherited by hash. Source revision and SHA256, seven exclusion references, lexical threshold `0.90`, selection seed `2026081603`, 120-prompt category allocation, and `64/28/28` prospective split are unchanged. The exact mixed-prompt and safe-response inventory gates also remain unchanged, so excluding empty rows cannot silently alter the registered source inventory.

Passing P1R authorizes only frozen blinded label-validation packet construction. BeaverTails labels remain candidate signals, not final labels. No model gradient or training is authorized.
