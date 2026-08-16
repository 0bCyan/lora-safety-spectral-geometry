# Step3 Exploratory Pilot Report

## Status

Route: `KEEP_C0R_STOP_C3A_C3B_EXPLORATORY`.

This is an exploratory, non-confirmatory pilot. It does not alter the frozen Step3Q stop records or support a formal research conclusion. The failed 90-row Step3Q subagent spool was not read, imported, or reused.

## Execution

- Hardware: NVIDIA GeForce RTX 4070 Ti, Qwen3-4B in 4-bit LoRA mode.
- Verification: 9 training-common tests and 6 T2R tests passed.
- Training data: existing 256 risk rows and 80 approved safety targets over 40 training prompts.
- Development data: the existing 40-row `proxy_dev` split only.
- Exploratory optimizer updates: 54 total, including smoke and three 16-step arms.
- Generation: 120 development responses, greedy decoding, 128-token cap, thinking disabled.
- Access boundary: `heldout_opened=false`, `core100_opened=false`, and no API authoring data used.

## Results

| Arm | Step 16 loss | Refusal marker | Redirect marker | Procedural marker | Mean tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| C0R constant-20 | 1.884 | 57.5% | 17.5% | 30.0% | 45.7 |
| C3a global binding | 2.039 | 42.5% | 7.5% | 35.0% | 54.3 |
| C3b hard-negative binding | 2.056 | 45.0% | 5.0% | 35.0% | 53.5 |

All three arms completed 16/16 steps with finite losses and gradients, zero skipped updates, and about 4.87 GiB peak allocated VRAM. All 120 responses were non-empty, no thinking tags were present, and no exact normalized response duplicates occurred.

Against C0R, C3a gained one refusal-marked response but lost seven; C3b gained one and lost six. Both binding arms added three procedural-marked responses and removed one. The two binding variants therefore fail this pilot's quick behavioral screen despite clean numerical execution.

## Decision

Keep C0R as the local reference and stop C3a/C3b expansion. These marker-based measurements are only triage signals, not blinded safety judgments. The next useful method change is a weaker or delayed binding schedule, tested first with the same short paired design.
