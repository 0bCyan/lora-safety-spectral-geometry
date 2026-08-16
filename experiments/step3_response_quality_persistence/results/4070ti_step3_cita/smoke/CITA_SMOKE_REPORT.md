# CITA Matched Three-Arm Smoke Report

Route: `PASS_CITA_SMOKE_AUTHORIZE_MATCHED_DEVELOPMENT16_PROTOCOL`.

All three arms were launched by one hash-bound orchestrator from fresh seed-1 LoRA states. Each arm executed `2` optimizer steps with the same risk order, safety target order, target exposure, response-token count, and total safety weight budget. First-step risk and uniform diagnostic safety losses matched within `1e-07`.

The final adapter hashes are distinct across uniform, CITA, and sham: `True`. This confirms execution of the three assignments, not behavioral benefit. No generation or evaluator was run, and core100/heldout remained sealed. Total optimizer steps across arms: `6`; `main_training_authorized=false`.
