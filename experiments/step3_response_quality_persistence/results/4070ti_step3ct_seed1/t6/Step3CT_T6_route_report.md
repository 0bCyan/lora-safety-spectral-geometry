# Step3C-T6 Tournament Route Report

## Decision

Route: `R1_no_c3_candidate_passes`

T6 joined only the frozen T5R4 adjudicated labels to the T4 hidden keys. Statistics used prompt ID as the paired unit, 10,000 bootstrap resamples with seed `2026072404`, exact McNemar tests, and Holm adjustment across the two candidate-versus-control primary comparisons.

## Denominators

- safety: 100 primary prompts per arm at steps 32 and 64;
- quality: 100 primary prompts per arm at step64;
- utility: 100 primary prompts per arm at step64;
- heldout: unopened.

All control, safety, quality, and utility gates are recorded in `step3ct_t6_gate_table.csv`; all point estimates, intervals, discordant counts, and adjusted/unadjusted p-values are in `step3ct_t6_statistics_and_route.json`.

No weighted composite, endpoint-only interpretation, behavior-driven scan, new training, or heldout access was used.
