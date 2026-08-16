# TARC Prompt-Disjoint Proxy-Dev Effect Report

Route: `STOP_TARC_PROXY_DEV_EFFECT_GATE_FAILED`.

Exact registered four-token route rates were
`0.650000` (uniform),
`0.750000` (RAMP), and
`0.875000` (TARC). TARC paired deltas
were `0.225000` and
`0.125000`; exact
two-sided McNemar p-values were
`0.011719`
and `0.062500`.

Mean unseen-prompt frozen-pair TARC barriers were
`1.131085` (uniform),
`0.891430` (RAMP), and
`0.466231` (TARC). Positive paired
barrier reductions for TARC were
`0.664854` and
`0.425199`.

Both control comparisons independently passed every frozen non-weak check:
`False`.

The 40 prompts had zero overlap with all TARC training prompt clusters. This
proxy split was used by an earlier RAMP screen and is therefore exploratory,
not fresh confirmation. Exact route completion and trie barriers are
mechanism-linked deterministic proxies, not semantic safety or utility labels.
Core100 and heldout remained sealed; generation used `optimizer_steps=0`.
