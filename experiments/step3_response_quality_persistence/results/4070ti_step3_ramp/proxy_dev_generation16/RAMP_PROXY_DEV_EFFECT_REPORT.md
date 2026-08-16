# RAMP Prompt-Disjoint Proxy-Dev Effect Report

Route: `STOP_RAMP_PROXY_DEV_EFFECT_GATE_FAILED`.

RAMP early-safe-route deltas were
`0.050000` versus uniform
and `0.075000` versus the
suffix placebo. Paired bootstrap lower bounds were
`-0.050000` and
`-0.025000`; exact
two-sided McNemar p-values were
`0.625000`
and `0.375000`.
Both independently passed every frozen non-weak check:
`False`.

The 40 free-generation prompts had zero overlap with all approved RAMP training
prompt clusters. The token-prefix marker is a mechanism-linked deterministic
proxy, not a semantic safety judgment. Core100 and heldout remained sealed;
generation used `optimizer_steps=0`. A pass authorizes only separate blinded
safety and utility review protocols. A failure stops RAMP expansion.
