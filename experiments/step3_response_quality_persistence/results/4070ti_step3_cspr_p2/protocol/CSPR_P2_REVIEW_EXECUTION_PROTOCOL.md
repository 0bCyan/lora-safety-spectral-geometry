# CSPR P2 Review Execution Protocol

Status: prospective before any CSPR P2 label is created or opened.

This execution layer does not change the frozen P2 constructs, rubrics, packets, bootstrap settings, reliability gates, pair qualification rule, or fixed-pool yield gates. It freezes only complete-review validation, reviewer provenance checks, exact duplicate checks, A/B reliability computation, disagreement-only Reviewer C packet construction, adjudication, hidden-key joining, and route reporting.

Four distinct fresh no-context sessions review the complete A/B packets: Safety A, Safety B, Actionability A, and Actionability B. Every output must preserve the exact template order, contain a non-empty short reason for every row, and bind its agent/session identity and all input/output hashes in a provenance file. Agent or session reuse across roles is invalid. Reviewers may read only their assigned packet, blank template, and unchanged rubric; hidden keys, pair roles, categories, prospective splits, source labels, another reviewer's output, training results, gradients, and prior conclusions are forbidden.

The main session validates structure and computes the prospectively frozen metrics. It does not assign or repair labels. Reviewer C is authorized only if every A/B reliability and informativeness gate passes, uses a new session for each construct, and receives only the original prompt/response rows on which A and B disagree. The C packet does not expose A/B labels, reasons, disagreement fields, pair identity, role, split, category, or hidden keys. Only disputed fields use C's judgment; agreed fields remain the A/B consensus.

If a reviewer cannot complete the packet because of a safety-policy block, the route is `STOP_CSPR_P2_REVIEW_INCONCLUSIVE`; partial labels are not accepted, no heuristic replaces the reviewer, and no replacement session is silently launched. Invalid structure stops at `STOP_CSPR_P2_REVIEW_INVALID`. Failed reliability stops at `STOP_CSPR_P2_REVIEW_UNRELIABLE`. Failed fixed-pool yield stops at `STOP_CSPR_P2_PAIR_YIELD`. Frozen pairs are never replaced and thresholds are never weakened.

Finalization emits exactly six result artifacts: safety labels, actionability labels, the 120-row qualification ledger, final analysis, route decision, and a human-readable report. A pass authorizes only a separately prospective P3 zero-update mechanism protocol. It does not authorize a model gradient, checkpoint experiment, or training. Throughout P2, `optimizer_steps=0` and `training_authorized=false`; core100 and heldout content remain closed.
