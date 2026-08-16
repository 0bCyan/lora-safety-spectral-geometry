# CSPR P1 Public-Source Feasibility Protocol

Status: prospective before acquiring or inspecting the frozen BeaverTails source bytes.

## Purpose

P1 asks only whether a fixed public source contains enough fresh same-prompt response contrasts to justify independent label validation. It does not accept BeaverTails safety labels as final labels and does not authorize model gradients or training.

The source is the pinned BeaverTails 30k train file. A source candidate requires one normalized prompt with at least one source-safe response, one source-unsafe response, and at least one unsafe-response category. A response carrying both source labels after normalization is removed from the unsafe side before eligibility is computed.

## Freshness Boundary

Before selection, candidates are compared with the frozen normal, control, and risk training pools, the Step3N source-review packet, the prior Step3O BeaverTails packet, the terminated Step3O-O4 prompt packet, and the non-content Step2G split manifest. Exact prompt hashes, normalized prompt hashes, and the registered 1-to-5-token lexical threshold are used where the frozen reference exposes the necessary field. Core100 and heldout text are not opened. Therefore this is a restricted lexical and hash audit, not semantic decontamination or a claim about model pretraining exposure.

## Fixed Candidate Design

The audit selects 120 unique prompts across all 14 BeaverTails categories using a frozen seed and deterministic scarcity-first allocation. Exactly one source-safe and one distinct source-unsafe response are selected per prompt by stable hash. The ledger contains only hashes, source row indices, category, and prospective split; no prompt or response text is committed.

Each category contributes two calibration candidates and two mechanism-development candidates. The remaining candidates are training candidates, giving frozen split counts of 28 calibration, 28 mechanism-development, and 64 training candidates. Failed review rows cannot be replaced after their labels are observed.

## Gates

The exact source must contain 27,186 rows, 2,480 mixed-label prompts, and 4,898 source-safe responses on those prompts. After the frozen freshness exclusions, at least 2,000 mixed-label prompts and at least 20 prompts per category must remain. Selection must produce 120 unique prompt pairs, every chosen/rejected response must differ after normalization, all category targets must be met, and the prospective split counts must be exact.

Passing P1 authorizes only construction of blinded label-validation packets. Before any real-model gradient, two independent reviewers must establish the reliability of the already validated Step3O component fields and clean-safety boundary on these exact rows. A pair can qualify only if the chosen response is independently judged clean-safe and receives `safe_route=yes`, `executable_step=yes`, and `prompt_binding=yes`, while the rejected response is independently judged not clean-safe. No holistic pair-preference label is used because that construct previously failed reliability.

## Stop Boundary

Any source hash, row count, schema, freshness, supply, category, distinctness, or split failure selects `STOP_CSPR_P1_DATA_INSUFFICIENT`. It does not permit changing thresholds, seeds, responses, categories, or source revision after observing the result. At P1, `optimizer_steps=0`, `training_authorized=false`, `core100_opened=false`, and `heldout_content_opened=false` remain mandatory.
