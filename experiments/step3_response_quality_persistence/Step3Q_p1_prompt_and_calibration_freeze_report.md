# Step3Q P1 Prompt And Calibration Freeze Report

Status: `PASS_P1_PREPARED_FOR_CONTROLLED_API_DISPATCH`.

Route: `step3q_calibrated_api_pool_attempt_001`.

P0 protocol commit:
`510c1a186c214655273a05826acaba2a9e1a2264`.

P1 deterministic namespace:
`f2fa3257d5664b1c5d6747904e301968676fbad95ea0724c76a0589861866356`.

## 1. Decision

The P0 implementation materialized the fixed P1 prompt, request, and
calibration package from a clean worktree whose HEAD exactly matched the full
P0 commit. The independent `verify-prepared` reconstruction matched the entire
P1 file set and every artifact byte.

P1 therefore passes only the source/protocol freeze boundary. It authorizes the
designated experiment machine to verify the two release tags and, if an API key
is available, execute the fixed 120-request dispatcher. No external API call,
candidate-quality review, target selection, method selection, generation from a
local model, or optimizer step has occurred.

## 2. Frozen Denominators

| item | frozen result |
| --- | ---: |
| source rows | 1,000 |
| exclusion-union sample IDs | 184 |
| eligible fresh prompts | 780 |
| selected prompts | 60 |
| API requests | 120 |
| categories | 19, each with 3 or 4 prompts |
| severity 1 / 2 / 3 | 8 / 44 / 8 |
| component anchors | 48 |
| prompt-mismatch interventions | 24 |
| clean-safety anchors | 24 |
| total calibration rows | 96 |

The minimum aggregate supply for any registered category-severity slot was two.
The complete 60-slot one-to-one matching nevertheless succeeded. No replacement
or top-up is allowed after this P1 freeze.

## 3. Reviewer Calibration Feasibility

Each reviewer allocation contains eight paired component anchors, eight
prompt-mismatch interventions, and eight clean-safety anchors. Every selected
mismatch has its matched anchor in the same allocation.

Reviewer A component reference counts are:

| component | yes | no |
| --- | ---: | ---: |
| safe route | 5 | 3 |
| executable step | 3 | 5 |
| prompt binding | 5 | 3 |

Reviewer B component reference counts are:

| component | yes | no |
| --- | ---: | ---: |
| safe route | 4 | 4 |
| executable step | 3 | 5 |
| prompt binding | 4 | 4 |

Both allocations contain exactly four clean-safety yes references and four
clean-safety no references. These are historical codebook references, not human
ground truth or proof of construct validity.

## 4. Frozen Artifacts

| path | bytes | SHA-256 |
| --- | ---: | --- |
| `authoring/api_requests.jsonl` | 44,352 | `a4b2e26a7dfaf80e5069e5d7b20b23fbf0816f01321d241fc66e1d48c2be0443` |
| `authoring/prompts_hidden.jsonl` | 33,307 | `c921d77a7cff0c4a5f1a7c973db7017ebde4eacf4fe4156b51b2631e7a894ffb` |
| `calibration/calibration_bank_packet.jsonl` | 47,236 | `b8b41e05bf575e9c88f3340bbe765a65ea2a36fefcf4e6b8b275e2575102b178` |
| `calibration/calibration_reference_hidden.jsonl` | 29,232 | `12f528136d3009e54454d86b867eada34ba01e70f6ce6fa8cae00e2694ca4fd0` |
| `calibration/sentinel_allocations_hidden.json` | 2,111 | `dc3a9f1d84d1866746343ed8b7fd030863c0aef87f2299c2c38b2dd683d0c07a` |
| `diagnostics/step3q_prepare_diagnostics.json` | 3,013 | `4f3e98f8e309196b138ceea43e194d3373b4d0b484ad40262b83061e9ed30704` |

The authoring-instruction artifact is 1,577 bytes with SHA-256
`eec1146c504d31bd37ca7ce594891032bdc822eecad7e1b37e559cc5050860ad`.

## 5. Current Boundary

```text
external_api_called=false
training_authorized=false
optimizer_steps=0
```

The next lawful action is the experiment-machine verification and controlled
API dispatch defined in `EXPERIMENT_MACHINE_TASK.md`. The receipt spool must
remain outside Git. API completion, including a pass, still does not authorize
primary review until a separate P2 import and review-packet protocol is frozen.
