# Step3C-T4 Generation and Blind-Packet Freeze Report

## Decision

Route: `T4_integrity_pass_packets_frozen_authorize_T5_after_commit_push`

All 12 registered generation cells completed from the pushed T4 implementation freeze at `b3375d2`. Generation integrity, source-output hashes, packet construction, hidden-duplicate checks, reviewer-template blankness, and heldout boundaries passed. T5 independent blind review becomes authorized only after this T4 result commit is pushed.

No safety, quality, utility, arm-ranking, or winner outcome was assigned in this package.

## Generation Completion

| Arm | Safety checkpoints | Utility checkpoint | Rows | Empty | Thinking persisted | Minimum VRAM headroom |
| --- | --- | --- | ---: | ---: | --- | ---: |
| C0-R | 16/32/64 | 64 | 400 | 0 | no | 64.24% |
| C3a | 16/32/64 | 64 | 400 | 0 | no | 64.24% |
| C3b | 16/32/64 | 64 | 400 | 0 | no | 64.24% |

Every cell contains 100 rows and 100 unique sample IDs. All 12 output SHA-256 values match their manifests. The combined manifest-reported generation time is 4,933.394 seconds. Greedy decoding used `max_new_tokens=256`, Qwen3 thinking was disabled, and no residual `<think>` tag, raw response field, empty response, or persisted thinking content was found.

## Frozen Packets

| Domain | Primary rows | Hidden duplicates | Packet rows | Visible fields | Duplicate balance |
| --- | ---: | ---: | ---: | --- | --- |
| Safety | 600 | 30 | 630 | opaque ID, prompt, response | 5 per arm/checkpoint cell |
| Quality | 300 | 30 | 330 | opaque ID, prompt, response | 10 per arm |
| Utility | 300 | 30 | 330 | opaque ID, prompt, response, reference | 10 per arm |

Safety uses steps 32 and 64 for all three arms. Quality and utility use step64 for all three arms. The step16 safety generations remain frozen for the later trajectory analysis but are not exposed to T5 safety reviewers.

All hidden duplicate pairs reproduce byte-equivalent visible content. Arm, checkpoint, sample ID, generation ID, duplicate status, and duplicate links occur only in hidden files. Reviewer A and B templates have identical row order and hashes within each domain, and every decision field is blank.

Packet hashes:

| Domain | Packet SHA-256 | Hidden-key SHA-256 | Duplicate-map SHA-256 |
| --- | --- | --- | --- |
| Safety | `851badc6f208b4ae80861b9b8d38ec713f3c4d8429c36767047f7115c50bc937` | `afd2fdb767afa4edbca427cf00dacfc29978c839ef4812d47f46ee7497e0fbc2` | `782e6f5aa3baadb24e8235853ee98b9d6fe2052e463b772edc0fae199c883b1c` |
| Quality | `c6172853c596ace5d06410254a5d4b63532c7751a9ebfa12c6817e5d816aaa6a` | `e5b6afba25c270eeef9d47db6ee2b38c7c667d438fd0b882a37021a1d5fcc64c` | `afd701d1dd5a4e5c8e1127d257100450c959221b6f286d70d6a3fc74556445b0` |
| Utility | `9050f85e09780a67fb8a9e8415ab589fbd07ebe86918338c1889ffefcba85f56` | `2d3ab965f220a50c30cfa2a101bb427aa494822c82a74794bcbb5db9200f247f` | `77e2917e5b64eb059700b29d7c96bed5961bb2f1d8faae1409db9dcb52b917df` |

The package freeze manifest SHA-256 is `b6b4a28739c991d960d452f2e069914a7b8e84a4f21f19cda518d99ac901cb68`.

## Verification

- four prospective T4 protocol tests passed;
- exactly 12 generation manifests and 1,200 generated rows were verified;
- safety and utility prompt alignment passed;
- all packet, hidden-key, duplicate-map, and reviewer-template hashes were recorded;
- all visible packet schemas were checked for identity or duplicate leakage;
- all reviewer decision fields were confirmed blank;
- no local heuristic or main-session outcome label was used.

Environment: Python 3.11.5, PyTorch 2.5.1+cu121, Transformers 4.53.3, PEFT 0.16.0, CUDA 12.1, NVIDIA driver 591.86, RTX 4070 Ti.

## Boundary and Next Authorization

Core100 and utility100 were opened only for the registered generation and packet freeze. Heldout remained unopened. No optimizer, scheduler, eta revision, new training, C4/C5, seed2/3, second backbone, or SpectraSafe run was authorized or started.

After this commit is pushed, T5 may launch separate fresh-context safety, quality, and utility Reviewer A/B sessions using only the corresponding visible packet, blank template, and frozen rubric. Reviewer C remains disagreement-only. The main executor may prepare packets and compute reliability statistics but may not assign outcome labels.
