# Experiment Machine Task: Step3C-T Seed1 C3 Training Tournament

## Authorization

Pull `exp/step3-response-quality-persistence` and execute only:

```text
experiments/step3_response_quality_persistence/Step3C_T_seed1_c3_training_tournament_plan.md
experiments/step3_response_quality_persistence/configs/step3ct_seed1_training_registry.json
```

Step3C-MF is complete. Its joint decision passed measurement validation and no-update feasibility for C3a/C3b, then stopped before training. This handoff prospectively authorizes a staged seed1 tournament for exactly three arms:

```text
C0R_concurrent_constant20
C3a_multi_positive_global_binding
C3b_hierarchical_hard_negative_binding
```

No other arm or behavior-driven revision is authorized.

## Global Prohibitions

- Do not run C4, C5, adaptive replay, SpectraSafe, seeds 2/3, a second backbone, or heldout.
- Do not change eta, temperature, queue size, target text, target exposure, decoding, projection architecture, or endpoint thresholds after candidate output is visible.
- Do not train the projection head.
- Do not restore mechanical-template or holistic pair-preference labels as promotion endpoints.
- Do not use local heuristics as final outcome labels.
- Do not let a main-session executor that knows arm identities assign blind labels.
- Do not commit adapters, checkpoints, caches, large raw generations, or unrelated paper files.

## Preconditions

1. Record pulled `HEAD`, environment, GPU, CUDA, torch, transformers, peft, bitsandbytes, and available VRAM.
2. Confirm the Step3C-M validation route and C3 joint route both pass and match the registry hashes.
3. Confirm the 80 natural-v1 targets cover exactly 40 safety-train prompts with two approved targets per prompt.
4. Confirm the constant20 source config and historical behavior/utility reports exist.
5. Confirm core100 is used only after all three main run manifests freeze; confirm heldout is never opened.
6. Confirm no unrelated tracked or untracked file will be committed.

## Package T1: Trainer Implementation and No-Update Freeze

Implement a separate Step3C-T path with shared code for all arms. Preserve the inherited constant20 trainer contract and use:

```text
L_total = 0.80 * L_risk + 0.20 * (L_safe + eta * L_bind)
```

The binding term must remain inside the safety multiplier. C0-R uses `eta=0` and does not construct or mutate a queue.

### Required implementation decisions

- Fixed two-layer GELU projection, dimension 256, seed `20260727`, excluded from optimizer state.
- Prompt and response pooling identical to the frozen no-update diagnostic.
- Queue size 32 with deterministic prefill/FIFO update, detached negatives, and same-prompt exclusion.
- C3a uses global cross-prompt negatives.
- C3b adds the registered same-category hard term with multiplier 1.0.
- C3a eta `0.08881576937607806`.
- C3b eta `0.05694228453242218`.
- Binding forwards use disabled dropout and isolated CPU/CUDA RNG state.

### Required tests

- source/config/hash/schema integrity;
- left/right padding, truncation, prompt/response masks;
- two-positive indexing and positive exclusion from negatives;
- queue initialization, wraparound, resume, and determinism;
- singleton-category fallback;
- C3b equality to C3a with hard term disabled;
- frozen projection head and frozen base model;
- finite non-zero registered LoRA gradients;
- objective-parentheses test proving the 0.20 multiplier applies to binding;
- C0-R loss/draw equivalence with zero-weight binding toggled;
- RNG isolation across task/safety forwards;
- deterministic checkpoint resume;
- direct C3a-C3b gradient cosine and energy-location comparison;
- at least 10% VRAM headroom.

Freeze code, config, source hashes, tests, environment, queue policy, run order, and exact commands. Commit and push T1 before any smoke optimizer step.

## Package T2: Three-Arm Smoke

### Disposable smoke

Run exactly two optimizer steps for each registered arm without generation or evaluation. Apply every smoke gate in the plan. If code/config changes, archive the failed smoke, re-freeze, and rerun all three arms.

Commit and push all three smoke manifests and gate results. No candidate generation or core100 access is allowed.

## Package T3: Full Three-Arm Training

After smoke passes, train all three arms to step64 with checkpoints:

```text
0, 6, 16, 32, 48, 64
```

Required run evidence:

- identical risk order and safety draw trace across arms;
- exact target exposure counts and hashes;
- component loss and coefficient traces;
- queue occupancy/insertion/eviction hashes;
- gradient norm/ratio/cosine diagnostics at registered steps;
- representation retrieval/gap/rank diagnostics at steps 0/16/32/64;
- checkpoint index and adapter hashes without committing weight files;
- runtime and peak VRAM;
- explicit core100=false and heldout=false during training.

Commit and push T3 after all three run manifests and integrity checks freeze.

## Package T4: Generation and Packet Freeze

Generate with identical greedy decoding:

```text
safety100: steps 16, 32, 64 for all three arms
utility100: step 64 for all three arms
```

Local diagnostics are integrity checks only. Do not select an arm from them.

Prepare separate arm-blind packets:

1. safety behavior at steps 32 and 64;
2. decomposed quality at step64;
3. utility at step64.

Use opaque IDs, balanced random order, hidden arm/checkpoint keys, packet manifests, source hashes, and hidden duplicates equal to 10% of primary rows rounded up with minimum 8 and maximum 30. Freeze all generation manifests and blind packets in T4 before any outcome label. Commit and push.

## Package T5: Independent Blind Review

For each packet type:

1. launch fresh-context Reviewer A and Reviewer B with only the packet and the frozen relevant rubric;
2. require complete independent labels for every primary and duplicate row;
3. freeze A/B files and hashes before comparison;
4. generate disagreement-only Reviewer C packets without A/B labels or reasons;
5. launch fresh Reviewer C and require complete disagreement coverage;
6. compute agreement and duplicate consistency before joining hidden arm keys;
7. join roles only after all labels and adjudication freeze.

Safety and decomposed quality reviewers must be separate sessions. Record reviewer IDs, allowed inputs, row counts, timestamps, hashes, blindness declarations, and main-executor non-labeling confirmation.

Apply every blind-review reliability gate in the plan and registry before adjudicated outcomes are used. A zero-variance field requires raw agreement `>=0.95` and records chance-corrected agreement as undefined. No rubric revision or complete rereview is permitted after candidate output exists.

Commit and push T5 labels and agreement artifacts before route analysis.

## Package T6: Statistics and Route

Compute:

- step16/32/64 safety trajectories;
- concurrent C0-R reproducibility gates;
- hard safety and utility gates for both candidates;
- `specific_safe_success` non-inferiority;
- `high_specific_safe_success` superiority;
- shared-clean-safe detail/actionability analyses;
- paired bootstrap intervals, discordant counts, exact McNemar tests, and Holm correction;
- mechanism trajectories and C3a/C3b incremental analysis;
- complete gate table and one registered route.

Allowed routes:

```text
R0_integrity_control_or_review_failure
R1_no_c3_candidate_passes
R2_select_C3a_as_Q_star
R3_select_C3b_as_Q_star
```

If both candidates pass but C3b improves primary behavior by less than 0.05, select C3a. If neither passes, close the current C3 recipe without a behavior-driven rescue scan.

Commit and push the final route package. Stop. A selected winner only requests a new C4/Q*/C5 seed1 factorial plan.

## Required Return State

Every package report must include:

- entry/result commit and exact commands;
- wall-clock runtime and environment;
- source, code, config, target, packet, and output hashes;
- arm-level row counts and checkpoint completeness;
- optimizer parameter inventory and projection/base freeze checks;
- draw/exposure/queue traces;
- all registered metrics, confidence intervals, and gates;
- reviewer provenance and agreement where applicable;
- core100 access phase and heldout=false confirmation;
- next authorized package or explicit stop.
