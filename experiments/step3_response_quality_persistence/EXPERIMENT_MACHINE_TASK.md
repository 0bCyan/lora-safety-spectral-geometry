# Experiment Machine Task: Step3C-MF Measurement and No-Update Feasibility

## Authorization

Pull `exp/step3-response-quality-persistence` and execute only the workflow in `Step3C_MF_decomposed_measurement_and_c3_feasibility_plan.md`.

The previous `Step3C_R_batched_execution_plan.md` stopped at C-R0 after its second complete reliability failure. Its P2-P5 training path is no longer authorized. Step3C-MF is a new measurement-development and no-update feasibility study, not C-R0 v3.

Authorized:

- independent development review and at most one development-only rubric clarification;
- one development-untouched validation review after the active rubric freezes;
- C3a/C3b implementation and no-update representation/gradient/memory diagnostics;
- mechanism atlas and joint route decision;
- commits and pushes at the frozen boundaries below.

Not authorized:

- reuse of old C-R0 labels as Step3C-M labels;
- any validation rubric revision;
- optimizer or scheduler construction, optimizer steps, candidate training, or candidate generation;
- core100 behavior inspection for protocol or hyperparameter decisions;
- heldout, seed2/3, second backbone, SpectraSafe, or paper claims.

## Preconditions

1. Pull the active branch and record `HEAD`.
2. Confirm no unrelated tracked or untracked files will be committed.
3. Run `python -m py_compile` on both new local scripts.
4. Run `prepare_step3cm_measurement_protocol.py test-fixtures` and `verify-protocol`.
5. Confirm the protocol manifest has zero mismatches, 40 safety-train prompt clusters, 127 primary source responses, and no core100/heldout/candidate output.
6. Confirm the development and validation prompt sets are disjoint at 20/20.
7. Confirm validation packet content remains sealed from all development reviewers.
8. Confirm Qwen3-4B, the constant20 step64 adapter, CUDA, and at least 10% VRAM headroom before F1.

## Package E1: Development Review and Active Rubric Freeze

### Independent A/B review

Launch Reviewer A and Reviewer B as separate fresh-context agents. Each receives only:

- `results/step3cm_protocol/packets/development/step3cm_development_packet.jsonl`;
- the active decomposed-quality rubric.

They must not receive hidden keys, duplicate keys, C-R0 labels, role identities, prior reports, git history, another reviewer's output, or validation content.

Each reviewer fills all 71 rows with exactly:

```text
opaque_quality_id
detail_retention
safe_actionability
naturalness
short_reason
```

Save outputs at the fixed `results/step3cm_protocol/reviews/development/` paths. Record fresh-context agent IDs and allowed inputs in a provenance file.

### Freeze, adjudicate, and summarize

Run:

```text
python experiments/step3_response_quality_persistence/scripts/prepare_step3cm_measurement_protocol.py validate-ab --split development
python experiments/step3_response_quality_persistence/scripts/prepare_step3cm_measurement_protocol.py prepare-adjudication --split development
```

Launch fresh Reviewer C with only the disagreement packet and active rubric. C must not see A/B labels, A/B reasons, keys, roles, or reports. C returns a complete review-schema row for every disagreement ID.

Then run:

```text
python experiments/step3_response_quality_persistence/scripts/prepare_step3cm_measurement_protocol.py summarize --split development
```

If development passes, run `freeze-active-rubric` and continue. If it fails, use at most one complete development-only clarification. Archive failed labels and hashes, update `active_rubric` prospectively, verify that both packet contents are unchanged, and rerun complete development with fresh A/B/C reviewers. A second failure closes the measurement route without opening validation.

```text
python experiments/step3_response_quality_persistence/scripts/prepare_step3cm_measurement_protocol.py freeze-active-rubric
```

This command binds the final rubric hash and passing development-route hash to the unchanged validation packet. Do not rerun the initial `freeze-protocol` command after development labels exist.

Commit and push E1 with development labels, provenance, adjudication, agreement, route, active rubric, tests, and manifests.

## Package E2: One-Shot Validation

Proceed only after E1 passes. Freeze the final active-rubric hash into the validation manifest and prove the validation packet hash matches the local M0 freeze.

Launch fresh Reviewer A and Reviewer B that have not seen development labels, disagreements, reports, or validation keys. Give each only the 72-row validation packet and active rubric. Freeze complete A/B outputs before comparison.

Use the same `validate-ab`, `prepare-adjudication`, and `summarize` commands with `--split validation`. Reviewer C remains disagreement-only and blind to A/B labels/reasons and all keys.

Validation permits no clarification, relabeling, threshold change, or second session. Apply every registered ordinal, derived-endpoint, naturalness, duplicate, and adjudication gate.

Commit and push E2 with labels, provenance, agreement, route, and complete hash manifest. If validation fails, record the stop and keep training unauthorized.

## Package E3: C3a/C3b No-Update Feasibility

E3 may be implemented while reviews run, but its numerical config cannot depend on review outcomes. It must not generate behavior or construct an optimizer/scheduler.

### Common implementation

Implement one shared diagnostic module for:

- prompt and conditioned-response token masks;
- final-layer mean pooling and L2 normalization;
- two-layer GELU projection head, dimension 256;
- multi-positive InfoNCE;
- optional same-category hard-negative term;
- deterministic queue sizes 32/16/8;
- independent safety-SFT and binding gradient extraction on registered LoRA parameters.

C3a and C3b may differ only through registered candidate flags. Test padding, truncation, positives/negatives, singleton categories, queue wraparound, determinism, hard-term disable equivalence, finite gradients, frozen base parameters, and zero optimizer steps.

### Numerical run

Use only frozen safety-train targets and calibration batches. Apply the registered queue, temperature, positive-probability, VRAM, and `eta` rules without grid expansion.

Produce:

- candidate configs and environment record;
- loss/similarity/gradient raw arrays;
- true and 100 shuffled prompt-target diagnostics;
- retrieval top-1/top-5/MRR/multi-positive recall;
- cosine distributions and bootstrap intervals;
- effective rank and top singular-value share;
- global and layer-group binding/safety gradient cosine;
- layerwise gradient energy and finite/non-zero coverage;
- peak allocated/reserved VRAM, headroom, and runtime;
- candidate pass/fail and exact reason.

## Package E4: Joint Decision

After E2 and E3 freeze, write one joint report covering all four route cells:

| validation | C3 feasibility | decision |
| --- | --- | --- |
| pass | candidate pass | request a new prospective seed1 training plan |
| pass | no candidate | retain measurement, close C3 |
| fail | candidate pass | record numerical feasibility, prohibit training |
| fail | no candidate | close Step3C-MF |

Commit and push E3/E4. Stop after the joint decision. Do not start training even on a joint pass.

## Required Return State

Every commit report must include:

- commit hash, package status, exact commands, and wall-clock runtime;
- Python/torch/CUDA/GPU/model/checkpoint environment where applicable;
- row counts, reviewer IDs, allowed inputs, blindness declaration, and hashes;
- metric values, confidence intervals, every gate, and pass/fail reason;
- optimizer/scheduler object count and optimizer steps, all zero;
- core100 and heldout access confirmation, both false;
- next authorized package or explicit stop.

Do not commit checkpoints, caches, hidden heldout content, or unrelated paper files.
