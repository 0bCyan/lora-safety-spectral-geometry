# Experiment Machine Task: Keep Idle During Step3O-O4 Local Pool Work

## Current State

Pull the latest `exp/step3o-o4-attempt2` for records only. There
is no active GPU task.

```text
T5R route:       T5R4_complete_labels_authorize_T6
T6 route:        R1_no_c3_candidate_passes
Track B:         M1_reconstruction_integrity_failure
Step3N-R1:       STOP before tuple writing or training
Step3O O1/O2:   PASS
Step3O O3:      PASS external actionability validation
Step3O-O4:      prospective public-randomness candidate-pool protocol; local only
```

Do not run training, model generation, model scoring, gradient extraction,
matching, T6, PCSR, TRIM, or method smoke tests.

## New Completed Result

Step3O replaced the unreliable directly assigned actionability ordinal with
three judged components: `safe_route`, `executable_step`, and
`prompt_binding`. The strict score-2 boundary is derived only when all three
are yes.

The untouched external validation used 120 BeaverTails primary rows plus 12
hidden duplicates and passed every frozen gate:

- component raw agreement: `0.942-0.958`;
- component Gwet AC1: `0.886-0.931`;
- derived ordinal exact agreement: `0.925`;
- derived ordinal Gwet AC2: `0.890`, O3 registered prompt-resampling lower
  sensitivity bound `0.808`;
- strict-boundary raw agreement: `0.958`;
- strict-boundary Gwet AC1: `0.931`, O3 registered prompt-resampling lower
  sensitivity bound `0.865`;
- hidden-duplicate complete-vector consistency: A `12/12`, B `12/12`;
- rubric revisions after validation opened: `0`.

This is a positive observed-agreement result for two fixed,
context-isolated same-model sessions. The resampling bounds describe O3 packet
sensitivity, not replacement-reviewer uncertainty. The result is not human
construct validity, a behavioral gain, a mechanism result, or causal proof.

## Active Local Work

`Step3O_O4_fresh_training_pool_plan.md` prospectively defines the next local
stage. It can construct only a machine-rubric-qualified candidate pool. It
will:

1. deterministically select exactly 60 unused PKU-SafeRLHF training prompts;
2. exclude all evaluation prompts, prior Step3 split IDs, old 40/37 prompts,
   tracked risk-ID exposures, and registered incidental prompt exposures;
3. release prompts only, never source harmful responses;
4. use two fresh, context-isolated arm-blind authors for exactly one candidate
   each per prompt;
5. use fresh, context-isolated A/B/C review sessions to screen detail, the three O3-screened
   actionability components, and the new O4-only naturalness and clean-safety
   fields;
6. require at least 48 of the fixed 60 prompts plus all category/severity
   retention minima;
7. prohibit prompt replacement, a third candidate, or another authoring round.

All selection and packet-order salts are bound to the first pushed P0 commit
plus P0-fixed future drand round `6320486` (`2026-07-26 14:00:00
Asia/Shanghai`). Its request receives a commit-bound GitHub Actions check before
publication. The full validation, statistics, adjudication,
and route implementation is frozen in P0 before any author or reviewer output.
R0 and R1 each use a separate single-path start commit that must be pushed and
immutable-release locked before the supervised result command can contact
drand. The result command freezes its strictly validated ordered mirror trace
and either produces its registered artifact or a terminal
`STOP_PROTOCOL_INTEGRITY`. A failed stage is never rerun within the same
attempt. P1 prompt materialization additionally holds a cross-process exclusive
worker lease; timeout or manual sealing can freeze only a complete ordered
output prefix after the worker has released that lease.
P0, R0-start, and either R0 success or R0 terminal failure must all be locked
before the fixed target round; a late R0-start is rejected before network access.

The accepted evidence chain does not claim that no hidden local clone ever
executed. It fixes the seed-bearing round in P0 and gives success and failure a
single shared outcome tag, so only one fixed-target outcome can be accepted.

All O4 prompt selection, authoring, and review work is local. The experiment
machine stays idle.

## Verification

```text
python experiments/step3_response_quality_persistence/scripts/step3o_measurement_repair.py verify-o0
python -m unittest discover experiments/step3_response_quality_persistence/scripts -p test_step3o_measurement_repair.py -v
python -m unittest discover experiments/step3_response_quality_persistence/scripts -p test_step3o_o4_training_pool.py -v
python -m unittest discover experiments/step3_response_quality_persistence/scripts -p test_step3o_o4_randomness.py -v
python -m unittest discover experiments/step3_response_quality_persistence/scripts -p test_step3o_o4_qualification.py -v
python -m unittest discover experiments/step3_response_quality_persistence/scripts -p test_step3o_o4_stage.py -v
```

Frozen aggregate O3 evidence is at:

```text
experiments/step3_response_quality_persistence/results/step3o_measurement_repair/reports/Step3O_validation_route_report.md
experiments/step3_response_quality_persistence/results/step3o_measurement_repair/reports/step3o_validation_review_agreement.json
experiments/step3_response_quality_persistence/results/step3o_measurement_repair/reports/step3o_validation_route_decision.json
```

## Prohibitions

- Do not modify P6 or replace M1.
- Do not rerun label64 or report approximate grouped values.
- Do not rerun, relabel, resample, or revise Step3O O1-O3.
- Do not release the old 87 rows, old 40 prompts, or old 37-prompt selection.
- Do not expose PKU source harmful responses to O4 authors or reviewers.
- Do not replace O4 prompts or add candidates after authoring begins.
- Do not implement or train PCSR-R2, TRIM, or a new method.
- Do not run C4/C5, seeds 2/3, a second backbone, SpectraSafe, or heldout.
- Do not commit checkpoints, adapters, caches, raw API caches, full gradients,
  or unrelated paper files.

## Future Experiment-Machine Entry

The experiment machine receives a task only if O4 first passes its fixed-pool
yield and review-reliability gates, a separate human or genuinely
heterogeneous training-suitability audit passes, and a new method protocol is
reviewed, committed, and pushed. That later protocol must define fresh method
names, data hashes, controls, implementation tests, smoke gates, sample size,
and failure routes. An O4 pass alone still authorizes zero optimizer steps.
