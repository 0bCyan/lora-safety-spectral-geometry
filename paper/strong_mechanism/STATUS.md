# Paper A Status

Last updated: 2026-08-30

## Branch state

- branch: `paper/strong-mechanism-safety-persistence`
- base branch: `codex/step3-exploratory-pilot`
- base commit: `7ffe500224585cf490035863ca640327adb420a1`
- manuscript status: evidence-gated working draft
- submission ready: **false**
- method validated: **false**
- existing global heldout opened for Paper A: **false**

## Completed evidence

- Step 1: risk LoRA behavior failure and diagnostic spectral/subspace/activation signals.
- Step 1: tested post-hoc parameter and activation controls did not recover safety.
- Step 2: early unsafe transition located and replicated across three risk seeds.
- Step 2: sequence-level anchor supervision changed behavior across three seeds.
- Step 2: direct and factor-proxy spectral controls, margin-only objectives, and tested local proxies did not provide behavioral safety.
- Step 2: seed1 `constant20` reduced late unsafe rebound relative to `decay10` under paired blind evaluation.
- Step 2: the same strong persistence control showed response-coverage and template-quality limitations.
- Step 3: several response-quality constructs were audited; unreliable constructs were stopped rather than promoted.
- Step 3: a component actionability scale achieved strong internal reliability on its registered external-source validation, but final Paper A arm remeasurement remains pending.

## Blocking evidence

1. cumulative-dose confounding between `constant20` and `decay10`;
2. final persistence result not yet replicated for seeds 2 and 3;
3. final paper arms not yet scored with one frozen prompt-aware rubric;
4. no independent human-validity subset for the final rubric;
5. no second model or second risk setting for the central mechanism;
6. no paper-specific prompt-disjoint confirmation;
7. no final integrated result tables and figures;
8. manuscript currently contains placeholders for all new confirmatory results.

## Current stage

`M0_PAPER_INFRASTRUCTURE`

The next legal scientific action is to validate and freeze Protocol P1. No new method search is part of Paper A.

## Completion milestones

| Milestone | Requirement | State |
|---|---|---|
| M0 | paper workspace, claims, protocols, validation skeleton | in progress |
| M1 | equal-dose schedules pass static and dry-run checks | pending |
| M2 | Qwen3-4B three-seed temporal experiment complete | pending |
| M3 | final blinded quality remeasurement complete | pending |
| M4 | human-validity subset passes registered reliability gates | pending |
| M5 | second setting reproduces minimum mechanism conjunction | pending |
| M6 | analysis frozen and confirmatory panel executed once | pending |
| M7 | all release gates and LaTeX build pass | pending |
