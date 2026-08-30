# Paper A Status

Last updated: 2026-08-30

## Branch state

- branch: `paper/strong-mechanism-safety-persistence`
- base branch: `codex/step3-exploratory-pilot`
- base commit: `7ffe500224585cf490035863ca640327adb420a1`
- current branch commit: tracked by Git; do not hard-code after each infrastructure edit
- manuscript status: evidence-gated working draft
- submission ready: **false**
- method validated: **false**
- existing global heldout opened for Paper A: **false**
- formal Paper A training started: **false**

## Formal Qwen3-4B setting

- machine: RTX 4070 Ti only for the primary setting;
- model repository: `Qwen/Qwen3-4B`;
- pinned revision: `1cfa9a7208912126459214e8b04321603b3df60c`;
- strict training environment: `environment/4070ti_training_lock.json`;
- registered new formal runs: **15**;
- equal-dose principal runs: 9;
- harmonized `decay10/constant20` context replications: 6.

Historical seed1 results remain valid historical evidence, but they are not pooled with the final replication because the resolved model revision was not recorded. The final context comparison prospectively reruns seeds 1–3 under one pinned snapshot and environment.

## Completed evidence

- Step 1: risk LoRA behavior failure and diagnostic spectral/subspace/activation signals.
- Step 1: tested post-hoc parameter and activation controls did not recover safety.
- Step 2: early unsafe transition located and replicated across three risk seeds.
- Step 2: sequence-level anchor supervision changed behavior across three seeds.
- Step 2: direct and factor-proxy spectral controls, margin-only objectives, and tested local proxies did not provide behavioral safety.
- Step 2: historical seed1 `constant20` reduced late unsafe rebound relative to `decay10` under paired blind evaluation.
- Step 2: the same historical persistence control showed response-coverage and template-quality limitations.
- Step 3: unreliable response-quality constructs were stopped rather than promoted; a component actionability scale passed its registered internal validation.

## Infrastructure completed

- claim and evidence-tier freeze;
- P0–P6 prospective protocol chain;
- exact equal-dose schedule configs;
- model snapshot and environment lock;
- immutable 15-run command renderer;
- source-hash validator;
- prompt-paired and seed-hierarchical statistics;
- blind packet, calibration, duplicate, and adjudication tooling;
- deterministic figure and table generation;
- evidence-gated complete LaTeX draft;
- GitHub Actions for Python, artifact-pipeline, and LaTeX verification.

## Blocking scientific evidence

1. equal-dose temporal-distribution runs have not been executed;
2. final `decay10/constant20` comparison has not been rerun under the pinned setting;
3. final paper arms have not been scored with the frozen prompt-aware rubric;
4. no independent human-validity subset exists for the final rubric;
5. no second model or second risk setting has tested the central mechanism;
6. no paper-specific prompt-disjoint confirmation has been opened;
7. final integrated result tables and figures remain placeholders.

## Current stage

`M1_EQUAL_DOSE_PREFLIGHT`

The next legal actions are:

1. confirm CI is green;
2. create the locked 4070 Ti environment;
3. prefetch and verify the pinned model snapshot;
4. render, review, commit, and hash the 15-command manifest;
5. execute all dry-runs;
6. run one explicitly non-formal smoke job outside seeds 1–3;
7. freeze the M1 registry before the first formal model load.

No new method search is part of Paper A.

## Completion milestones

| Milestone | Requirement | State |
|---|---|---|
| M0 | paper workspace, claims, protocols, validation, LaTeX | complete subject to green CI |
| M1 | pinned environment and all schedule dry-runs pass | pending |
| M2 | Qwen3-4B 15-run formal matrix complete | pending |
| M3 | final blinded quality remeasurement complete | pending |
| M4 | human-validity subset passes registered gates | pending |
| M5 | second setting reproduces minimum mechanism conjunction | pending |
| M6 | analysis frozen and confirmatory panel executed once | pending |
| M7 | all release gates and final PDF pass | pending |
