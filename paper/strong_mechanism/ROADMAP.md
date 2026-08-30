# Strong Mechanism Paper Roadmap

## M0 — Infrastructure and claim freeze

Deliverables:

- branch and paper directory;
- frozen claim ledger;
- source data/code manifest;
- prospective protocols P1–P6;
- model revision and 4070 Ti environment lock;
- schedule validator, command renderer, evaluation pipeline, aggregation scripts, tests;
- evidence-gated LaTeX skeleton.

Exit gate: CI passes and no manuscript claim exceeds `CLAIMS.md`.

## M1 — Equal-dose temporal-distribution qualification

1. create the strict 4070 Ti environment;
2. resolve the exact model snapshot online and then offline;
3. materialize all 15 runtime configs;
4. render 15 training and 45 evaluation/audit commands;
5. execute all 60 dry-runs;
6. run one non-formal one-step smoke job outside seeds 1–3;
7. create and commit `preflight/M1_preflight_lock.json` from a clean worktree.

Exit gate: immutable M1 lock committed before the first formal model load.

## M2 — Qwen3-4B three-seed mechanism experiment

Principal arms:

- `front_loaded_equal_dose`;
- `uniform_equal_dose`;
- `back_loaded_equal_dose`.

Harmonized context arms:

- `natural_diverse_10`;
- `natural_constant20`.

Required checkpoints: 0, 6, 16, 32, 48, 64.

Required outputs:

- adapter/checkpoint index;
- exact anchor-draw and exposure manifests;
- safety100 and utility100 generations;
- registered gradient audit at 32/48/64;
- environment, model revision, command, and file hashes.

Exit gate: all 15 runs complete or technical missingness is preserved without replacement.

## M3 — Final quality remeasurement

- validate and materialize the registered generation tables;
- freeze safety, quality, and utility calibration packets;
- construct arm-blind A/B packets;
- require calibration, duplicates, and component reliability before adjudication;
- preserve prompt-level labels for paired statistics.

Exit gate: reliable arm-blind safety and quality datasets plus integrity reports.

## M4 — Human-validity subset

- select the subset before model-review outcomes are opened;
- two independent humans plus one adjudicator;
- compare components, ordinal boundary, and arm preference with model review;
- report disagreement taxonomy.

Exit gate: registered reliability or an explicit measurement limitation that lowers C4.

## M5 — Second setting

Preferred order:

1. second backbone with the same risk and safety sources;
2. if infeasible for a reason frozen before outcomes, same backbone with a prospectively selected second risk source.

Run only the minimum mechanism conjunction, not the historical exploration tree.

Exit gate: replication, bounded non-replication, or model-specific scope decision.

## M6 — Analysis freeze and prompt-disjoint confirmation

Before opening confirmation:

- freeze manuscript hypotheses;
- freeze exclusions and missingness;
- freeze all analysis, table, and figure code;
- freeze checkpoint/seed aggregation rules;
- record commit and file hashes.

Then open the Paper-A-specific prompt-disjoint panel exactly once.

## M7 — Submission release

- no primary result placeholders or `PENDING EVIDENCE` markers;
- full release validation passes;
- all figures/tables derive from normalized data;
- LaTeX compiles without undefined references/citations;
- claim audit and simulated reviewer attack audit signed off;
- artifact and PDF manifests produced.
