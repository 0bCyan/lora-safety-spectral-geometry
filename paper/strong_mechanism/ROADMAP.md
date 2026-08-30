# Strong Mechanism Paper Roadmap

## M0 — Infrastructure and claim freeze

Deliverables:

- branch and paper directory;
- frozen claim ledger;
- source data/code manifest;
- prospective protocols P1–P6;
- schedule validator, command renderer, aggregation scripts, tests;
- evidence-gated LaTeX skeleton.

Exit gate: `make validate` passes and no new manuscript claim exceeds `CLAIMS.md`.

## M1 — Equal-dose temporal-distribution qualification

1. validate exact discrete schedule sums;
2. dry-run every arm and seed;
3. verify identical model, data, target manifest, total steps, checkpoint positions, and generation settings;
4. freeze config bytes and command manifest;
5. run a one-step smoke test outside the formal seed set;
6. authorize formal seeds only after smoke integrity passes.

Exit gate: signed/frozen M1 registry with no result access.

## M2 — Qwen3-4B three-seed mechanism experiment

Principal arms:

- `front_loaded_equal_dose`;
- `uniform_equal_dose`;
- `back_loaded_equal_dose`.

Historical context arms:

- `natural_diverse_10`;
- `natural_constant20`.

Required checkpoints: 0, 6, 16, 32, 48, 64.

Required outputs:

- adapter/checkpoint index;
- exact anchor-draw and exposure manifests;
- safety100 and utility100 generations;
- gradient-contribution audit at 32/48/64;
- environment and file hashes.

Exit gate: all three seeds complete or the study stops and reports the missingness mechanism without replacement.

## M3 — Final quality remeasurement

- freeze rubric and packet construction before arm-key join;
- evaluate the same prompt IDs and decoding setup for every principal arm/seed;
- A/B reviewers must pass calibration, duplicates, and component reliability;
- C adjudicates only registered disagreements;
- preserve prompt-level labels for paired statistics.

Exit gate: reliable arm-blind quality dataset and integrity report.

## M4 — Human-validity subset

- select the subset from hashes before model-review outcomes are opened;
- two independent humans plus a third adjudicator;
- compare component labels, ordinal score, and arm preference with model-review labels;
- report disagreement taxonomy.

Exit gate: registered minimum reliability or an explicit measurement limitation that lowers C4.

## M5 — Second setting

Preferred order:

1. second backbone with the same risk and safety sources;
2. if infeasible, same backbone with a prospectively selected second risk source.

Run only the minimum mechanism conjunction, not the full historical exploration.

Exit gate: replication, bounded non-replication, or model-specific scope decision.

## M6 — Analysis freeze and prompt-disjoint confirmation

Before opening confirmation:

- freeze manuscript hypotheses;
- freeze exclusion and missing-data rules;
- freeze all analysis scripts and figure/table code;
- freeze checkpoint and seed aggregation rules;
- record a commit hash and file hashes.

Then open the paper-specific prompt-disjoint panel exactly once.

## M7 — Submission release

- no pending primary result placeholders;
- full validation suite passes;
- all figures generated from normalized data;
- all tables generated from normalized data;
- LaTeX compiles without undefined references or citations;
- claim audit and simulated reviewer attack audit signed off;
- artifact manifest produced.
