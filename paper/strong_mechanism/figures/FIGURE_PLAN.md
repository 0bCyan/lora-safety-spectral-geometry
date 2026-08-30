# Main Figure Plan

## Available now: design and frozen historical evidence

### Figure 0 — Paper story

File: `available/fig00_paper_story.svg`

- diagnostic geometry;
- behavior-aligned sequence supervision;
- persistence–quality tension;
- equal-dose confirmatory mechanism program;
- explicit statement that formal evidence remains pending.

### Figure D1 — Equal-dose temporal-distribution design

File: `available/fig01_equal_dose_design.svg`

- panel A: front-loaded, uniform, and back-loaded safety weights;
- panel B: exact equality of cumulative safety and risk coefficients;
- message: the registered principal contrast changes timing, not total dose.

### Figure D2 — Evidence pipeline and stopping gates

File: `available/fig02_evidence_pipeline.svg`

- M0–M7 sequence;
- no result-dependent schedule changes;
- no heuristic substitution;
- no seed × prompt pseudoreplication;
- no confirmation before analysis freeze.

### Figure D3 — Data roles and leakage boundaries

File: `available/fig03_data_boundaries.svg`

- safety training targets;
- Core100 development evaluation;
- Paper-A prompt-disjoint confirmation;
- global heldout reserved for the future method paper;
- benign utility as a separate stream.

### Historical quantitative panels

- `available/fig10_historical_subspace_signature.svg`;
- `available/fig11_historical_posthoc_control_failure.svg`;
- `available/fig12_historical_sequence_anchor_multiseed.svg`;
- `available/fig13_historical_seed1_persistence_context.svg`.

These panels are frozen historical context. They do not substitute for the registered final replication.

## Pending formal result figures

### Figure 2 — Equal-dose safety trajectories

- unsafe rate at steps 0/6/16/32/48/64;
- principal temporal arms, with seed range and mean;
- historical `decay10/constant20` shown only as clearly separated context;
- requires `data/derived/run_metrics.csv`.

### Figure 3 — Seed-level paired effects

- prompt-paired step64 effects for back-loaded versus front-loaded and uniform versus front-loaded;
- one interval per seed plus seed-hierarchical pooled estimate in the table;
- requires `data/derived/paired_effects.csv`.

### Figure 4 — Late effective gradient contribution

- weighted safety/risk contribution at steps 32/48/64;
- all registered checkpoints shown;
- no favorable-checkpoint selection;
- requires `data/derived/gradient_summary.csv`.

### Figure 5 — Safety–quality frontier

- x: step64 unsafe rate;
- y: unconditional prompt-aware derived quality;
- conditional-safe quality only as sensitivity analysis;
- utility shown separately;
- requires `run_metrics.csv` and `quality_metrics.csv`.

## Appendix figures after corresponding evidence exists

- anchor exposure balance;
- run-order and technical-failure audit;
- component-level reviewer agreement;
- model–human agreement;
- response-length and truncation sensitivity;
- second-setting replication;
- confirmatory-panel effects.

## Generation rules

- current legal assets: `plot_available_figures.py`;
- formal quantitative assets: `plot_main_figures.py`;
- no final-result input means no final-result figure;
- all values must be source-driven and reproducible.
