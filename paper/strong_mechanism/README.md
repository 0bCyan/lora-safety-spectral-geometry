# Paper A — Strong Mechanism Study of LoRA Safety Persistence

This directory is the single paper-facing workspace for the mechanism paper.
It is deliberately separated from the exploratory Step 3 method branches.

## Scientific target

The paper studies the following chain:

1. risk LoRA fine-tuning produces reproducible safety drift and diagnostic geometric structure;
2. changing geometry or local scalar proxies is not sufficient to control behavior in the tested setting;
3. sequence-level safety supervision is behaviorally active;
4. the temporal distribution and late effective contribution of that supervision may govern persistence;
5. persistent supervision may create a prompt-relevance and safe-response-quality bottleneck.

The paper does **not** claim that CDR-OPC, CSPR, SpectraSafe, or any other candidate is a validated method. Method-oriented probes may appear only as future-design motivation or appendix material unless they produce real trained checkpoints, behavioral comparisons, multi-seed replication, and a sealed confirmation.

## Completion standard

The target is a strong mechanism paper, not a narrow case report. Submission readiness requires all of the following:

- equal-cumulative-dose temporal-distribution experiment;
- complete three-seed final replication;
- frozen prompt-aware quality remeasurement;
- independent human-validity subset;
- one second model or second risk setting;
- one prompt-disjoint confirmatory panel opened only after analysis freeze;
- reproducible statistics, figures, tables, and compiling LaTeX;
- an explicit claim-to-evidence ledger with no unsupported promotion.

Until those gates pass, `submission_ready` remains false.

## Directory map

```text
paper/strong_mechanism/
  CLAIMS.md                 claim freeze and promotion rules
  EVIDENCE_MAP.md           claim-to-artifact ledger
  ROADMAP.md                ordered completion plan
  DATA_AND_CODE_MAP.md      canonical source paths
  configs/                  new paper-specific experiment registries
  protocols/                prospective study protocols
  data/                     immutable source manifest and derived-data contract
  scripts/                  validation, aggregation, table, and figure code
  tests/                    invariance and smoke tests
  figures/                  figure specifications and generated outputs
  tables/                   table specifications and generated outputs
  latex/                    complete evidence-gated manuscript
  verification/             operations, release gates, reviewer attack checks
```

Canonical historical code and data are not duplicated. They remain at their existing repository paths and are pinned by `data/source_manifest.json`. Paper-specific derived data must be written below `paper/strong_mechanism/data/derived/`.

## Required order of work

```text
P0 claim freeze
→ P1 equal-dose temporal distribution
→ P2 multi-seed replication
→ P3 final quality remeasurement
→ P4 human validation
→ P5 second setting
→ analysis and manuscript freeze
→ P6 confirmatory panel
→ release verification
```

No confirmatory panel may be opened before the preceding analysis and decision rules are frozen.

## Local commands

From the repository root:

```bash
make -C paper/strong_mechanism validate
make -C paper/strong_mechanism commands
make -C paper/strong_mechanism analysis
make -C paper/strong_mechanism figures
make -C paper/strong_mechanism tables
make -C paper/strong_mechanism paper
make -C paper/strong_mechanism all
```

The first command is intentionally runnable before GPU experiments. Later commands fail clearly when required result files are absent.

## Branch policy

Development branch:

```text
paper/strong-mechanism-safety-persistence
```

Do not rewrite this branch after a prospective protocol is frozen. Experimental outputs must be append-only. Any protocol correction must be a new versioned file that preserves the original bytes and explains the effect on claims.
