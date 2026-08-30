# P5 — Second-Setting Replication

Status: design-ready; exact setting must be frozen before any result is inspected.

## Preferred setting

A second 3B–8B instruction-tuned backbone compatible with the existing QLoRA pipeline, using the same canonical risk source and natural safety targets where tokenizer and license constraints permit.

## Fallback setting

The same Qwen3-4B backbone with a prospectively selected second risk fine-tuning source that is prompt-disjoint from the Paper A confirmation set.

The choice between preferred and fallback must be justified by feasibility, not by observed results.

## Minimum experiment

1. vanilla risk trajectory at checkpoints 0/16/32/64;
2. one preselected geometry diagnostic;
3. one preselected geometry-control negative baseline;
4. sequence-supervised final arm;
5. equal-dose `front_loaded` and either `uniform` or `back_loaded` arm;
6. final frozen safety and quality evaluation.

## Minimum mechanism conjunction

- risk adaptation produces a measurable unsafe transition;
- sequence supervision improves behavior relative to the risk arm;
- temporal-distribution effect has the registered direction or is reported as a bounded non-replication;
- quality measurements are computed with the same rubric;
- all divergences in tokenizer, target truncation, or step count are disclosed.

## Interpretation

A non-replication is not repaired by switching models. It establishes model/data specificity and sets the final claim ceiling.
