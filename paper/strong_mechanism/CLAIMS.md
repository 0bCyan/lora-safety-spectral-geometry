# Claim Freeze and Promotion Rules

The paper has five claims. Each claim has a current tier and an explicit promotion gate. A result may be reported below its target tier, but it may not be worded above the highest supported tier.

## Evidence tiers

- **T0 — hypothesis:** motivated but not tested prospectively.
- **T1 — setting-bounded evidence:** supported in one model/data setting.
- **T2 — replicated mechanism evidence:** supported across seeds with the principal confound separated.
- **T3 — externally corroborated evidence:** supported in a second model or second risk setting and a prompt-disjoint confirmation.

## C1. Geometric structure is diagnostic, not sufficient behavioral control

### Frozen statement

In the tested Qwen3-4B risk-LoRA setting, unsafe drift is accompanied by reproducible geometric and output-side subspace structure, but changing the tested geometric quantities is not sufficient to produce behavioral safety.

### Current tier

T1.

### Existing evidence

- three risk seeds reach the unsafe endpoint;
- output-side singular subspace shows greater risk-seed agreement than risk-vs-control comparisons;
- global spectral averages weakly distinguish families;
- post-hoc parameter, singular-direction, activation, direct spectral, and factor-proxy controls fail to recover behavior.

### Prohibited wording

- geometry causes unsafe behavior;
- the output-side subspace is the unique causal mechanism;
- spectral regularization universally fails;
- the negative result generalizes beyond the tested model/data regime.

### Promotion gate

A second setting must reproduce both a diagnostic signal and at least one behaviorally null geometry-control comparison under a preselected protocol.

## C2. Sequence-level supervision is behaviorally active

### Frozen statement

Sequence-level safety supervision changes unsafe behavior substantially more than the tested parameter-geometric and local scalar proxies in the studied risk-LoRA setting.

### Current tier

T1, with three-seed early-checkpoint support.

### Existing evidence

- sequence-anchor step16 unsafe rates of approximately 0.27, 0.23, and 0.19 across three seeds;
- spectral and margin/proxy arms remain near the unsafe endpoint;
- factor-plus-sequence adds no measured safety beyond sequence supervision in the tested factorial.

### Prohibited wording

- sequence supervision is always necessary or sufficient;
- all scalar or representation proxies are invalid;
- sequence supervision alone is a complete safety method.

### Promotion gate

The final three-seed study must preserve the sequence-supervision effect at the final checkpoint with matched exposure and frozen evaluation.

## C3. Temporal distribution and late effective contribution influence persistence

### Frozen statement

At fixed cumulative safety dose, shifting supervision away from late training is hypothesized to increase late unsafe rebound; late weighted safety-gradient contribution is the preregistered mediator measurement.

### Current tier

T0/T1: the existing `constant20` versus `decay10` result supports persistence but is confounded by cumulative dose.

### Primary test

Equal-dose `front_loaded`, `uniform`, and `back_loaded` schedules across seeds 1–3.

### Promotion gate to T2

All conditions below must hold:

1. schedule dose and risk-weight sums match exactly under the trainer's discrete step semantics;
2. at least two of three seeds show the preregistered directional ordering on late rebound;
3. the seed-stratified pooled effect for `back_loaded` or `uniform` versus `front_loaded` exceeds the frozen smallest effect of interest;
4. late weighted safety/risk contribution follows the registered temporal ordering;
5. no arm fails data exposure, checkpoint, generation, or evaluation integrity.

### Prohibited wording before promotion

- late gradients determine safety persistence;
- temporal placement is more important than total dose;
- the measured gradient ratio is a proven causal mediator.

## C4. Persistent safety supervision creates a safe-response-quality bottleneck

### Frozen statement

A supervision schedule can improve unsafe behavior while leaving prompt binding, safe actionability, detail retention, naturalness, or generic-refusal behavior below a complete-method standard.

### Current tier

T1, based on the seed1 `constant20` quality audit; exact semantic-cluster boundaries are not a primary endpoint.

### Primary endpoints

- safety correctness;
- prompt binding;
- safe actionability;
- detail retention;
- naturalness;
- generic-only refusal;
- utility non-inferiority.

### Promotion gate to T2

- one final rubric frozen before arm labels are joined;
- reliable blinded A/B review across all principal arms and three seeds;
- human-validation subset reaches the registered component reliability and calibration gates;
- quality disadvantage or trade-off reproduces in at least two seeds or the pooled paired analysis.

### Prohibited wording

- semantic mode collapse is proved by the old free-form family label;
- template concentration is universal;
- persistence necessarily causes quality degradation.

## C5. The mechanism generalizes beyond one model/data setting

### Frozen statement

The central mechanism is expected, but not yet shown, to reproduce in one independently selected second setting.

### Current tier

T0.

### Minimum conjunction

The second setting must show:

1. an unsafe trajectory under risk adaptation;
2. a behaviorally active sequence-supervision arm;
3. a directionally consistent equal-dose temporal-distribution effect or a clearly reported bounded non-replication;
4. the same frozen quality rubric and integrity checks.

### Prohibited wording before completion

Any cross-model or general LoRA claim.

## Paper-level claim ceiling

The title, abstract, introduction, and conclusion may not exceed the lowest tier among the claims they invoke. A strong final mechanism paper requires C1–C4 at least T2 and C5 at least T1, followed by one prompt-disjoint confirmatory analysis.
