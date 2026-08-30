# P0 — Claim Freeze

Status: prospective paper-governance protocol.

## Primary research question

Does the temporal distribution and late effective contribution of sequence-level safety supervision influence safety persistence under risk LoRA adaptation, after cumulative supervision dose is held fixed, and what safe-response-quality trade-off accompanies persistent supervision?

## Confirmatory hypotheses

### H1: temporal persistence

At equal cumulative anchor weight, `back_loaded_equal_dose` will have lower step64 unsafe rate and lower step32-to-step64 rebound than `front_loaded_equal_dose`.

### H2: uniform persistence

At equal cumulative anchor weight, `uniform_equal_dose` will have lower step64 unsafe rate and lower rebound than `front_loaded_equal_dose`.

### H3: late contribution

At steps 48 and 64, the weighted safety/risk gradient-contribution ratio will be ordered:

```text
back_loaded > uniform > front_loaded
```

The ratio is a preregistered mechanism measurement, not a proven mediator unless additional mediation assumptions are justified.

### H4: quality bottleneck

Arms with stronger late safety persistence may improve safety without passing all frozen prompt binding, actionability, detail, naturalness, generic-refusal, and utility criteria.

## Primary endpoints

1. prompt-level unsafe including mixed behavior at step64;
2. step32-to-step64 unsafe rebound;
3. weighted safety/risk gradient-contribution ratio at steps 48 and 64.

## Secondary endpoints

- step16 and step32 unsafe;
- clean refusal and mixed behavior;
- prompt binding;
- safe actionability;
- detail retention;
- naturalness;
- generic-only refusal;
- task success, unnecessary refusal, invalid output, and material degradation.

## Analysis population

All registered seeds and prompts that pass pre-result integrity checks. A run is not removed for unfavorable behavior. Technical missingness is reported by arm and seed; replacement runs are forbidden unless the failure occurred before model initialization and the replacement rule was frozen in advance.

## Claim boundary

Paper A is a mechanism and empirical study. It does not present a new validated training method. Any future optimization design is discussed as a consequence of the measured constraints, not as a result of this protocol.
