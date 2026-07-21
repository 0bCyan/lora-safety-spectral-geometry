# Step3C Stage 0A Protocol Freeze Report

Status: frozen before model generation, new positive writing, or reviewer labeling.

## Boundary Validation

| check | value |
| --- | ---: |
| safety_train IDs | 40 |
| natural-v1 target rows | 80 |
| targets per prompt | 2 |
| core100 overlap | 0 |
| proxy_dev overlap | 0 |
| heldout overlap | 0 |
| prompts with existing non-mechanical positive | 33 |
| prompts requiring targeted positive | 7 |

Prompts requiring exactly one prospective positive after this freeze:

```text
eval_safety_000101, eval_safety_000113, eval_safety_000116, eval_safety_000156, eval_safety_000253, eval_safety_000265, eval_safety_000295
```

## Frozen Protocol

- Triplet schema: `step3c_stage0_triplet_schema.json`
- Review protocol: `step3c_stage0_review_protocol.json`
- Behavior rubric: `review_rubrics/step3c_triplet_behavior_rubric.md`
- Structure codebook: `review_rubrics/step3c_triplet_structure_codebook.md`

No optimizer steps, model generation, target additions, core100 triplet use, heldout access, or training are authorized by this freeze.
