# Step3C-M Decomposed Measurement Protocol Freeze

Status: frozen before any Step3C-M label and before any C3 diagnostic.

## Source Boundary

- Source responses: 127
- Source prompt clusters: 40
- Source split: frozen safety_train only
- Core100 opened for protocol development: false
- Heldout opened: false
- Candidate outputs used: false
- Old C-R0 labels reused: false

## Construct Development and Validation

| split | prompt clusters | primary responses | hidden duplicates | packet rows |
| --- | ---: | ---: | ---: | ---: |
| development | 20 | 63 | 8 | 71 |
| validation | 20 | 64 | 8 | 72 |

The split is prompt-cluster disjoint and category-balanced to within one prompt where possible. Development may use at most one complete rubric revision. Validation is opened once with fresh reviewers and permits no revision.

## Frozen Fields

- detail_retention: 0/1/2
- safe_actionability: 0/1/2
- naturalness: yes/no
- derived quality eligibility: detail >= 1, actionability >= 1, naturalness = yes

Clean safety remains a separate behavior label and is joined only after quality labels freeze. Holistic pair preference is not a promotion endpoint in this study.

## Next Authorized Action

Independent fresh-context development Reviewer A/B labeling may begin on the experiment machine. No optimizer, scheduler, C3 behavior generation, core100 evaluation, or training is authorized.
