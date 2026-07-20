# Step3C Candidate-Data Feasibility

## Result

| metric | value |
| --- | ---: |
| safety-training prompts | 40 |
| prompts with two non-mechanical targets | 26 |
| prompts with one non-mechanical target | 7 |
| prompts with no non-mechanical target | 7 |
| existing positive coverage | 0.825 |

The preliminary C2 positive-coverage gate (`>=0.80`) passes. Seven prompts need one targeted non-mechanical positive. This does not authorize writing those targets or training C2 until the triplet rubric and Step3B route are frozen.

C3 can use the existing independently approved safe targets for all 40 prompts, subject to representation and queue feasibility.
