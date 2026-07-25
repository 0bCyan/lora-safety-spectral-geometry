# Step3C-T5R4 Review Reliability Report

## Decision

Route: `T5R4_complete_labels_authorize_T6`

All prespecified Safety, Quality, Utility, hidden-duplicate, and Reviewer C coverage gates passed. The reliability analysis executed from H5 commit `95f822e551400b8537d61eec6639ae37e1c3292c` with:

```text
python experiments/step3_response_quality_persistence/scripts/run_step3ct_t5r_review.py finalize
```

Reliability was computed before adjudicated labels were written and before any hidden outcome-key join. Heldout remained sealed and no new training was authorized.

## Review Coverage

- Safety: 600 primary rows, 30 hidden duplicate pairs, 11 disagreement rows, and 11/11 Reviewer C rows.
- Quality: 300 primary rows, 30 hidden duplicate pairs, 69 disagreement rows, and 69/69 Reviewer C rows.
- Utility: 300 primary rows, 30 hidden duplicate pairs, 33 disagreement rows, and 33/33 Reviewer C rows.
- Every Reviewer A/B hidden-duplicate full vector consistency was `1.0`.
- All 113 disagreement rows and all 156 disputed decision fields were covered before adjudication.

## Reliability Results

Safety agreement was strong. Unsafe-with-mixed reached raw agreement `0.9833` and Gwet AC1 `0.9756`; mixed reached `0.9883` and `0.9880`; prompt-relevant redirect reached `0.9883` and `0.9830`. Invalid was a zero-variance field with raw agreement `1.0`, so chance-corrected agreement is correctly recorded as undefined.

Quality detail retention reached linear weighted agreement `0.9733`, Gwet AC2 `0.9628`, and zero two-level disagreements. Safe actionability was the weakest registered measure but still passed prospectively frozen gates: linear weighted agreement `0.8983`, Gwet AC2 `0.8031`, and two-level disagreement rate `0.02`. Derived eligibility reached raw agreement `0.95` and Gwet AC1 `0.9177`; naturalness raw agreement was `0.9933`.

Utility task success reached raw agreement `0.95` and Gwet AC1 `0.9245`; unnecessary refusal reached `0.99` and `0.9892`; material degradation reached `0.94` and `0.8985`. Invalid-or-irrelevant was zero variance with raw agreement `1.0`. The unregistered diagnostic field format failure also showed raw agreement `0.9867` and Gwet AC1 `0.9862`.

## Interpretation Boundary

This is a positive measurement result: the blind-review labels are reliable enough for the prespecified tournament statistics, and complete adjudicated Safety, Quality, and Utility labels now exist. It is not yet evidence that C3a or C3b improves behavior, wins the tournament, or validates a causal mechanism. Those claims depend on the hidden-key join and paired T6 comparisons.

The main residual measurement caution is adjacent-category ambiguity in safe actionability. Its weighted and chance-corrected agreement passed with useful margin, while exact raw agreement was `0.8167`; therefore T6 should retain ordinal and threshold-derived reporting and must not present the construct as perfectly objective.

## Next Authorized Action

Commit and push this T5R4 package, then run the unchanged local CPU T6 statistics path. T6 may join only the frozen adjudicated labels to the frozen hidden keys, use prompt ID as the paired unit, run 10,000 bootstrap resamples with the registered seed, apply exact McNemar tests and Holm correction, and select only a preregistered route. New training, heldout, extra seeds, a second backbone, and method revision remain unauthorized.
