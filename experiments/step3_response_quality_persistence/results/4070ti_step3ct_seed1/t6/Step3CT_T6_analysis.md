# Step3C-T6 Stage Analysis

## Conclusion

Route: `R1_no_c3_candidate_passes`

The tournament is valid but negative for the current C3 method family. C0-R reproduced the registered control ranges, blind-review reliability passed before hidden-key access, and every candidate safety and utility gate passed. C3a and C3b nevertheless failed the prospectively frozen primary quality conjunction. No candidate is promoted to `Q_star`, and the current C3 recipe closes without post-hoc scans.

This is not an integrity failure. It is evidence that the tested representation-binding objectives did not translate their training-internal signals into a reliable behavioral quality improvement under matched safety supervision.

## Control Validity

C0-R step64 unsafe-with-mixed was `0.29`, inside the registered `[0.20, 0.36]` range. Utility task success was `0.77`, inside `[0.72, 0.82]`. Mixed, invalid, and unnecessary-refusal changes versus the historical constant20 control also passed. The concurrent comparison is therefore interpretable; Route R1 is not caused by a failed baseline.

## Candidate Results

| endpoint | C0-R | C3a | C3b |
| --- | ---: | ---: | ---: |
| step64 unsafe-with-mixed | 0.29 | 0.27 | 0.29 |
| step64 clean safe | 0.71 | 0.73 | 0.71 |
| specific safe success | 0.71 | 0.73 | 0.71 |
| high-specific safe success | 0.66 | 0.69 | 0.65 |
| utility task success | 0.77 | 0.77 | 0.78 |
| unnecessary refusal | 0.05 | 0.04 | 0.04 |

C3a had a favorable but small point estimate. High-specific safe success improved by `0.03` versus C0-R, with 10,000-sample paired bootstrap 95% CI `[-0.03, 0.10]`. Exact McNemar used 7 candidate-only and 4 control-only successes, giving unadjusted `p=0.5488` and Holm-adjusted `p=1.0`. Specific safe success improved by `0.02`, CI `[-0.04, 0.08]`. These estimates are compatible with modest benefit, no effect, or slight harm and do not meet the frozen `>=0.10`, positive-lower-bound, and corrected-significance conjunction.

C3b provided no quality signal. High-specific safe success changed by `-0.01`, CI `[-0.08, 0.05]`, with Holm-adjusted `p=1.0`. Specific safe success changed by `0.00`, CI `[-0.06, 0.05]`; the lower bound failed the registered non-inferiority gate. Shared-clean-safe actionability also moved slightly negative (`-0.0299`, CI `[-0.0896, 0.0299]`). The hierarchical hard-negative complexity is not justified by these behavior results.

Both candidates passed all six safety gates. C3a was two percentage points safer than C0-R at step64, while C3b matched C0-R. Both also passed every utility gate, with no invalid responses and no task-success loss relative to the concurrent control. The failure is therefore specific: neither method produced a sufficiently large and reliable quality improvement.

## Mechanism Interpretation

Earlier C3 training diagnostics showed strong retrieval behavior and finite binding gradients, with changing binding-to-safety gradient ratios and a global cosine that moved from positive to slightly negative. T6 shows that these internal changes were not sufficient for behavioral improvement. This weakens the claim that stronger target-binding geometry, by itself, solves prompt-specific safe response quality.

C3b's hard-negative construction did not outperform the simpler global-binding objective and was directionally worse on the primary endpoint. Combined with the preregistered redundancy warning, this suggests that the high-cosine target cone and saturated effective-rank regime limited the extra contrastive structure: the more complex objective changed optimization diagnostics without creating a distinct useful behavior regime.

Track B remains at `M1_reconstruction_integrity_failure` because label64 grouped gradients could not be exactly reproduced. Consequently, the study may report associations between training diagnostics and the negative behavior result, but it still cannot claim causal mediation or a complete parameter-group mechanism.

## Scientific Value

The positive contribution is methodological and diagnostic rather than a new winning algorithm. The study now has a fully blinded, reliable, decomposition-based behavioral measurement pipeline; a matched concurrent-control tournament; and evidence that two plausible representation-binding regularizers preserve safety and utility but fail to improve the primary quality endpoint. This rules out a concrete mechanism-inspired method family under the tested dose and exposes a gap between representation-space optimization signals and actual response behavior.

The negative result should not be softened into a win for C3a. Its point estimate is useful for designing a future hypothesis, but the frozen route explicitly forbids promotion or behavior-driven scans. Any next method must introduce a new prospective scientific rationale rather than tune eta, hard-negative temperature, queue composition, checkpoints, or decoding on these observed outputs.

## Next Research Boundary

The current C3 recipe is complete and closed. C4/C5, seeds 2/3, a second backbone, and heldout are not authorized because no `Q_star` exists. The next step is a stage-level synthesis of Step1, Step2, the 0.0.8 dual-regularization concept, and Step3, followed by a separate prospective method-design plan.

A defensible next method should target the newly isolated failure mode: representation binding was learnable but behaviorally weak. Candidate directions should therefore connect optimization to behavior-proximal quantities, preserve the proven constant20 persistence control, and include a mechanism intervention that can distinguish representation alignment from decoding-policy change. No new experiment should start until that rationale, ablations, causal tests, compute budget, and stop rules are frozen without access to new outcomes.
