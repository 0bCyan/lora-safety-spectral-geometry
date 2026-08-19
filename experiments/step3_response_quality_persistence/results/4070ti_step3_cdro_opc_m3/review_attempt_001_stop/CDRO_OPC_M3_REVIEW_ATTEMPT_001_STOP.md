# CDR-OPC M3 Review Attempt 001 Stop

The frozen A/B review route selected `STOP_M3_REVIEW_UNRELIABLE`. Both new
context-isolated reviewers completed 156 rows and reproduced all 12 hidden
primary duplicates exactly. On the 120 primary candidates, full-vector exact
agreement was `0.9667`; derived qualification raw agreement was `0.9667`, Gwet
AC1 was `0.9655`, and the prompt-cluster bootstrap AC1 lower 2.5% quantile was
`0.9094`. There were four derived qualification disagreements.

Those strong primary agreement values do not override the registered
calibration conjunction. Reviewer A failed the frozen safe-route and
prompt-binding sentinel boundaries. Reviewer B failed those boundaries and
the clean-safety sentinel boundary. Both reviewers were perfect on hidden
duplicates, so the failure is not random response instability; it is a
systematic mismatch at the old sentinel negative-class boundary.

The primary candidate pool was also nearly one-class: A derived 116 yes and
four no labels, while B derived 120 yes and zero no labels. The registered
minimum negative-rating and negative-agreement gates therefore failed. This
gate is scientifically meaningful for estimating two-class reliability, but
it cannot be inferred from a deliberately high-yield positive candidate pool.

No Reviewer C was launched because the frozen route forbids arbitration when
A/B calibration or reliability fails. No individual label was accepted by the
main executor, no forty-prompt cohort was selected, and no candidate response
or review text was committed. The immutable repository-external review hashes
are:

```text
reviewer_a=ec134a5633ea0744617fcf2fc6827428c6bff8ba9eb6b7c580ff57b1e9f213bb
reviewer_b=b729d598bea29b680a1a14be933be9422cf6e0c966ed71ee78d3995fff8f35c5
ab_outcome=6a39206542914fd2f2ab61546dd357951e71e4418832eb05ba3d16e225575113
```

## Interpretation and repair boundary

This is a measurement failure, not evidence for or against CDR-OPC or the
conformal-dose controller. A valid recovery must not reinterpret the 116 A/B
agreements as qualified targets. It may freeze a cohort using only the
already-registered seed and opaque prompt IDs, independently of every A/B
label, and evaluate that cohort with new reviewers plus an explicitly
two-class, unambiguous anchor set. The failed A/B labels remain development
evidence only and may not select prompts, candidates, thresholds, or gates.

```text
reviewer_c_launched=false
cohort_selected=false
gradient_measurement_started=false
optimizer_steps=0
training_authorized=false
core100_opened=false
heldout_opened=false
```
