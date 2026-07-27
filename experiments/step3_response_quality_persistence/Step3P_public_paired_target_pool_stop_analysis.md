# Step3P Public Paired Target-Pool Stop Analysis

## Decision

Step3P attempt 001 stopped at the frozen A/B review gate with route
`STOP_REVIEW_UNRELIABLE`. Reviewer C was not opened, hidden category and
severity yield were not inspected, the 48-target flow was not run, and no
optimizer step was authorized.

This is a measurement-transport failure for this attempt. It is not evidence
that the public responses are unsafe, that a 48-target pool cannot exist, or
that any proposed training method is effective or ineffective.

## Frozen Evidence

Two distinct context-isolated reviewer sessions each completed 92 primary
candidates plus 12 hidden duplicates. Provenance and byte hashes were
validated before aggregate statistics were written.

| frozen gate or diagnostic | observed | threshold | result |
| --- | ---: | ---: | --- |
| full-vector exact agreement | 0.739 (`68/92`) | `>=0.75` | fail |
| detail score-2 boundary negative agreement | 0.667 | `>=0.70` | fail |
| detail score-2 boundary negative ratings | 3 | `>=12` | fail |
| prompt-binding negative agreement | undefined | `>=0.70` | fail |
| prompt-binding negative ratings | 0 | `>=12` | fail |
| derived candidate-qualified raw agreement | 0.891 | `>=0.85` | pass |
| derived candidate-qualified Gwet AC1 | 0.792 | `>=0.75` | pass |
| candidate-qualified positive / negative agreement | 0.861 / 0.911 | `>=0.70` / `>=0.70` | pass |
| prompt-cluster bootstrap lower bound | 0.661 | `>=0.60` | pass |
| hidden-duplicate full-vector consistency | A `12/12`, B `12/12` | `>=11/12` | pass |

The remaining registered gates passed. In particular, detail exact agreement
was 0.989, `safe_route` raw agreement was 0.913, `executable_step` was 0.946,
naturalness was 0.935, and clean safety was 0.935.

## Why The Route Stopped

The route name must remain `STOP_REVIEW_UNRELIABLE`, but the evidence does not
look like careless or globally inconsistent review. Both reviewers reproduced
all hidden duplicates, and the final qualification conjunction passed every
one of its own agreement and prevalence gates.

The main problem is missing class support after transporting the O4 rubric to
an exactly prompt-paired, public `is_safe=true` source. Both reviewers marked
all 92 primary responses as prompt-bound, leaving zero negative ratings for a
gate that prospectively required at least 12. The strict detail boundary was
similarly almost one-class, with only three negative ratings across 184
ratings. Those fields therefore cannot demonstrate the required negative-class
reproducibility on this packet, even though their raw agreement is very high.

There is also a narrow but real complete-vector failure: 68 exact matches were
observed where 69 were required. It would be outcome-driven to waive that one
row, remove the negative-class requirements, or promote the passing derived
qualification score after opening the labels. Attempt 001 must stop under its
registered conjunction.

## Scientific Meaning

Step3P provides a useful negative design result. The O4 agreement contract was
validated on a broader external packet, but it does not transport unchanged to
a safe-only, exact-prompt-paired candidate packet. A reliability design that
requires both classes for every component needs prospectively supplied
calibration contrast; it cannot assume that naturally selected target
candidates will contain enough genuine prompt-mismatch or low-detail examples.

The passing candidate-qualified result is encouraging only as a diagnostic.
It suggests that the two fixed reviewer sessions can reproduce the ultimate
all-fields conjunction on this packet. It does not override the component
contract, establish human construct validity, authorize target selection, or
support a behavioral or mechanism claim.

## Closed Actions

The following actions are forbidden for attempt 001:

- relabeling, replacing, topping up, or resampling the 92 candidates;
- relaxing the five failed gates after seeing their values;
- opening Reviewer C or using adjudication to rescue the reliability failure;
- inspecting hidden category or severity yield or running the 48-target flow;
- importing the four successful O4R shards as study targets;
- running tokenizer, loss-mask, method-smoke, or GPU training work.

## Next Legitimate Route

The next step is prospective protocol design, not GPU execution. A defensible
successor should first separate reviewer calibration from candidate
qualification: freeze balanced external calibration items for every required
positive and negative class, keep an untouched transport-validation packet,
and specify how calibration evidence combines with primary-packet agreement
before any new labels are opened. Fresh heterogeneous or human review would
strengthen that bridge; another pair of same-model sessions would establish
only session-level reproducibility.

If a new target-generation route is still needed after that measurement gate
is repaired, the Step3P plan permits one separately frozen controlled
structured-API dispatcher. It must use a new prospective attempt, a complete
new evidence chain, and fixed no-top-up rules. It must not return to
agent-managed file delivery or reuse partial O4R targets. Only a passed target
pool, a separate training-suitability audit, and a later frozen method protocol
can authorize optimizer steps.
