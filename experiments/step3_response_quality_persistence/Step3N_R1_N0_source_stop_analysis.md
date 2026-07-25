# Step3N-R1 N0 Source Qualification Stop Analysis

## Decision

N0-R1 stopped at the fresh source-review reliability gate. PCSR-M and
PCSR-MW did not reach generic-safe writing, prompt-mismatch matching, model
implementation, or training.

This is a measurement failure, not evidence that the PCSR objective would or
would not improve generated behavior.

## Frozen Evidence

Two independent no-context reviewers scored 87 primary responses plus eight
hidden duplicates. A third no-context reviewer adjudicated all 60 primary rows
with at least one A/B field disagreement. A/B and C files were separately
hashed, committed, and pushed before aggregate statistics were computed.

| construct | exact agreement | weighted agreement | Gwet | gate |
| --- | ---: | ---: | ---: | --- |
| detail retention | 0.793 | 0.897 | AC2 0.846 | pass |
| safe actionability | 0.379 | 0.690 | AC2 0.371 | fail (`>=0.70`) |
| naturalness | 1.000 | NA | AC1 1.000 | pass |
| clean safety | 1.000 | NA | AC1 1.000 | pass |

Both complete reviewers reproduced all four decision fields on `8/8` hidden
duplicates. Naturalness and clean safety were one-class outcomes: positive
agreement was 1.0, negative agreement was undefined, and pooled yes prevalence
was 1.0.

The actionability confusion was systematic:

```text
Reviewer A: 0=1, 1=18, 2=68
Reviewer B: 0=0, 1=72, 2=15
A/B pairs:  (0,1)=1, (1,1)=18, (2,1)=53, (2,2)=15
```

There were no severe `0` versus `2` disagreements. Instead, the reviewers used
the boundary between "relevant safe direction" (`1`) and "concrete safe next
step" (`2`) differently on 53 responses. This is a scale-calibration failure
on the exact threshold that defines a high-specific positive.

## Coverage Is Not A Rescue

Reviewer C's frozen adjudication yielded 68 qualifying responses across 37
prompts, one prompt above the required minimum. Five multi-prompt categories
were represented. Those descriptive counts show that suitable text probably
exists, but adjudication cannot replace the failed independent reliability
gate. Selecting the 37 prompts would make inclusion depend on a reviewer-scale
choice that A and B could not reproduce.

The following are prohibited:

- lower the actionability threshold from `2` to `1`;
- replace AC2 with weighted agreement after seeing the result;
- treat C as a tie-breaking gold standard and ignore A/B reliability;
- revise the rubric and rerun the same 87 responses until it passes;
- release the descriptively eligible 37 prompts for `yG` construction;
- continue to matching or GPU training.

## Scientific Meaning

The repaired PCSR mathematics remains coherent and the exact matching solver
is technically valid. What failed is the construct-validity bridge from frozen
responses to the claimed high-specific, actionable behavior. Training despite
that failure would weaken the paper more than a negative result does, because
the method's supervision would be selected by a non-reproducible boundary.

The result narrows the research problem: prompt detail is measurable here,
while "concrete safe next step" is not stable across independent model-review
sessions without additional calibration. This also cautions against treating
small changes in the existing high-specific endpoint as precise mechanism
evidence.

## Next Legitimate Routes

There are two scientifically defensible options, and neither reopens R1.

1. Run a separate measurement-development study on an external calibration
   corpus, with anchored examples for the `1/2` actionability boundary and an
   untouched validation corpus. Only a prospective R2 protocol may then apply
   the frozen revised instrument once to these training responses.
2. Design a new method whose training-data gate uses objective prompt-response
   correspondence and clean safety without selecting targets through the
   failed `actionability=2` boundary. Its behavioral endpoint still requires a
   separately validated actionability instrument before a quality claim.

The second route better preserves method momentum, but it must not disguise
the old 37 adjudicated prompts as newly reliable supervision. A new design
audit should explicitly compare novelty, causal identifiability, data needs,
and whether it merely repeats the failed C3 prompt-binding family.
