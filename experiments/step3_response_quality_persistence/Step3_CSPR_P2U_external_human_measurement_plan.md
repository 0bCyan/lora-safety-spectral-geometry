# Step3 CSPR P2U External Human Measurement Plan

## 1. Decision

CSPR P2T ended at commit `04f93276ac6ebc6f71b1e028f966724f40f454ee`
with route `STOP_CSPR_P2T_CALIBRATION_UNRELIABLE`. All four automated
review files were complete and provenance-valid, but the registered
calibration and reference-concordance gates failed. In particular, the
Actionability target labels collapsed to score 2 on every primary row, so
negative agreement was undefined. This is a measurement failure, not evidence
for or against the CSPR optimization method.

P2U is a new prospective measurement study using qualified external human
reviewers. It does not retry P2T, reuse P2T labels, change a rubric, revise a
threshold, inspect target yield, or authorize training. Automated model
reviewers and sub-agents are not eligible for P2U labels.

## 2. Frozen Scientific Question

P2U asks whether the unchanged Step3O component rubric and clean-safety rubric
can be applied reproducibly by an independent human panel to the unchanged
P2S rows.

The experiment separates two hypotheses:

- `H_measurement_fail`: independent human reviewers do not pass the frozen
  calibration gates, so this instrument cannot support target qualification.
- `H_automated_transport_fail`: independent human reviewers pass the frozen
  calibration gates, so P2T's failure is attributable to the automated
  reviewer channel under the tested conditions rather than to a demonstrated
  defect in the CSPR candidate pool.

Passing calibration is necessary but not sufficient for a CSPR result. Target
reliability, disagreement adjudication, fixed-pool yield, and a separately
frozen zero-update mechanism study would still be required.

## 3. Unchanged Inputs

P2U binds the P2S packet manifest and all eight packet/template files, both
hidden keys, both rubrics, all metric thresholds, the 10,000-replicate
bootstrap settings, duplicate allocation, target pool, and pair-yield gates.
The original files remain immutable.

The P2U preparation tool creates calibration-only views as exact ordered byte
subsequences of the original reviewer-specific files:

| construct | primary anchors | hidden anchor duplicates | rows per reviewer |
| --- | ---: | ---: | ---: |
| Safety | 40 | 4 | 44 |
| Actionability | 60 | 6 | 66 |

No prompt or response is redacted, sanitized, paraphrased, regenerated, or
reserialized. Target rows are not written during P2U P0.

## 4. Reviewer Eligibility and Welfare

P2U requires four distinct adult human reviewers: Safety A/B and
Actionability A/B. A reviewer must be fluent in the packet language, have
documented safety-annotation or adjacent professional experience, provide
informed consent to potentially disturbing material, and have an available
opt-out and support process. Reviewers must not be study authors and must not
have accessed CSPR generations, hidden keys, old labels, aggregate results,
training results, gradient reports, arm identities, or another reviewer's
work.

An accountable external review coordinator must verify eligibility and local
ethics, labor, privacy, and data-handling requirements. The coordinator may
hold real identities outside the repository; the repository stores only
pseudonymous reviewer IDs and signed attestations. No packet access is allowed
before those attestations and the P2U execution protocol are frozen and
pushed.

There is one registered A/B panel per construct. Reviewer replacement,
threshold shopping, repeated qualification, and selective row omission are
forbidden after packet access. A welfare withdrawal stops the affected
construct as inconclusive and does not authorize a replacement in this
attempt.

## 5. P0: Protocol and Calibration-View Freeze

P0 verifies the P2T terminal route, all frozen hashes, exact row counts, packet
schemas, the absence of P2T Reviewer C artifacts, and the zero-training
boundary. It then writes only the four 44/66-row calibration packet/template
pairs plus a manifest. The generated rows must be exact byte subsequences of
their frozen parent files and may contain only `calibration_anchor` rows.

The only P0 pass route is
`P2U0_CALIBRATION_VIEWS_FROZEN_REVIEWER_QUALIFICATION_ONLY`. It authorizes
external reviewer recruitment and zero-access qualification only.

## 6. P1: Zero-Access Human Qualification

Before packet access, the external coordinator records for each role:

- a unique pseudonymous reviewer ID;
- adult-status, language, and relevant-experience attestations;
- independence and conflict-of-interest attestations;
- informed-consent and welfare-process attestations;
- zero access to every forbidden artifact;
- commitment to one complete independent calibration attempt.

P1 passes only if all four reviewers qualify and all IDs are distinct. P1 and
the hash-bound calibration execution protocol must be committed and pushed
before any reviewer receives a packet. Failure selects
`STOP_CSPR_P2U_HUMAN_QUALIFICATION`.

## 7. P2: Calibration-Only Review

Each reviewer receives only the role-specific calibration packet, blank
template, unchanged rubric, and neutral completion instructions. Reviewers
work independently. Complete outputs and provenance are frozen before the
main session opens a hidden key or compares labels.

The original P2S calibration gates apply unchanged. This includes A/B raw and
chance-corrected agreement, positive and negative agreement, per-reviewer
concordance to the frozen reference labels, Actionability ordinal and boundary
metrics, prevalence informativeness, and the frozen bootstrap lower bounds.
P2U adds stricter calibration-subset duplicate requirements of 4/4 for Safety
and 6/6 for Actionability. These additions do not relax any inherited gate.

If A/B reliability passes but concordance to the legacy frozen reference
fails, the route remains a stop and is reported specifically as
`STOP_CSPR_P2U_REFERENCE_NONTRANSPORT`. P2U does not silently replace a frozen
reference after seeing labels.

Any missing, malformed, reordered, policy-blocked, withdrawn, or
unprovenanced review selects `STOP_CSPR_P2U_REVIEW_INCONCLUSIVE`. Any numerical
gate failure selects `STOP_CSPR_P2U_CALIBRATION_UNRELIABLE`. No target packet
is created on either route.

## 8. Why the Boundary Gates Are Necessary

For binary ratings, positive and negative agreement are

\[
P_+ = \frac{2n_{11}}{2n_{11}+n_{10}+n_{01}}, \qquad
P_- = \frac{2n_{00}}{2n_{00}+n_{10}+n_{01}}.
\]

If every target is rated positive by both reviewers, `P_+=1` but the
denominator of `P_-` is zero. The data then contain no observed negative
boundary case and cannot establish reliable discrimination on both sides.
Treating undefined `P_-` as a pass would convert absence of information into
evidence, so P2U retains the registered fail-on-undefined rule.

Likewise, A/B agreement alone does not prove validity: two reviewers can agree
on the same shifted interpretation. The balanced frozen references test
concordance with the pre-existing construct, while hidden duplicates test
within-reviewer consistency. The conjunction is deliberately stronger than
any one statistic.

The 10,000-replicate row bootstrap estimates sampling uncertainty for the
finite calibration set. It does not prove population-wide construct validity,
and all comparisons use unrounded values.

## 9. P3: Conditional Target Review

Only a complete P2 pass may authorize a separately committed P3 protocol that
creates target-only views as exact ordered byte subsequences of the same P2S
packets. The same qualified human reviewers then label their target views.

The unchanged target gates apply. Combining the strict calibration duplicate
requirements with target requirements preserves the original aggregate
duplicate gates: Safety target duplicates must pass at least 19/20, and
Actionability target duplicates at least 5/6, yielding at least 23/24 and
11/12 overall respectively.

Target score prevalence remains descriptive. Undefined negative agreement or
another zero-variance boundary selects `STOP_CSPR_P2U_TARGET_UNINFORMATIVE`,
not a favorable result.

## 10. P4-P6 Conditional Stages

Reviewer C is a new qualified human per construct and is authorized only after
all calibration, target, bootstrap, and duplicate gates pass. C sees only
target disagreement rows and neither A/B labels nor reasons. C cannot rescue a
failed A/B reliability gate.

Only complete adjudication permits fixed-pool qualification under the unchanged
P2S pair and yield rules. A yield pass can authorize only a separately frozen
CSPR zero-update mechanism protocol. That study must verify the P0 projection
theorem numerically on the real model, include label-swap and wrong-prompt
controls, and perform no optimizer step. Training requires another prospective
decision after those mechanism results.

## 11. Route and Claim Boundary

P2U cannot establish a behavioral gain, a causal mechanism, generalization,
or paper-ready construct validity by itself. It can establish only
rubric-conditional human inter-rater reliability on the frozen calibration and
target designs. A failed P2U closes this measurement route. A passed P2U keeps
CSPR eligible for later zero-update mechanism validation.

Throughout P2U, core100 and heldout remain closed,
`optimizer_steps=0`, and `training_authorized=false`.
