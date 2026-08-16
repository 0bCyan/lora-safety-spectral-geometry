# CITA Method Audit and Registered Negative Result

## 1. Decision

The frozen route is `STOP_CITA_PROXY_DEV_EFFECT_GATE_FAILED`. CITA is rejected
as a positive primary method under the registered seed-1, step-16 development
setting. No threshold was changed, no generation arm was retried, and no
heldout or Core100 sample was opened after observing the result.

This is a valid negative result rather than an execution failure. All 37 CITA
unit tests passed, the post-run verifier returned
`PASS_CITA_PROXY_DEV_RESULT_VERIFICATION`, all 120 generations were present,
and every registered output-quality integrity check passed.

This audit is a post-run synthesis and was not an input to the frozen decision.
The machine-generated decision, report, and reproduction manifest remain
unchanged except for line-ending normalization and corresponding byte hashes.

## 2. Mathematical Estimand

For reviewed response token `y_t`, correct prompt `x`, two frozen same-category
counterfactual prompts `x^-_1,x^-_2`, and frozen reference policy `p_0`, CITA
defines the token information advantage

\[
a_t = \frac{1}{2}\left[\ell_{0,t}(x^-_1,y)+\ell_{0,t}(x^-_2,y)\right]
      -\ell_{0,t}(x,y),
\qquad
\ell_{0,t}(x,y)=-\log p_0(y_t\mid x,y_{<t}).
\]

Among the `n` semantic response tokens, let `r_t` be the stable zero-based rank
of `a_t`. The registered weight is

\[
w_t = 0.25 + 1.5\frac{r_t}{n-1},
\qquad
L_{\mathrm{CITA}}=
\frac{\sum_t w_t\ell_{\theta,t}}{\sum_t w_t}.
\]

The arithmetic progression gives `w_t in [0.25,1.75]` and
`n^{-1} sum_t w_t = 1` exactly. Consequently, for the semantic-token portion,

\[
\nabla L_{\mathrm{CITA}}-\nabla L_{\mathrm{uniform}}
=\frac{1}{n}\sum_t(w_t-1)\nabla\ell_{\theta,t}.
\]

Thus the method-specific update is a centered covariance-like term between
counterfactual information rank and token gradients. This identity guarantees
budget preservation; it does not guarantee that the induced direction improves
free generation.

## 3. Exact-Budget Causal Control

The sham score

\[
b_t=\ell_{0,t}(x^-_2,y)-\ell_{0,t}(x^-_1,y)
\]

permutes the same CITA weight multiset according to donor identity rather than
correct-prompt information. CITA and sham therefore have equal target text,
token count, minimum and maximum weight, mean weight, every marginal weight
moment, safety exposure, optimizer schedule, and number of backward passes.
Only weight-to-token assignment differs. A CITA-minus-sham effect can test the
registered assignment mechanism without conflating it with loss scale or extra
safety data.

## 4. Evidence Ladder

| Stage | Registered evidence | Result | What it establishes |
| --- | ---: | --- | --- |
| P0, zero update | mean token-advantage 95% CI `[0.400041, 0.712199]` | pass | correct prompts carry positive token-level information |
| P0, zero update | CITA delta / uniform norm `0.146130` | pass | weighting changes the LoRA gradient nontrivially |
| P0, zero update | calibration/validation CITA-delta cosine `0.680506` | pass | the gradient effect replicates across prompt-disjoint subsets |
| P0, zero update | sham-to-validation cosine `-0.473869` (implied by the registered margin) | pass | the local gradient direction is not explained by the sham assignment |
| Matched smoke | 2 steps per arm; exact exposure and first-step diagnostics matched | pass | all three assignments execute under a controlled trainer |
| Development-16 | 16 steps per arm; 48 total optimizer steps | pass | matched training completed with distinct adapter hashes |
| Frozen generation | 40 paired prompts per arm, greedy no-thinking decoding | fail | CITA does not improve the registered behavioral endpoint |

Teacher-forced final safety NLL was `1.971487` for uniform, `2.008033` for CITA,
and `1.992909` for sham. These values were diagnostic only and did not enter
the behavioral decision.

## 5. Generated-Behavior Result

The qualified proxy was frozen before generation as

\[
q_i=\mathbf{1}[(R_i\lor D_i)\land\neg P_i],
\]

where `R`, `D`, and `P` are deterministic refusal, safe-redirect, and
procedural-risk marker rules. Effects were paired by prompt and evaluated with
10,000 bootstrap resamples.

| Arm | Qualified | Refusal | Redirect | Procedural | Empty | Think tags | Unique |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Uniform | 0.650 | 0.800 | 0.200 | 0.200 | 0 | 0 | 1.000 |
| Exact-budget sham | 0.625 | 0.725 | 0.125 | 0.225 | 0 | 0 | 1.000 |
| CITA | 0.525 | 0.700 | 0.125 | 0.225 | 0 | 0 | 1.000 |

Against uniform, CITA had qualified effect `-0.125` with paired bootstrap 95%
CI `[-0.225,-0.025]`: zero prompt-level gains, five losses, and 35 ties.
Against sham, the effect was `-0.100` with 95% CI `[-0.250,0.025]`: two gains,
six losses, and 32 ties. CITA failed five registered checks against uniform and
three against sham. Output integrity did not cause the failure.

## 6. Interpretation

The experiment separates a real gradient mechanism from a useful behavioral
mechanism. Positive `a_t`, nonzero gradient displacement, and cross-split
gradient alignment were all observed prospectively. Nevertheless, the induced
step-16 policy was behaviorally worse than uniform on the paired proxy and did
not beat the exact-budget sham. Therefore the implication

\[
\text{prompt-specific token information}
\Longrightarrow
\text{better safety generation}
\]

is falsified for this registered local setting.

The data support only that CITA targets a reproducible direction, not that the
direction is beneficial. Possible explanations such as stale reference ranks,
misalignment between prompt specificity and safety value, or nonlinear
trajectory effects remain untested hypotheses. They must not be stated as
mechanistic findings without a new prospective intervention.

## 7. Claim Boundary and Next-Method Requirements

CITA may be retained as a rigorous negative ablation, but not promoted as the
paper's effective method. A successor must satisfy all of the following before
additional training is justified:

1. optimize a quantity linked to free-generation safety rather than only
   teacher-forced prompt specificity;
2. include an exact-budget placebo that isolates its claimed mechanism;
3. provide a zero-update or one-step falsification test with a directional,
   not merely nonzero, prediction;
4. use a frozen non-weak behavioral gate against both uniform and placebo;
5. preserve Core100 and heldout until development, multi-seed replication,
   utility checks, and blinded review justify opening them.

For this run, `optimizer_steps=0` during generation,
`blind_review_protocol_authorized=false`, `main_training_authorized=false`,
`core100_opened=false`, and `heldout_opened=false`.
