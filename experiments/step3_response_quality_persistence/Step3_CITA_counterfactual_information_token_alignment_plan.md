# Step3 CITA: Counterfactual Information Token Alignment

## 1. Evidence-Driven Decision

The next candidate is **CITA**, Counterfactual Information Token Alignment.
CITA is a token-level safety-supervision method, not another parameter-space
regularizer or whole-response ranking loss.

The proposal follows four frozen negative results:

1. C3 made prompt-response representations more identifiable but did not yield
   the registered generated-quality improvement.
2. TRIM-CS0 found no transition-specific causal steering effect; the prompt-last
   placebo was larger than the proposed transition patch.
3. PCR-M/PCR-MW produced large training-margin movement and response-only
   conditionality transfer, but neither improved the frozen generation gate.
4. CSPR repaired finite conflicting steps, but aggregate safety conflict did
   not replicate across batches and therefore cannot be the primary method.

Together these results localize the remaining hypothesis: complete-response
supervision is behaviorally active, but uniform or sequence-aggregate
objectives may spend most of their safety gradient on reusable response style
instead of the freely generated tokens whose probability is specifically
licensed by the prompt.

## 2. Method

For reviewed safety response `y` under its correct prompt `x` and two frozen
same-category wrong prompts `x^-_1,x^-_2`, define the reference-model token
advantage after the complete no-thinking generation prefix:

\[
a_t = \frac{1}{2}\sum_{k=1}^2
\ell_{0,t}(x^-_k,y)-\ell_{0,t}(x,y),
\]

where `ell_0,t` is the next-token negative log likelihood under the frozen
fresh seed-1 step-0 policy. Positive `a_t` means that the correct prompt gives
the model more evidence for target token `y_t` than the two content-matched
counterfactual prompts do.

Let `r_t` be the stable zero-based rank of `a_t` among non-special freely
generated response tokens. CITA assigns

\[
w_t^{\rm CITA}=0.25+1.5\frac{r_t}{n-1},
\]

with weight one for special termination tokens. Thus semantic-token weights
are exactly in `[0.25,1.75]` and have exact mean one. The safety loss is

\[
L_{\rm CITA}
=\frac{\sum_t w_t^{\rm CITA}\ell_{\theta,t}(x,y)}
       {\sum_t w_t^{\rm CITA}}.
\]

The full training objective, if later authorized, preserves the proven replay
schedule:

\[
L=(1-p)L_{\rm risk}+pL_{\rm CITA},\qquad p=0.20.
\]

CITA introduces no auxiliary coefficient, extra safety draw, online judge, or
behavior-dependent adaptation.

## 3. Exact-Budget Sham

For the same response tokens, define an arbitrary donor-identity contrast

\[
b_t=\ell_{0,t}(x^-_2,y)-\ell_{0,t}(x^-_1,y).
\]

The sham arm assigns the **same CITA weight multiset** according to the stable
rank of `b_t` instead of `a_t`. Therefore CITA and sham have exactly equal:

- minimum, maximum, mean, and multiset of token weights;
- response text, prompt, target exposure, safety dose, optimizer schedule, and
  number of backward passes;
- sequence length and total per-response loss weight.

Only the alignment between weight and correct-prompt token information differs.
This rules out explanations based on a larger loss scale, more safety exposure,
or generic emphasis on difficult tokens.

## 4. Mathematical Interpretation

For semantic tokens, `sum_t(w_t-1)=0`. Hence

\[
\nabla L_{\rm CITA}
=\nabla L_{\rm uniform}
+\frac{1}{n}\sum_t(w_t-1)\nabla\ell_t.
\]

The method-specific term is the empirical covariance between
correct-prompt information rank and token gradients. It is zero when token
gradients are exchangeable with respect to prompt information, and non-zero
only when high-information tokens induce a systematically different update.
Because the sham weights are a permutation of the CITA weights, their zeroth
moment, first moment, squared norm, minimum, maximum, and every higher marginal
moment are exactly equal. The CITA-minus-sham contrast therefore isolates the
assignment of the fixed weight budget to token positions; it cannot be caused
by a larger scalar loss budget.

The rank transform is invariant to every strictly increasing rescaling of
`a_t`; this removes temperature and score-scale tuning. Stable positional tie
breaking makes the cache byte deterministic.

These facts are algebraic guarantees. They do not guarantee generated behavior,
which remains a separate matched-arm endpoint.

## 5. Serialization Boundary

All safety arms use the complete `enable_thinking=false` generation prefix and
score only tokens after that prefix. The forced four-token empty-thinking
preamble is input, never a loss target. This applies equally to the future
uniform control, CITA, and sham arms, so serialization repair cannot masquerade
as a CITA effect.

Risk training remains byte-identical to the existing matched trainer.

## 6. P0 Zero-Update Gate

P0 uses the frozen 74-row, 37-prompt natural-v1 mismatch packet. It creates a
74-row immutable token-weight cache at the fresh seed-1 step-0 model and forms
two deterministic, prompt-disjoint 16-prompt sets: calibration and validation.
No proxy-dev, core100, heldout, generation, reviewer, optimizer, or checkpoint
is used.

P0 passes only if all registered conditions hold:

1. every correct/wrong response suffix has identical token IDs, at least two
   semantic tokens, finite NLLs, and exact CITA/sham weight-multiset equality;
2. the row-level mean token advantage has a 10,000-draw paired-bootstrap lower
   bound above zero;
3. median positive-token fraction is at least `0.50`;
4. median absolute CITA-score/sham-score rank correlation is at most `0.50`;
5. all uniform, CITA, and sham calibration/validation gradient objects cover
   every registered trainable LoRA tensor, contain only finite values, and have
   non-zero aggregate norm. The number of individually non-zero tensors is
   reported separately. Fresh PEFT LoRA initializes each `lora_B` to zero, so
   the chain rule makes the first-step `lora_A` gradient exactly zero; treating
   that structural zero as missing coverage would make a fresh-step P0
   algebraically impossible;
6. the calibration CITA-minus-uniform gradient norm is at least `0.05` of the
   uniform gradient norm;
7. calibration CITA/uniform gradient cosine is at most `0.999`;
8. calibration CITA-minus-uniform and sham-minus-uniform delta cosine is at most
   `0.95`;
9. calibration-to-validation CITA delta cosine is at least `0.05` and exceeds
   sham-calibration-to-CITA-validation cosine by at least `0.05`;
10. peak VRAM headroom is at least `10%`, the exact state hash is restored, and
    `optimizer_steps=0`.

Routes:

- structural, finite, coverage, memory, or restoration failure:
  `STOP_CITA_P0_INVALID`;
- weak counterfactual token signal:
  `STOP_CITA_P0_TOKEN_SIGNAL_WEAK`;
- weak or sham-indistinguishable gradient effect:
  `STOP_CITA_P0_GRADIENT_EFFECT_WEAK`;
- failure to replicate token-aligned gradient structure across prompt-disjoint
  sets: `STOP_CITA_P0_ALIGNMENT_NOT_REPLICATED`;
- all gates pass: `PASS_CITA_P0_AUTHORIZE_MATCHED_SMOKE`.

No threshold may be changed after P0 model execution.

## 7. Later Matched Experiment

Only a P0 pass may request a separate pushed protocol for three matched arms:

1. serialization-aligned uniform constant20 control;
2. CITA;
3. exact-budget sham.

The first stage is a two-step smoke, followed by one prospectively frozen
16-step development run. CITA must beat both control and sham on a separately
frozen generated-behavior gate with a nontrivial absolute effect. The gate must
include an absolute improvement floor in addition to arm ordering; training
loss, gradient geometry, or teacher-forced token scores cannot promote it.
Human review remains last, after deterministic behavior and safety gates pass.

## 8. Literature Position

The closest discovered token-selection precedent is Rho-1 / Selective Language
Modeling (Lin et al., 2024, arXiv:2404.07965), which uses reference-model excess
loss to select useful pretraining tokens. CITA differs in estimand and control:
it ranks the same reviewed response tokens by correct-versus-counterfactual
prompt evidence, preserves all target tokens with bounded mean-one weights, and
uses an exact weight-multiset sham. The broader data-selection literature is
summarized by Albalak et al. (2024, arXiv:2402.16827).

The dual-engine search is a discovery audit, not proof of novelty. Before a
paper novelty claim, titles, full methods, and forward citations must receive a
separate systematic review.

## 9. Claim Boundary

P0 can establish a non-degenerate, prompt-specific token-gradient mechanism.
It cannot establish behavior, safety, utility, or publication readiness. A
later development result remains exploratory until multi-seed replication,
utility testing, blind human measurement, and one sealed heldout opening.

Throughout P0, `optimizer_steps=0` and `training_authorized=false`.
