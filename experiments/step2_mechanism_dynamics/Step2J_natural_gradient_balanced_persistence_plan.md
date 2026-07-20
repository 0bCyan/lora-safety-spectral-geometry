# Step2J Natural-Target Gradient-Balanced Persistence Plan

## 1. Entry Decision

Step2I-R natural-target seed1 establishes a transient but non-persistent safety effect:

| Checkpoint | Natural decay-to-10 unsafe | Previous programmatic decay-to-10 unsafe |
|---|---:|---:|
| step16 | 0.67 | 0.36 |
| step32 | 0.38 | 0.12 |
| step64 | 0.76 | 0.55 |

For the natural arm, 41 of the 62 step32 refusals become unsafe at step64, while only 3 unsafe responses become refusals. Utility100 is preserved (`0.80` versus `0.81` vanilla; drop `0.01`; degradation `0.00`). The current evidence therefore supports late safety loss, not an unavoidable safety-utility tradeoff.

The precise closure is:

> Natural replay with a schedule decaying from 20% to 10% fails to preserve step64 safety on risk seed1.

Do not claim that every fixed-weight natural replay strategy has failed. A constant-20 control is still required to distinguish insufficient safety-gradient magnitude from a need for adaptive balancing.

Do not run seeds 2 and 3, open `heldout_test`, or reintroduce SpectraSafe in Step2J.

## 2. Primary Question

> Can a safety weight derived from the measured risk and natural-target gradient magnitudes preserve late safety better than decay-to-10, without the utility cost of a constant 20% safety weight?

The experiment must distinguish three explanations:

```text
H1: decay-to-10 simply leaves too little late safety-gradient magnitude
H2: risk-safety gradient competition requires dynamic balancing
H3: neither weighting strategy is sufficient under the current target and LoRA setup
```

Step2J is an exploratory method-development stage. A positive seed1 result may nominate a method for cross-seed validation but is not confirmatory evidence by itself.

## 3. Step2J-A: Mandatory Entry Audit

Complete all Step2J-A blocks before new training.

### 3.1 Independent natural-target review

The existing `step2ir_natural_diverse_targets_v1_review.csv` is a local rule-based preflight produced by the same preparation script. It is not an independent blind review.

Use a separate agent or human reviewer to inspect all 80 frozen v1 targets without seeing training results. Record:

```text
target_id
explicit_refusal_validity
harmful_procedural_leakage
prompt_relevant_redirect
grammatical_naturalness
semantic_coherence
keyword_stuffing
mechanical_filler
overall_decision
short_reason
```

Frozen target gate:

```text
harmful procedural leakage = 0
prompt-relevant redirect >= 0.90
grammatical naturalness >= 0.90
keyword stuffing <= 0.10
mechanical filler <= 0.10
```

If every gate passes, retain v1 unchanged. If any gate fails, create `natural_diverse_targets_v2`; do not edit v1 in place. Review and freeze v2 before training, and rerun every Step2J arm on v2.

### 3.2 Evidence-integrity export

Recover and commit small reproducibility artifacts from the completed natural seed1 run:

```text
anchor_draw_trace.csv
anchor_exposure_manifest.csv
checkpoint_index.csv
training/config and target hashes
sampler cursor final state
```

Verify:

```text
80 targets were used
256 anchor draws occurred
each target appeared 3 or 4 times
fixed and natural prompt order matches the frozen manifest
manifest and review hashes match their freeze file
```

If the exposure trace cannot be recovered, the existing decay-to-10 run may remain a historical diagnostic, but it cannot serve as the formal Step2J control. Rerun it alongside the other arms.

### 3.3 Correct the natural-pilot record

Update the pilot report so that it states:

```text
the 1.000 target-quality values are local preflight values, not independent approvals
step64 unsafe 0.76 fails the 0.60 gate
step64 all-prompt redirect 0.09 fails the 0.80 gate
semantic clustering and calibrated four-class behavior remain pending
only decay-to-10 is closed, not every fixed-weight strategy
```

Do not change any observed result while correcting its interpretation.

## 4. Step2J-A Natural-Target Gradient Audit

The existing Step2I-R pairwise audit used the rejected programmatic targets. Do not assume that its gradient distribution transfers to natural targets.

At natural decay-to-10 checkpoints step16, step32, and step64, run two fixed disjoint diagnostic blocks. Each block contains 8 risk examples and 8 natural safety examples. Disable LoRA dropout and use deterministic evaluation mode.

For every 8 x 8 risk-safety pair and for all LoRA parameters plus attention/MLP groups, report:

```text
risk gradient norm ||g_r||
safety gradient norm ||g_s||
cosine(g_r, g_s)
negative-pair fraction
mean negative cosine magnitude
10th, 50th, and 90th cosine percentiles
agreement between the two diagnostic blocks
```

Because the trainer combines losses as

```text
g_combined = (1 - w) * g_r + w * g_s
```

also report at the observed schedule weight:

```text
risk contribution       = (1 - w) * ||g_r||
safety contribution     = w * ||g_s||
contribution ratio      = safety contribution / risk contribution
cosine(g_combined, g_r)
cosine(g_combined, g_s)
equal-norm weight       = ||g_r|| / (||g_r|| + ||g_s||)
```

This audit chooses between the following preregistered interpretations:

```text
late safety contribution ratio < 0.50:
    magnitude insufficiency is supported

negative-pair fraction >= 0.60 in both blocks, with stable sign across checkpoints:
    widespread gradient competition is supported

neither condition:
    do not build an adaptive method from the current mechanism story
```

The audit is diagnostic only. Do not select favorable prompts, layers, or checkpoints after inspection.

## 5. Step2J-B Three-Arm Seed1 Design

Run Step2J-B only if target review passes and the natural gradient audit supports magnitude insufficiency or stable competition.

All arms use:

```text
Qwen3-4B
risk seed1
same 256 risk rows and ordering
same frozen natural target manifest
same deterministic micro-step sampler
same 80-target order and 256 safety draws
same optimizer, LoRA modules, rank, dropout, quantization, and checkpoints
```

### 5.1 Arm J-control-decay10

```text
step0 through step16: w = 0.20
step16 through step64: linear decay from 0.20 to 0.10
```

Reuse the existing natural seed1 run only if the independent review leaves the manifest unchanged and all exposure artifacts are recovered. Otherwise rerun it.

### 5.2 Arm J-control-constant20

```text
step0 through step64: w = 0.20
```

This is the essential simple control. If it matches or beats the adaptive arm without additional utility loss, do not promote the adaptive method.

### 5.3 Arm J-adaptive-norm

Use gradient-norm balancing without projection.

Warm start:

```text
step0 through step16: w = 0.20
```

For each optimizer step after step16, estimate unweighted risk and safety gradient norms on the first micro-batch, using all trainable LoRA parameters. Maintain exponential moving averages:

```text
ema_r[t] = 0.90 * ema_r[t-1] + 0.10 * ||g_r[t]||
ema_s[t] = 0.90 * ema_s[t-1] + 0.10 * ||g_s[t]||
raw_w[t] = ema_r[t] / (ema_r[t] + ema_s[t] + 1e-12)
w[t] = clip(raw_w[t], 0.10, 0.30)
```

Use `w[t]` for all four gradient-accumulation micro-batches in that optimizer step. Treat the weight as stop-gradient. Do not add a cosine bonus, gradient projection, or manually tuned checkpoint-specific override in this pilot.

Log every step:

```text
raw risk and safety norms
EMA norms
raw and clipped weight
risk-safety cosine
weighted contribution ratio
clipping boundary hit
anchor indices and sampler cursor
```

Before the full run, add deterministic unit tests and a two-step smoke test verifying:

```text
weight formula and clipping
one weight update per optimizer step
four micro-batches share the same weight
no gradient from the weight estimator enters model parameters
sampler exposure order is unchanged
resume restores EMA, adaptive weight, and sampler cursor
no NaN/Inf under fp16 and 4-bit loading
```

If full-LoRA `autograd.grad(..., retain_graph=True)` exceeds 4070 Ti memory, stop and report the measured peak memory. Do not silently switch to a selected-layer proxy. A proxy requires a separately reviewed plan.

## 6. Evaluation

Run deterministic safety30 at:

```text
step0, step6, step16, step32, step48, step64
```

Run safety100 at:

```text
step16, step32, step64
```

Step64 remains the primary endpoint. Report every arm and checkpoint, including:

```text
four-class calibrated behavior when available
heuristic unsafe and refusal for continuity
mixed refusal-compliance counted unsafe
invalid rate
degradation
all-prompt relevant redirect
redirect conditional on clean refusal
largest exact cluster
agent semantic-template cluster
distinct-2
response length
unsafe trajectory area
```

Run utility100 at step64 for all three arms using the same frozen utility set and vanilla baseline.

Keep `heldout_test` sealed.

## 7. Seed1 Candidate Gate

An intervention arm is eligible for seed expansion only if all conditions hold:

```text
step64 unsafe <= 0.60
absolute step64 unsafe improvement >= 0.15 versus decay-to-10
step64 all-prompt relevant redirect >= 0.80
largest exact cluster <= 0.10
largest semantic cluster <= 0.20
mixed refusal-compliance <= 0.05
invalid-response increase <= 0.05
maximum degradation <= 0.15
utility100 task-success drop <= 0.05
utility100 unnecessary-refusal increase <= 0.05
```

An unmeasured item is pending, not passing. Do not replace all-prompt redirect with redirect conditional on clean refusal after seeing results.

## 8. Method Promotion Rule

### Constant-20 wins or ties

If constant-20 passes and adaptive-norm does not provide a reproducible improvement of at least `0.05` unsafe rate or `0.03` utility task success:

```text
retain constant-20 as the simpler candidate
do not claim a new adaptive optimization method
interpret Step2I-R as a safety-gradient dose result
```

### Adaptive-norm wins

Adaptive-norm becomes a provisional method candidate only if:

```text
it passes every seed1 gate
constant-20 fails a gate, or adaptive improves a primary outcome by the minimum margin above
its weight trajectory is finite, stable, and not pinned at one clipping boundary for more than 80% of post-step16 updates
the result matches the preregistered gradient mechanism
```

Then freeze the algorithm and run unchanged risk seeds 2 and 3. No method name is assigned before cross-seed validation.

### Neither arm passes

If neither constant-20 nor adaptive-norm passes:

```text
close loss-weight replay as the primary method route
retain the result as a controlled mechanism study
do not scan more weight floors
move to parameter-isolated retention only under a new plan
```

## 9. Required Deliverables

```text
independent 80-target blind review and adjudication
corrected natural seed1 report
recovered exposure and checkpoint metadata
natural-target pairwise gradient audit tables
Step2J implementation tests and smoke-test log
three-arm seed1 configs and run manifests
safety30 and safety100 trajectories
utility100 comparison
semantic-template audit
explicit constant-dose, adaptive-method, or route-closure decision
```

## 10. Immediate Execution Order

```text
1. independent target review
2. evidence-integrity export and pilot-report correction
3. natural-target gradient audit
4. audit-only Step2J-A decision report
5. implement and smoke-test adaptive-norm only if audit conditions pass
6. run the three-arm Step2J-B seed1 comparison
7. issue the promotion or closure decision
```
