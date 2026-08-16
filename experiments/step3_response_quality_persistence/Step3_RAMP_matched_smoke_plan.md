# Step3 RAMP Matched Three-Arm Smoke

## 1. Authorization

RAMP-P0 selected `PASS_RAMP_P0_AUTHORIZE_MATCHED_SMOKE_PROTOCOL` at pushed
result commit `4514b16`. That route authorizes only this separately frozen
two-step smoke. It does not authorize development training, generation,
Core100, heldout, or a paper claim.

## 2. Arms

All arms use Qwen3-4B, fresh seed-1 rank-8 LoRA, the same 256 risk rows, the
unchanged 80 approved natural-v1 safety targets, safety mix `0.20`, four
microsteps per optimizer step, and two optimizer steps.

1. `uniform`: complete-response safety cross entropy.
2. `ramp`: uniform CE plus the first-16-token strongest-competitor margin
   barrier with frozen P0 coefficient `0.26223245656756566`.
3. `suffix`: uniform CE plus the last-16-token placebo barrier with frozen P0
   coefficient `0.23919251636494498`.

Every safety forward computes both margin diagnostics, so the arms execute the
same forward structure and RNG schedule. Only the registered auxiliary term is
connected to the backward objective.

## 3. Matching

The single orchestrator runs arms in order `uniform`, `suffix`, `ramp`. It must
verify equality of:

- initial trainable-state hash;
- risk order and first-step risk loss;
- safety target order, exposure, response-token count, and available prefix
  and suffix margin terms;
- first-step uniform safety CE;
- initial prompt-disjoint prefix and suffix diagnostics;
- optimizer steps, gradient accumulation, learning-rate schedule, and sealed
  data boundary.

All arms must finish two finite updates without a skipped GradScaler step, keep
at least 10% VRAM headroom, and write distinct adapter hashes.

## 4. Prompt-Disjoint Direction Realization

The two-step schedule consumes the first eight target rows, corresponding to
four prompt clusters. None overlaps the 16 RAMP-P0 validation prompts. Each arm
evaluates that frozen validation set before step 1 and after step 2 with model
dropout disabled and no gradients.

Let

\[
R_a=C_{pre,a}^{initial}-C_{pre,a}^{final}
\]

be the prefix-certificate-loss reduction for arm `a`. The smoke direction gate
requires all of:

\[
R_{RAMP}\ge 0.005,
\quad R_{RAMP}-R_{uniform}\ge 0.002,
\quad R_{RAMP}-R_{suffix}\ge 0.002.
\]

These are teacher-forced mechanism diagnostics, not generated behavior. They
test whether the optimizer realizes P0's directional prediction after two
actual mixed-objective steps.

## 5. Routes

- input, matching, exposure, or sealed-boundary failure:
  `STOP_RAMP_SMOKE_MATCHING_FAILED`;
- numerical, step-count, state, or VRAM failure:
  `STOP_RAMP_SMOKE_NUMERICAL_FAILED`;
- identical final adapters or missing assignment-specific auxiliary terms:
  `STOP_RAMP_SMOKE_ASSIGNMENT_NOT_EXERCISED`;
- prompt-disjoint direction gate failure:
  `STOP_RAMP_SMOKE_DIRECTION_NOT_REALIZED`;
- all checks pass:
  `PASS_RAMP_SMOKE_AUTHORIZE_MATCHED_DEVELOPMENT16_PROTOCOL`.

A pass authorizes only a new protocol commit for a matched 16-step development
run. No threshold may change after this smoke protocol is committed.

## 6. Boundaries

The smoke has six total optimizer steps across three arms. It generates no
candidate response and opens neither proxy-dev, Core100, nor heldout. Local
adapter files are retained for exact verification but remain gitignored.
`main_training_authorized=false` throughout.
