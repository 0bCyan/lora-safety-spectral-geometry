# OACPO D0 Authoring Protocol

Status: prospective before any OACPO author output or blind-review label.

## Why a new target source is necessary

The existing evidence does not contain a valid shortcut to OACPO triplets.

- CSPR P2 proved that its fixed 120-pair pool could yield at most 26 actionability-qualified candidates under every legal disagreement adjudication, below the frozen requirement of 48.
- CSPR P2T then failed its prospective reviewer-calibration gates. Those labels cannot be reinterpreted as final merely because some marginal agreement values look favorable.
- Step2J contains reliable safety labels for 400 responses, including 104 unsafe responses, but no `generic_refusal_only=yes` response. Across its 100 prompts, 47 contain both a clean and an unsafe response, while zero contain the complete mechanical-safe, nonmechanical-safe, unsafe combination needed here.
- The 40 natural-target prompts and the 100 Step2J prompts have zero exact prompt overlap. Their responses therefore cannot be combined into same-prompt triplets.
- Core100 and heldout material are excluded from OACPO D0.

The data bottleneck is therefore specific: the public pool supplies same-prompt safe and unsafe candidate responses, but it lacks enough independently qualified, prompt-bound, actionable safe positives. D0 adds only that missing response role.

## Frozen selection

The source is the already frozen 120-pair BeaverTails ledger. Selection is independent of prior labels and outcomes. For each of 14 categories, D0 selects two `train_candidate`, one `calibration`, and one `mechanism_dev` prompt using SHA-256 ranks and seed `2026081901`. The result is 56 unique prompts with exact category balance and a 28/14/14 split.

The author receives only shuffled opaque IDs and prompts. Existing answers, roles, labels, splits, categories, training results, and gradient reports are hidden. One fresh no-context sub-agent writes all 56 answers under the frozen authoring rubric. The main session does not write or judge answers.

## Claim boundary

Packet freeze establishes deterministic selection, source integrity, blinding, and schema only. Import can establish completeness and mechanical validity only. Neither step establishes answer safety or quality.

After a valid import, a separately committed protocol must construct a three-role blind packet and launch new independent reviewers. Only reliable adjudicated labels may form triplets. A later P0 may compute zero-update gradients only if at least 42 triplets pass, including 21 train, 10 calibration, 10 mechanism-development, and 12 represented categories.

No checkpoint is opened in D0. `optimizer_steps=0`, `training_authorized=false`, Core100 remains closed, and heldout remains closed.
