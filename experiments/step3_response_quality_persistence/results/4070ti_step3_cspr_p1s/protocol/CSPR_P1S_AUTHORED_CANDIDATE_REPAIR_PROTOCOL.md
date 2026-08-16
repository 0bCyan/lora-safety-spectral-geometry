# CSPR P1S Authored Candidate Repair Protocol

Status: prospective before constructing an author packet or creating a new candidate response.

The closed P2 pool has an actionability-only qualification upper bound of 26, below the unchanged requirement of 48. P1S retains the 120 frozen prompts, pair identities, categories, prospective 64/28/28 splits, and original rejected side, but it does not reuse an old chosen response as a candidate. Two distinct fresh no-context authors each receive every prompt once in different deterministic orders. Each prompt receives one `specific_redirect` and one `risk_mitigation` assignment. Authors cannot see the old candidate, rejected response, source label, split, category, prior labels, or another author's output.

Authoring is a single complete attempt. Structural failure stops the attempt. A mechanically invalid candidate is excluded without repair, retry, replacement, or top-up. Mechanical validation checks only the unchanged 40-160 word envelope, non-empty response, forbidden thinking tags, and normalized exact duplication against prior or newly authored responses. It does not use a semantic heuristic as a safety or actionability label.

Before semantic review, the count of prompts retaining at least one mechanically valid candidate must still make all unchanged resource gates reachable: 24/64 train candidates, 12/28 calibration, 12/28 mechanism development, 48/120 total, and at least two prompts in 10/14 categories. This is only a best-case source-supply gate.

P1S also freezes the design separation required for the later P2S review protocol. Reviewer reliability will be measured on hidden, interleaved, independently adjudicated Step3O calibration anchors: 60 actionability rows balanced 20/20/20 across final scores and 40 safety rows balanced 20/20 across final labels. Existing P2 reliability thresholds are inherited without weakening and reference concordance uses the same raw thresholds. Target-pool score prevalence is an outcome, not a reviewer-reliability gate. This prevents an intentionally improved candidate distribution from invalidating the measurement instrument merely because it contains many qualified responses.

Fresh safety and actionability reviewers must be distinct from both authors and all P2 reviewers. Reviewer C can begin only after calibration passes and can see only target A/B disagreements. A prompt qualifies only when one authored candidate is independently adjudicated clean-safe and score 2 while the original rejected response is independently adjudicated not clean-safe. If both authored candidates qualify, a frozen stable hash selects one; no holistic preference label is used.

P1S can authorize only a separately frozen P2S review-packet protocol. It does not authorize semantic labels, Reviewer C, model gradients, checkpoints, core100, heldout content, or training. `optimizer_steps=0` and `training_authorized=false`.
