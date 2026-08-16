# CSPR M1 Data and Execution Freeze

Route: `CSPR_M1_DATA_AND_EXECUTION_FROZEN_ZERO_UPDATE_ONLY`.

Three mutually disjoint batches were frozen. Each contains six category-diverse M0-disjoint mechanism-dev preference pairs and six category-diverse M0-disjoint constant20 safety rows. All true, swapped, wrong-prompt, and safety sequences fit the 256-token encoding without truncation. No human label is used and no candidate is claimed behaviorally qualified.

The package authorizes one three-batch real-model zero-update replication only after commit and push. It does not authorize an optimizer, checkpoint write, generation, target qualification, core100, heldout, or training. `optimizer_steps=0` and `training_authorized=false`.
