# CSPR M0 Data and Execution Freeze

Route: `CSPR_M0_DATA_AND_EXECUTION_FROZEN_ZERO_UPDATE_ONLY`.

Eight category-diverse mechanism-dev preference pairs and eight category-diverse constant20 safety rows were selected by the frozen token-length and stable-hash rules. All true, swapped, wrong-prompt, and safety sequences fit the 256-token encoding without truncation. No human label is used and no candidate is claimed behaviorally qualified.

The frozen package authorizes one real-model zero-update mechanism run only after commit and push. It does not authorize an optimizer, checkpoint write, generation, target qualification, core100, heldout, or training. `optimizer_steps=0` and `training_authorized=false`.
