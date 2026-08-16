# CSPR M1 Disjoint-Batch Replication Report

Route: `STOP_CSPR_M1_PROJECTION_NOT_REPLICATED`.

Active pairwise projections: `8/18`. Active aggregate batches: `1/3`. Active batches with a projected dual-Armijo step: `1/3`. Registered same-step safety rescues: `1`.

This is a real-model zero-update mechanism replication over three frozen disjoint batches. It is not a behavior, safety-generalization, target-quality, or training-benefit result. No optimizer or scheduler was created, no checkpoint or candidate output was written, and the exact initial trainable-state hash was restored after every batch. `optimizer_steps=0` and `training_authorized=false`.
