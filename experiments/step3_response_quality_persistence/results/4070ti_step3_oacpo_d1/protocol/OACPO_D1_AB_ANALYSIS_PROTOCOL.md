# OACPO D1 A/B Analysis Protocol

Status: prospective before any A/B label comparison.

The analysis accepts only six complete, order-preserving review files with exact schemas and matching sub-agent provenance. It opens hidden duplicate links only after all six files validate, computes the reliability gates frozen in the D1 base protocol, and writes an opaque disagreement ledger.

If any construct or hidden-duplicate gate fails, D1 stops as unreliable. If all gates pass and disagreements exist, the only next authorization is a separately frozen Reviewer C protocol. If all gates pass with no disagreements, the only next authorization is a separately frozen triplet-qualification protocol.

This stage cannot assign labels, inspect source roles while judging, qualify a triplet, compute a gradient, open a model checkpoint, or train. `optimizer_steps=0` and `training_authorized=false`.
