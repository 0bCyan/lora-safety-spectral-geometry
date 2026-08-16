# CSPR P2T A/B Analysis Execution Protocol

Status: prospective after all four A/B outputs are committed and before any A/B label comparison.

This execution layer reuses the unchanged P2S calibration, reference-concordance, target-reliability, bootstrap, hidden-duplicate, and disagreement definitions. It changes no label, packet, rubric, hidden key, threshold, bootstrap seed, pair rule, or fixed-pool gate.

The four canonical review files and their provenance records are hash-bound before analysis. Preflight validates complete packet coverage, template order, decision domains, output hashes, four distinct reviewer identities and sessions, and all frozen input hashes. It reports no label prevalence or agreement.

Analysis separates calibration primaries from target primaries. Calibration A/B reliability, each reviewer's frozen-reference concordance, 10,000-replicate Actionability bootstrap intervals, target A/B reliability, and hidden duplicate consistency must all pass. Target score prevalence remains descriptive and cannot qualify or disqualify a reviewer.

Calibration failure selects `STOP_CSPR_P2T_CALIBRATION_UNRELIABLE`. Duplicate failure selects `STOP_CSPR_P2T_DUPLICATE_UNRELIABLE`. Target reliability failure selects `STOP_CSPR_P2T_TARGET_UNRELIABLE`. Only an all-gate pass selects `AUTHORIZE_CSPR_P2T_REVIEWER_C_PROTOCOL_ONLY` and writes target-disagreement-only C packets and blank templates.

No Reviewer C is launched by this analysis. C requires a separate zero-access qualification and execution protocol. Pair qualification, hidden outcome selection, model gradients, optimizer updates, training, core100, and heldout remain unauthorized. `optimizer_steps=0` and `training_authorized=false`.
