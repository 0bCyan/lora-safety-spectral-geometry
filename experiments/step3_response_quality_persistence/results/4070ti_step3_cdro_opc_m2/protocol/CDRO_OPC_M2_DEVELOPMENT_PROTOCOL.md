# CDR-OPC M2 Development Protocol

This is a retrospective replay of the already-open M1 panel through the new
conformal safe-dose rule. The fixed M1 endpoint and fallback are not changed.
For each of the twenty units, the runner computes the exact maximum dose that
keeps all four protected first-order scores within the frozen 4% standardized
drop allowance. At `alpha=0.05`, the development dose is the minimum of those
twenty limits and carries a `20/21` marginal bound only under the theorem's
exchangeability and fit-fixed-direction assumptions.

Because this rule was created after M1 was opened, the replay can establish
internal arithmetic consistency and estimate whether a nonzero useful dose
exists. It cannot be called fresh confirmation, and the M1 units may not be
reused to confirm M2. No training, generation, Core100, or heldout access is
authorized.
