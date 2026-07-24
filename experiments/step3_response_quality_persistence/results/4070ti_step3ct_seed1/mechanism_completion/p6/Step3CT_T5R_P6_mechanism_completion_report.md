# Step3C-T5R P6 Mechanism Completion Report

Route: `M1_reconstruction_integrity_failure`

## Integrity

- Representation probes: 12/12 complete with exact 40-prompt/80-target order.
- Grouped-gradient labels: 3/4 reproduced.
- Frozen source artifacts unchanged: True.
- No diagnostic optimizer update, full-gradient serialization, new arm, new training, or heldout access occurred.

## Representation Trajectories

| Arm | Step | Top1 | Top5 | MRR | Recall@5 | Global gap | Hard gap | Prompt rank | Target rank |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| C0R_concurrent_constant20 | 0 | 0.2500 | 0.5000 | 0.3862 | 0.3500 | 0.0422 | 0.0356 | 19.78 | 26.85 |
| C0R_concurrent_constant20 | 16 | 0.1500 | 0.5750 | 0.3278 | 0.4250 | 0.0329 | 0.0280 | 19.68 | 25.54 |
| C0R_concurrent_constant20 | 32 | 0.1250 | 0.5000 | 0.2938 | 0.3500 | 0.0335 | 0.0287 | 20.02 | 25.44 |
| C0R_concurrent_constant20 | 64 | 0.1500 | 0.4250 | 0.3077 | 0.3250 | 0.0361 | 0.0313 | 20.18 | 25.91 |
| C3a_multi_positive_global_binding | 0 | 0.2500 | 0.5000 | 0.3862 | 0.3500 | 0.0422 | 0.0356 | 19.78 | 26.85 |
| C3a_multi_positive_global_binding | 16 | 0.7750 | 0.9500 | 0.8488 | 0.8875 | 0.0253 | 0.0218 | 18.89 | 24.80 |
| C3a_multi_positive_global_binding | 32 | 0.8250 | 0.9750 | 0.8840 | 0.8875 | 0.0220 | 0.0188 | 18.43 | 23.53 |
| C3a_multi_positive_global_binding | 64 | 0.9000 | 0.9750 | 0.9340 | 0.9500 | 0.0264 | 0.0225 | 18.64 | 23.25 |
| C3b_hierarchical_hard_negative_binding | 0 | 0.2500 | 0.5000 | 0.3862 | 0.3500 | 0.0422 | 0.0356 | 19.78 | 26.85 |
| C3b_hierarchical_hard_negative_binding | 16 | 0.7750 | 0.9500 | 0.8509 | 0.8750 | 0.0255 | 0.0220 | 18.97 | 24.83 |
| C3b_hierarchical_hard_negative_binding | 32 | 0.8250 | 0.9750 | 0.8853 | 0.8875 | 0.0220 | 0.0188 | 18.45 | 23.53 |
| C3b_hierarchical_hard_negative_binding | 64 | 0.8500 | 1.0000 | 0.9175 | 0.9750 | 0.0261 | 0.0225 | 18.62 | 23.25 |

## Boundary

These results are mechanism-only evidence. They do not select or rescue a candidate, and the prior C3b redundancy warning remains active. Track A remains at `T5R1_reviewer_access_stop`; without `T5R4`, T6 and all new training remain unauthorized.
