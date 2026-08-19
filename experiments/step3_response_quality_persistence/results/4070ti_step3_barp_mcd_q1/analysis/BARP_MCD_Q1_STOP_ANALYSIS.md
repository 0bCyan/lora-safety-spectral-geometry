# BARP-MCD Q1 Stop Analysis

## Decision

The frozen two-step matched qualification reproduced

```text
STOP_BARP_MCD_Q1_VRAM_FAILED
```

All four authorized optimizer steps completed and independent `verify-run`
reproduced six artifact hashes and the decision. The result is not promoted to
a pass: the registered VRAM gate failed.

## Mechanism result before the stop

The current-state optimizer map worked as intended. Its maximum relative L2
error against the real PyTorch AdamW update was `1.1031e-6`, below the frozen
`5e-6` limit. BARP selected nonzero homotopy fractions `0.25` and `0.125` on
its two consecutive steps. Its minimum protected retention was `0.950623`,
and its minimum target-bottleneck gain over the corresponding current raw
anchor was `0.007497`. Every same-batch registered loss decreased after every
real update in both arms.

The excluded diagnostic block was descriptive only. Relative to raw MCD,
BARP's two-step reductions changed by `+3.80%` on E-TARC, `+3.78%` on branch
entropy, and `+18.26%` on branch negative-gap. Protected reductions changed
by `-0.13%` on risk, `-0.56%` on uniform, `-8.03%` on TARC, and `-0.62%` on
continuation. These values did not select the route and are not confirmation.

## VRAM root cause

The machine reports `12,878,086,144` physical bytes. Q1 recorded peak
reserved values of `16,894,656,512` and `16,877,879,296` bytes, so the frozen
10% headroom gate correctly failed.

The cause is an implementation tensor-lifetime error. The Q1 current-state
search retained all 126 non-full subset directions on CUDA until final
selection. With `16,515,072` trainable FP32 parameters, those retained
directions alone account for

```text
126 * 16,515,072 * 4 bytes = 8,323,596,288 bytes = 7.751953 GiB.
```

They are unnecessary. Endpoint scoring needs only scalar records; after the
Top-16 masks are known, the corresponding directions can be reconstructed
deterministically from the unchanged Gram matrix. This matches the lifetime
strategy used by the frozen P0 implementation.

## Next boundary

Q1 remains a stop. Its optimizer budget cannot be reused and its artifacts
must remain unchanged. A Q1R rerun requires a new prospective protocol. The
only allowed implementation change is to stop retaining all endpoint tensors
and reconstruct screened directions. Data, two-arm assignment, four-step
budget, candidate family, tie breakers, thresholds, optimizer, and diagnostic
role remain unchanged.
