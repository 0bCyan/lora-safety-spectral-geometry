# CDR-OPC M5 Zero-Step Confirmation Interpretation

The prospectively fixed dose `0.0337860296353059` passed the prompt-disjoint
M5 confirmation. Calibration protected `20/20` units and therefore authorized
opening the sealed audit. The sealed audit protected `19/20` units, exactly the
registered minimum.

Both behavior pools remained positive. Calibration quality/safety
selected-minus-fallback values were `0.0024378443` and `0.0038091288`; sealed
audit values were `0.0024094809` and `0.0039110438`. Aggregate protection also
passed in both splits.

This is stronger than the M4 same-review-pool holdback result because the M5
prompts were selected independently of M3. It remains a zero-step gradient
mechanism result: one sealed unit failed, no parameter update was applied, no
new response was generated, and neither Core100 nor heldout data was opened.
The result may motivate a separately frozen finite-radius virtual-step design,
but it does not authorize training.
