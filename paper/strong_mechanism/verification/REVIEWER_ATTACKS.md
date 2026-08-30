# Simulated Reviewer Attack Checklist

Before submission, answer every question with a table, figure, protocol clause, or explicit limitation.

## Novelty and mechanism

1. Is `constant20` simply more safety data than `decay10`?
2. Does equal-dose timing change persistence, or does the effect disappear?
3. Is the late gradient ratio explanatory, merely correlated, or a post-hoc measurement?
4. Is the sequence-supervision result stronger than a trivial refusal increase?
5. Why is the geometry section necessary to the paper's central argument?

## Generalization

6. Are the strongest findings confined to one Qwen3-4B seed?
7. Does the direction reproduce across all three seeds?
8. What happens on a second model or risk source?
9. Are conclusions specific to the natural target manifest or greedy decoding?

## Measurement

10. Why should model-review actionability labels be trusted?
11. Does the final rubric agree with independent humans?
12. Were endpoints repeatedly changed after failures?
13. Which measurements are exploratory and which are confirmatory?
14. Is quality evaluated only after conditioning on safe responses?
15. Could response length or truncation drive quality scores?

## Statistics

16. Are prompts incorrectly treated as independent across seeds?
17. Are paired tests and confidence intervals reported?
18. Is the effect practically meaningful, not merely significant?
19. How are missing or crashed runs handled?
20. Was the confirmatory panel opened only after analysis freeze?

## Claims

21. Does any section imply a new method was validated?
22. Are spectral negatives overgeneralized?
23. Is `causal`, `governs`, or `determines` used beyond identified contrasts?
24. Are M5/M6/M7 zero-step or virtual-displacement probes clearly separated from real training evidence?
25. Can every abstract sentence be mapped to `EVIDENCE_MAP.md`?
