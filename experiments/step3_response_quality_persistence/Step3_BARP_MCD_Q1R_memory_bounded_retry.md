# Step3 BARP-MCD Q1R: Memory-Bounded Qualification Retry

Q1 completed all four optimizer steps but stopped only at its frozen VRAM
gate. The implementation retained 126 full FP32 endpoint directions, an
unnecessary estimated `7.751953 GiB`. All optimizer-map, protection, target,
assignment, and finite-step gates passed.

Q1R is a new prospective run with a new optimizer budget. It changes one
implementation detail only:

- endpoint directions are deleted immediately after scalar scoring;
- after the unchanged Top-16 mask screen, each screened direction is
  reconstructed deterministically from the unchanged normalized-gradient Gram
  matrix;
- at most one reconstructed endpoint direction is retained at a time.

The two seed-4 blocks, excluded diagnostic, two arms, two steps per arm,
AdamW configuration, all 127 endpoint masks, Top-16 screen, 129 final
candidates, tie breakers, four protected coordinates, three target
coordinates, 95% retention floor, route order, and every numeric threshold are
unchanged. Q1 artifacts and its stop decision remain immutable.

Q1R passes only under the original Q1 gates. A pass authorizes a separately
frozen longer matched-training protocol, not generation or a behavior claim.
