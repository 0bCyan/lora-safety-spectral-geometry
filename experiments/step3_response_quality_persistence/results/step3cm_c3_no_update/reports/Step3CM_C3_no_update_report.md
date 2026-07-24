# Step3C-MF C3 No-Update Feasibility Report

Status: at least one candidate passes.

- Model/checkpoint: `Qwen/Qwen3-4B` / `experiments/step2_mechanism_dynamics/results/4070ti_step2j_b1/runs/step2jb1_4070ti_qwen3-4b_risk_s1_natural_constant20/checkpoints/step0064_pct100`
- Selected queue: 32; temperature: 0.05
- Retrieval top1/top5/MRR: 0.150 / 0.450 / 0.301
- C3a_multi_positive_global_binding: pass=True, eta=0.08881576937607806, ratio=0.25, gradient cosine=0.004502436656979584
- C3b_hierarchical_hard_negative_binding: pass=True, eta=0.05694228453242218, ratio=0.25, gradient cosine=0.004177711699927165

No optimizer or scheduler was constructed. Optimizer steps, behavior generations, core100 access, and heldout access were all zero/false. No training is authorized.
