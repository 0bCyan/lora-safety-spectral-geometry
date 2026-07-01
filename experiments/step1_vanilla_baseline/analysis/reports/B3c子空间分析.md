# B3c adapter update subspace analysis

## 1. 产物

- `analysis/tables/subspace_overlap_long.csv`：逐模块、逐 pair、逐 side/k 的子空间重叠长表。
- `analysis/tables/subspace_overlap_summary.csv`：按 pair_type、block_type、side、k 聚合。
- `analysis/tables/subspace_overlap_module_summary.csv`：进一步按 module 聚合。

## 2. 关键读数：right/input side

| pair_type | k=1 cosine | k=2 cosine | k=4 cosine |
|---|---:|---:|---:|
| risk_seed_to_seed | 0.0936 | 0.0686 | 0.0531 |
| risk_vs_normal | 0.3072 | 0.3017 | 0.3120 |
| risk_vs_control | 0.3086 | 0.3034 | 0.3129 |

## 3. 关键读数：left/output side

| pair_type | k=1 cosine | k=2 cosine | k=4 cosine |
|---|---:|---:|---:|
| risk_seed_to_seed | 0.6259 | 0.5606 | 0.4666 |
| risk_vs_normal | 0.3087 | 0.2833 | 0.2444 |
| risk_vs_control | 0.2808 | 0.2658 | 0.2384 |

## 4. left/output k=1 模块分布

| module | risk seed-to-seed | risk vs normal | risk vs control |
|---|---:|---:|---:|
| q_proj | 0.5370 | 0.2347 | 0.2185 |
| k_proj | 0.5432 | 0.2373 | 0.2231 |
| v_proj | 0.7305 | 0.4301 | 0.3887 |
| o_proj | 0.7370 | 0.3951 | 0.3514 |
| gate_proj | 0.5432 | 0.2349 | 0.2207 |
| up_proj | 0.6374 | 0.2823 | 0.2572 |
| down_proj | 0.6534 | 0.3468 | 0.3058 |

## 5. 判读

B3c 支持子空间机制需要看到 risk seed-to-seed overlap 明显高于 risk vs normal/control，且该差异在少数模块族或层段稳定集中。

当前结果是混合的：left/output side 上 risk seed-to-seed overlap 明显高于 risk vs normal/control，说明 risk adapter 的输出方向存在 family 内一致性；但 right/input side 上 risk seed-to-seed overlap 很低，且低于 risk vs normal/control。

因此 B3c 提供了部分描述性信号，但还不足以推翻 B2/B3a/B3b 的负结果，也不足以支持 post-hoc 参数空间干预的强因果机制叙事。更稳妥的结论是：LoRA 更新几何能描述 family 差异，尤其是 output-side 子空间差异；但简单谱坐标和 top direction intervention 目前不能稳定控制 unsafe behavior。
