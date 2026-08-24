# Step3 CDR-OPC M7 固定半径镜像分解诊断计划

## 1. 登记时点与定位

本计划在 M6 结果已经封存、任何 M7 正向镜像参数写入发生之前登记。

M6 的结论保持不变：

- 路线为 STOP_CDRO_OPC_M6_CALIBRATION；
- calibration 仅覆盖 16/20；
- 已知四个失败单元为 m5_cal_14、m5_cal_18、m5_cal_24、m5_cal_36，均由 risk 证书触发；
- sealed audit 未打开；
- 不允许根据结果缩小半径、修改剂量、重跑 M6 或授权训练。

M7 只是 post-M6、固定半径、validity-only 的对称有限差分诊断。它解释 M6 已观察到的单边有限半径恶化，不是独立确认、救援实验或因果归因。

## 2. 数据与写入边界

M7 只使用 M6 已打开的 20 个 calibration 单元及四个保护证书：

- risk；
- uniform；
- tarc；
- continuation。

协议冻结 unit/sample allow-list，并冻结 sealed unit ID 哈希。新损失评价器只接收 calibration allow-list，输出 ID 必须与 allow-list 精确相等且与 sealed 集合无交集。

固定参数：

- fixed dose = 0.0337860296353059；
- eta = 0.0002；
- score floor = 0.02；
- fallback 与 selected 方向、特征、损失函数和单位顺序全部继承 M6。

唯一新增损失测量是：

- 20 次 theta0 + eta*d_F；
- 20 次 theta0 + eta*d_S。

共 40 次正向镜像损失写入。另有 fallback/selected 各一次无损失 map precheck，共 2 次。负向臂重算、endpoint、behavior、sealed、Core100、heldout、checkpoint、generation 和 optimizer step 均固定为 0。

## 3. 先源重放、后镜像

在任何正向镜像写入前，先一次性重放全部 20 个 calibration 单元的基线和梯度：

1. 重建固定方向和 reference norm；
2. 逐单元重算 fresh L0、原始梯度范数、四个证书范数与 fallback/selected 一阶分数；
3. 对照冻结的 M6 行、unit/sample 顺序及 coverage 结构；
4. 全部通过后才允许 2 次 map precheck 和 40 次正向镜像损失测量。

预冻结重放容差：

- loss：atol = 1e-6，rtol = 1e-6；
- norm：atol = 1e-8，rtol = 1e-6；
- first-order score：atol = 1e-8，rtol = 1e-6；
- 源算术与分解恒等式：1e-10。

fresh L0 只用于源重放审计。主分解使用 M6 冻结的 L0、Lminus、norm、reference norm 和 first-order score，防止悄然改写 M6 的观测量。

## 4. 冻结数学分解

对单元 i、证书 j、方向 A 属于 {F,S}，令：

\[
q_{ij}=\eta\lVert g_{ij}\rVert R
\]

\[
L^0_{ij}=L_{ij}(\theta_0),\quad
L^-_{ij,A}=L_{ij}(\theta_0-\eta d_A),\quad
L^+_{ij,A}=L_{ij}(\theta_0+\eta d_A)
\]

其中 L0 与 Lminus 只读自 M6，Lplus 是 M7 唯一新测量。

定义 central/odd 分量：

\[
C_{ij,A}=\frac{L^+_{ij,A}-L^-_{ij,A}}{2q_{ij}}
\]

定义固定半径中心偶对称分量：

\[
E_{ij,A}=\frac{L^+_{ij,A}+L^-_{ij,A}-2L^0_{ij}}{2q_{ij}}
\]

M6 单边实际分数满足：

\[
a^-_{ij,A}=\frac{L^0_{ij}-L^-_{ij,A}}{q_{ij}}
=C_{ij,A}-E_{ij,A}
\]

令：

\[
h_{ij}=\max(|b_{ij,F}|,0.02)
\]

则：

\[
D^{M6}_{ij}
=D^{FO}_{ij}+D^{odd-rem}_{ij}+D^{even}_{ij}
\]

其中：

\[
D^{FO}_{ij}=\frac{b_{ij,F}-b_{ij,S}}{h_{ij}}
\]

\[
D^{odd-rem}_{ij}
=\frac{(C_{ij,F}-b_{ij,F})-(C_{ij,S}-b_{ij,S})}{h_{ij}}
\]

\[
D^{even}_{ij}=\frac{E_{ij,S}-E_{ij,F}}{h_{ij}}
\]

E 只能称作“固定半径偶对称分量”或“中心二阶差分代理”，不能称作纯 Hessian、局部曲率或因果机制。

## 5. 完整性与路线

每个正向镜像臂显式执行 theta ← theta0 + eta*d_A，map 审计使用 (theta-theta0)/eta。门限沿用 M6：

- relative L2 error <= 2e-6；
- maximum absolute error <= 1e-5。

每臂以 try/finally 恢复并核验：

- trainable parameters；
- gradients；
- CPU 与全部 CUDA RNG；
- 每个 module 的 train/eval mode；
- named buffers。

optimizer.step 被硬拦截，optimizer state 必须始终为空。

路线只由输入、源重放、map/恢复、算术、完备性和最终完整性决定。任何分量的正负、大小、失败单元数量或 0.04 阈值关系均不参与路线。

可能路线：

1. STOP_CDRO_OPC_M7_INVALID_INPUT
2. STOP_CDRO_OPC_M7_SOURCE_REPLAY
3. STOP_CDRO_OPC_M7_MIRROR_INTEGRITY
4. STOP_CDRO_OPC_M7_ALGEBRA
5. STOP_CDRO_OPC_M7_FINAL_INTEGRITY
6. PASS_CDRO_OPC_M7_VALIDITY_ONLY_SYMMETRIC_DIAGNOSTIC

前五项是运行时停止条件：触发时不封存 decision 或任何部分结果，也不得重试。只有 PASS 会写出完整、可由 verify-run 重算的产物。PASS 只表示 80 个单元—证书格的名义镜像分解在登记映射误差界内完整有效，不表示保护通过。

## 6. 预登记输出

M7 输出：

- decision、diagnostics、environment 与 reproduction manifest；
- 80 行 attribution CSV；
- 全证书聚合描述；
- 四个已知 M6 risk 失败格的 post-hoc 加法分解；
- risk 分解 SVG 与 300 DPI PNG；
- 中英文复现与解释说明。

图表只作描述，不作为 gate。结果无论符号如何，都不得推翻 M6 STOP、推断更小半径会通过，或授权训练、生成、checkpoint、sealed、Core100/heldout。
