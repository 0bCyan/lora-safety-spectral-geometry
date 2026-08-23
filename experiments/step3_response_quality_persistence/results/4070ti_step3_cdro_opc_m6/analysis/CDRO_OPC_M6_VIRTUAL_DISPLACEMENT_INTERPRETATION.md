# CDR-OPC M6 有限半径直接虚拟位移解释

## 结论

M6 按前瞻协议在固定剂量 `0.0337860296353059`、固定尺度
`eta=2e-4` 上停止于 calibration：有限保护覆盖为 `16/20`，低于
预注册门 `19/20`。因此路线是
`STOP_CDRO_OPC_M6_CALIBRATION`，sealed audit 未打开。

这不是训练结果，也不是新的外部确认。M6 复用了已经打开的 M5 cohort，
只提供固定半径的有限损失开发证据。全过程
`optimizer_steps=0`、`training_authorized=false`。

## 通过与失败的门

固定方向重放、求解器、实际位移精度、梯度覆盖、有限值、参数/梯度/RNG/
模式恢复、行为门和 selected-vs-fallback 四证书平均相对恶化门全部通过。
这里的平均门不是 selected 相对初始点改善。唯一失败门是单元保护覆盖。

四个失败单元均只由 risk 证书触发：

| unit | finite risk deterioration | first-order value | finite minus first-order |
|---|---:|---:|---:|
| `m5_cal_14` | 0.045814 | 0.023349 | 0.022464 |
| `m5_cal_18` | 0.041146 | 0.025417 | 0.015729 |
| `m5_cal_24` | 0.046497 | 0.036368 | 0.010129 |
| `m5_cal_36` | 0.043900 | -0.000454 | 0.044354 |

uniform、TARC 和 continuation 均没有单元超过 `0.04`。risk 的总体均值
恶化只有 `0.004613`，但尾部覆盖门要求逐单元至少 `19/20`，不能用总体
均值掩盖四个越界单元。

## 行为与数值审计

quality 的 selected-over-fallback 实际优势为 `0.000922`，预测值为
`0.001053`，比率 `0.8756`；safety 的实际优势为 `0.000205`，预测值为
`0.000170`，比率 `1.2039`。两组比较门均通过。

这里的 quality 结论只表示 selected 比 fallback 好；两者的 surrogate loss
都高于初始点，不能解释为 selected 相对初始点改善。safety 的 fallback 和
selected 均优于初始点，且 selected 更好。

实际写入的相对 L2 映射误差约为 `1.18e-6`，最大绝对映射误差
`4.71e-6`，均低于冻结门。最终参数哈希、张量、空 optimizer state、梯度、
RNG 和 eval mode 全部恢复。

## 可支持和不可支持的主张

结果支持：M5 一阶方向在该固定半径上保持了 selected-vs-fallback 四证书
平均相对恶化门和相对行为优势，但没有保持预注册的 risk 尾部覆盖。

M5 已知的 sealed 尾部也没有被消除：其覆盖仍是 `19/20`，失败单元
`m5_audit_27` 的 risk 恶化为 `0.070039`。M6 没有打开 sealed audit，
因此没有新的有限损失证据可以改写这一限制。

结果不支持训练、checkpoint、生成、Core100、heldout 或 sealed finite-loss
结论；也不能据此声称更小半径一定通过，或声称 CDR-OPC 在所有半径都失败。
协议禁止结果依赖的缩半径、调剂量或重跑。任何后续曲率或 risk-tail 诊断
都必须另行前瞻冻结。
