# 多形状泛化基准报告

运行脚本：`tests/generalization_benchmark.py`

固定边界修正系数：`1.0`。测试共 720 个独立案例，包含 ellipsoid、dome、plaque、saddle 四类形状，low/medium/high 三种点密度，0/5/10/20% 离群点，0/30/60 度随机姿态，每个条件 5 个随机重复。

## 关键结果

固定修正对椭球有效，但存在明显形状依赖和噪声下过校正：

| Shape | Noise | Baseline MAE | Adaptive MAE |
|---|---:|---:|---:|
| Ellipsoid | 0% | 3.96 | 2.12 |
| Ellipsoid | 10% | 3.12 | 1.26 |
| Dome | 10% | 0.76 | 2.04 |
| Dome | 20% | 0.93 | 6.37 |
| Plaque | 10% | 0.51 | 2.10 |
| Plaque | 20% | 0.42 | 5.03 |
| Saddle | 10% | 0.37 | 1.82 |
| Saddle | 20% | 0.62 | 7.00 |

这里的 Adaptive MAE 使用 tail-ratio 触发的边界修正，并不是新的训练模型。它仍在非椭球形状和高噪声下明显劣化，说明单一边界修正系数不能泛化到不同病灶形态。

## 决策

不将 boundary-aware correction 替换为论文主方法，也不将其结果写入主结果表。保留代码接口和扩展基准，作为失败的泛化尝试和后续工作依据。当前主方法继续使用未修正的 Robust OBB，论文只报告其离群鲁棒性。
