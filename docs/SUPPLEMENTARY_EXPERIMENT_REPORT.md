# Supplementary Controlled Experiments

运行脚本：`tests/supplementary_benchmark.py`

输出目录：`outputs/supplementary/`

## 1. 多随机种子离群鲁棒性

使用 10 个随机种子、椭球真值 `24 x 14 x 6 mm`，加入 0--20% 空间离群点。每个条件保存逐例结果。

| 离群比例 | PCA MAE (mm) | Robust OBB MAE (mm) |
|---:|---:|---:|
| 0% | 1.14 | 4.68 |
| 5% | 57.10 | 4.03 |
| 10% | 65.00 | 3.62 |
| 15% | 67.63 | 3.51 |
| 20% | 67.86 | 3.71 |

结论：Robust OBB 的主要优势是污染后的稳定性，而不是无噪声时的 full-extent accuracy。论文应同时报告 clean-case 偏差，不能只报告噪声下的改善倍数。

## 2. 工具尺度与 mask 敏感性

在 20、30、45 mm 工作距离，工具投影宽度误差为 -20% 至 +20%，并测试 clean、eroded、dilated 三种 mask。结果保存在 `scale_tool_trials.csv`。

全条件平均相对尺度误差为约 74.0%，其中距离增加和 mask/宽度偏差会显著放大误差。该结果说明当前基于最小外接矩形的工具尺度模型对细长工具 mask 非常敏感，不能作为可靠的临床绝对标尺。

## 3. 解析曲面曲率

半径 5、10、20 mm 球面和法向噪声平面均保存逐例结果。无噪声半径 10 mm 球面可恢复 `H≈0.092 mm^-1`、`K≈0.0092 mm^-2`，但加入很小的法向噪声后 `K` 明显不稳定。这是当前曲率模块的明确限制：需要法向平滑、鲁棒拟合和噪声敏感性校准后，才能用于临床形态学解释。

## 4. pipeline 尺度对比

在已知深度的程序化场景中比较 tool scale 与 prior-only scale。当前实现的平均 major-diameter 误差约为：tool `10.24 mm`，prior `11.91 mm`。两者均不满足临床测量要求，且该实验仅用于定位尺度误差，不构成患者数据验证。

## 5. 投稿影响

新增实验不支持“已解决 monocular scale ambiguity”的表述。建议将尺度模块定位为实验性 scale cue，并在 Limitations 中明确：工具 mask 误差、工具姿态、透视畸变和细长目标外接矩形都会造成明显误差。
