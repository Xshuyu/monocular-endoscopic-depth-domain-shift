# 改进方法独立基准报告

运行脚本：`tests/controlled_improvement_benchmark.py`

## 实验设计

- 训练种子：0--19
- 验证种子：20--29
- 测试种子：30--49
- 测试尺寸：8、12、20、28、35 mm 等多尺度椭球
- 测试条件：clean 和 10% spatial outlier
- 边界修正系数只在 validation split 上选择，未使用 test split。

## 结果

验证集选择的边界修正系数为 `1.0`，处于预设搜索网格的上界，因此该参数应视为数据集相关的候选参数，而不是通用常数。

| 测试条件 | Baseline OBB MAE | Boundary-aware OBB MAE |
|---|---:|---:|
| Clean | 3.76 mm | 2.13 mm |
| 10% outlier | 3.30 mm | 1.06 mm |

## 解释

在当前程序化椭球基准上，边界修正改善了独立测试种子的误差；但最优系数触及搜索边界，说明仍可能存在模型失配。该结果不能直接外推到真实内镜数据，也不能称为已超过外部 SOTA。

新增实现还包括：

- `MetricScaleCalibrator.calibrate_adaptive()`：根据工具 mask 质量自适应融合 tool scale 与 prior scale。
- `MetricScaleCalibrator.estimate_tool_quality()`：使用面积、细长度、连通性和深度变异估计工具质量。
- `estimate_with_mask_uncertainty()`：对 eroded/original/dilated mask 输出测量中位数、区间和敏感性。
- `compute_robust_obb_lie(..., boundary_correction=...)`：支持验证集确定的边界修正。

这些新增能力尚未替换主论文 benchmark 的默认输出，因此正式稿件不能把本报告结果与原始 50 例表格混写。下一步应在固定验证集选定参数后，锁定参数并重新生成完整主表。
