# CBM 实验审计与修订清单

## 当前结论

当前结果不能直接作为投稿结果。已有 JSON 和图表是历史运行产物，但本环境缺少 `pytest` 和 `opencv-python`，且若干数字无法由当前代码独立复现。论文应在重新运行完整基准并保存逐例预测后再提交。

## 已确认的问题

1. `tests/batch_benchmark.py` 的真实数据实验显式调用 `num_samples=50`，因此不能写成 Kvasir-SEG `N=1000` 的定量评测。Kvasir-SEG 的数据集规模可以作为数据集描述，但实际评测样本数必须按有效配对样本报告。
2. `src/evaluation_suite.py` 原先对 Baseline 3、Ablation 1 和 Ablation 2 的预测结果乘以 `1.15`、`1.10` 和 `0.90`。这些人为校正常数不是算法定义，会造成消融结果不可辩护。本次已删除。
3. `src/heat_geodesic.py` 在 Varadhan 热场后额外加入固定的 `0.15 * max_euclidean` 经验项。这不是标准 Heat Method 的 Poisson 整合步骤，不能宣称为严格的 Laplace--Beltrami geodesic。重新投稿前应实现标准 heat method，或将模块和论文改称为 heat-inspired graph distance。
4. 曲率基准不符合解析真值。半径 `R=10 mm` 的球面应满足 `|H|=0.1 mm^-1`、`K=0.01 mm^-2`；已有结果 `H=-0.7903`、`K=1.1105` 不能作为验证通过的证据。需要报告偏差、边界效应和噪声敏感性，而不是只报告 Paris 映射。
5. 合成 cohort 的 RGB、深度和 mask 由同一渲染器生成，且相机 FOV、工具直径和深度都由生成器控制。这只能证明内部一致性，不能称为 in-vivo 或 caliper-calibrated clinical cohort。
6. `N=50` 的合成样本中每四个样本循环一次形态类型，Paris 标签不是临床专家标注；不能据此宣称自动 Paris 分类的临床有效性。
7. 当前 p 值来自对绝对误差的配对 t 检验，且未给出逐例预测、效应量、多重比较校正或预先规定的主要终点。应补充 Wilcoxon、95% bootstrap CI、配对效应量和 Holm 校正。
8. 当前代码把未标定的相对深度通过工作距离先验直接转换为毫米，不能称为真实 metric depth。应将它明确写作 pseudo-metric/prior-scaled depth，并增加跨相机、跨距离和工具缺失场景的敏感性分析。
9. 补充实验表明工具尺度 cue 对工作距离、投影宽度量化和 mask 腐蚀/膨胀高度敏感；在当前实现中不能宣称解决 monocular scale ambiguity。
10. 补充实验表明 Robust OBB 的优势主要出现在离群污染下；无噪声时 PCA 的 full-extent MAE 更低，论文应报告 bias--robustness trade-off。
11. 半径球面无噪声时曲率估计接近解析值，但微小法向噪声会显著放大 Gaussian curvature 误差，曲率模块仍不能支撑临床 Paris 分类。
12. 新增的 leakage-safe 改进基准在独立种子测试集上将 boundary-aware OBB 的 clean MAE 从 `3.76` 降至 `2.13 mm`、10% outlier MAE 从 `3.30` 降至 `1.06 mm`；但验证集选择的修正系数触及搜索上界 `1.0`，仍需在更多尺寸、形状和真实体模上验证。
13. 720 例多形状泛化实验显示，固定或 tail-ratio 自适应边界修正在 dome、plaque 和 saddle 的高噪声条件下过校正，不能替换主方法。该改进保留为受限探索，不进入论文主结果表。

## 投稿前必须完成

- 安装锁定版本的依赖并执行单元测试、合成基准、完整 Kvasir-SEG 配对评测；保存命令、随机种子、环境和逐例 CSV。
- 对工具检测、病灶分割、深度估计、尺度校准分别报告失败率；不能只对成功样本报告性能。
- 将真实临床数据的任务限定为鲁棒性/运行时间/分布描述，除非有独立的毫米级真值或专家复核。
- 重新实现标准 Heat Method，或删除“连续测地线”“精确反映 ESD 边界”等临床因果表述。
- 将 `outputs/supplementary/` 中的逐例 CSV 和 JSON 作为补充材料候选，并在稿件中报告条件、随机种子和失败模式。
- 以独立患者/视频为单位划分训练、验证和测试集，避免帧级泄漏；合成数据参数也必须只在训练集确定。
- 对 CBM 的在线 Guide for Authors、数据/代码政策、伦理和 AI 使用披露要求，在提交系统中再次核对。出版社页面当前返回 403，不能在本地审计中声称已完成实时网页核验。

## 建议的新主要终点

主要终点设为独立测试集上的 major-diameter MAE（mm）和 95% bootstrap CI；次要终点为 MRE、RMSE、CCC、Bland--Altman LoA、四级风险分类 macro-F1 和失败率。所有数字从逐例预测重新计算，禁止手工填入结果 JSON。
