# 论文实验完善报告

本报告对应 `outputs/paper_strengthening/` 的固定协议运行，不混用旧版主基准数字。

## 已实施

- 主场景从单种子扩展到 10 个预先写入协议的种子，共 320 个场景、7680 条方法结果。
- 新增两组严格匹配过滤策略和测量分位数的 PCA / SO(3) 比较；另保留原始自适应策略作探索比较。
- 逐例预测、按尺寸分层、混淆矩阵、各档召回率、CCC、真正的预测 R²、Bland–Altman 偏差和界限均有输出。
- 不确定性以整批随机种子为重采样单位；配对检验基于种子内平均误差差，12项比较统一Holm校正。区间本身未作多重比较校正。
- 场景生成使用局部随机数生成器，默认种子保持原场景不变，不重置调用者的全局随机状态。
- 修正旧统计函数对ICC的错误说明，保留接口兼容。没有把CCC称为标注一致性。
- 保留原始器械数据与划分，另输出建议去重划分（从训练集移除跨集完全重复图像，测试集不变）。未训练模型。
- 保存执行前协议、逐例结果、软件版本、代码与输出哈希和源代码快照。

## 结果：主径 MAE（mm）

| Depth/scale condition | PCA full | PCA q01–99 | MAD–PCA q01–99 | SO(3) no MAD | SO(3) always MAD | SO(3) adaptive MAD |
|---|---:|---:|---:|---:|---:|---:|
| oracle_depth | 0.560 | 0.656 | 0.730 | 0.716 | 0.803 | 0.800 |
| prior | 4.079 | 4.543 | 4.688 | 4.630 | 4.763 | 4.754 |
| tool | 12.727 | 11.309 | 10.853 | 11.054 | 10.628 | 10.656 |
| fusion | 4.368 | 3.413 | 3.105 | 3.253 | 2.986 | 3.006 |

## 匹配条件下的几何比较

| Scale | Matched filter | SO(3) − PCA MAE (mm) | 95% interval | Holm p |
|---|---|---:|---|---:|
| oracle_depth | matched_no_filter | 0.060 | [0.046, 0.074] | 0.0234 |
| oracle_depth | matched_radial_MAD | 0.072 | [0.059, 0.084] | 0.0234 |
| prior | matched_no_filter | 0.087 | [0.075, 0.098] | 0.0234 |
| prior | matched_radial_MAD | 0.075 | [0.063, 0.087] | 0.0234 |
| tool | matched_no_filter | -0.255 | [-0.290, -0.222] | 0.0234 |
| tool | matched_radial_MAD | -0.225 | [-0.262, -0.187] | 0.0234 |
| fusion | matched_no_filter | -0.160 | [-0.184, -0.134] | 0.0234 |
| fusion | matched_radial_MAD | -0.119 | [-0.144, -0.094] | 0.0234 |

负值表示SO(3)误差更低；所有结果必须结合尺度条件解释，不因显著性筛选展示。

## 数据及真值边界

器械数据审计发现 3 组训练/测试完全重复图像。建议划分仅解决字节相同的重复，不保证患者/视频级独立。

渲染表面的最大三维弦长与名义主径平均绝对差为 0.561 mm。两者是不同测量定义，本项用于揭示真值语义差异，未用其重新标定或修正算法。

主实验仍保留旧生成器的形态—尺寸混杂、共享光度假设和真值掩膜。多种子不能消除这些设计局限。Oracle还更换深度来源，不能解释为纯尺度因子。未获得真实病灶物理尺寸、多标注者重复测量或外部临床验证。

## 复现

从项目目录执行（新输出目录必须不存在）：

```bash
/ssd-1/temp/kimicode/multmodel/venv-gpu/bin/python -B tests/paper_strengthening_benchmark.py --output /tmp/lesion_paper_repeat
```

渲染报告：

```bash
/ssd-1/temp/kimicode/multmodel/venv-gpu/bin/python -B scripts/build_paper_strengthening_report.py
```

默认运行仅使用CPU光度回退路径，不下载权重、不训练。运行时约 8.3 分钟（本机本次结果）。每个方法使用同样的场景、掩膜和尺度输入，未进行模型选择。

`source_snapshot/lesion_3d_measurement/tests/paper_strengthening_benchmark.py` 保存了本次代码，可在以后代码改变时用于重放；数据链接仅供独立数据审计，主测量数据由种子生成。

## 仍需真实资源的工作

见 `docs/PHYSICAL_VALIDATION_PROTOCOL.md`。没有用合成数据替代医生标注ICC、真实毫米真值或患者级外部验证。当前主文是以本次数据为依据的修订稿，投稿前仍需完成相关工作引用、作者声明核实及真实验证决策。
