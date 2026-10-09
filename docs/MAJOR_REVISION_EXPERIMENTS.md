# Major Revision 实验与手稿修订报告

本次补充已实际运行；不是仅增加引用或实验计划。所有 Python 检查、训练与推理使用 `/ssd-1/temp/kimicode/multmodel/venv-gpu/bin/python`。处理附加依赖放在项目隔离目录。历史冻结算法、标定头和结果未覆盖。

## 三项要求的完成程度

| 要求 | 实际补充 | 仍有边界 |
|---|---|---|
| 至少一个独立训练的域适应单目基线 | 从通用 DA V2 Small 初始化，独立训练监督式度量深度解码器；源域/适配两种条件 × 3 seeds，共 6 个模型 | 是低数据量监督式适配，不是自监督 EndoDAC 复现，也不是已经证明很强的 SOTA；通用预训练无法穷尽审计 |
| 增加独立对象/设备数量 | 新增 DRENDS：5 类仿真器官实体组、4 类离体器官准备组，27 个固定首帧；两套公开相机配置 | 供体 ID、设备序列号缺失，因此不能声称增加 9 个独立供体或 2 台可核实独立设备；这项只部分解决 |
| 更受控的迁移失败分析 | 同一图像亮度×模糊 2×2 配对、100/300 mm 输出范围、参考深度范围分层、真值辅助偏移/尺度诊断、参考滤波单位敏感性 | 不能单独识别真实设备、材质、解剖与照明的物理因果贡献；真实配对设备实验仍缺 |

## 独立训练协议

- 冻结通用 DA V2 Small 骨干 22,056,576 参数；训练 neck/head 2,728,513 参数，最后标量卷积重新初始化为 log-mm 预测。没有继承任何 SCARED 适配的 EndoDAC/PCC 神经网络权重。
- 源域：原 52 帧 C3VD 开发数据。目标适配：SCARED 1、2、3、6 号对象，各 5 个实测关键帧，共 20 帧。
- SCARED 7 号未使用；8、9 号不参与训练或模型选择，测试共 10 帧。旧冻结模型的七对象表与新适配模型的两对象表不可混为一谈。
- seeds=11、23、47；每个模型 600 updates，batch=2，AdamW lr=1e-4、weight_decay=.01、梯度裁剪1。取最终模型，不按测试/验证分数择优。
- 源域对照每步2张源域图；适配每步1张源域+1张目标域图。总更新量相同，但源域曝光量减半并引入目标域标签。因此识别的是固定预算下的监督分配变化，不能称为“无额外监督”的纯算法收益。
- 224×280 输入，官方归一化；训练 GT/mask 最近邻缩放。每图 log 深度 SmoothL1(beta=.1)，再对两图平均。
- 统计为先图像/条件、再对象组、再随机种子。种子 SD 不是患者置信区间；四个 C3VD 视频也不是四个独立患者。
- 与四参数标定比较时，明确网络训练使用更多标注像素和参数，不能宣称等参数/等像素预算。

## 实测主要结果

MAE，mm；训练模型为三随机种子均值 ± 样本 SD。主分析保留所有有效 GT，预测共同限制1–100 mm，不做测试真值缩放。

| 测试集 | 原 Small 径向正则 | 独立训练，仅源域 | 独立训练，域适配 |
|---|---:|---:|---:|
| C3VD，4视频/48帧 | 8.899 | 8.021 ± 1.095 | 9.085 ± 3.710 |
| SCARED，2留出对象/10帧 | 37.115 | 36.718 ± 1.831 | 13.666 ± 1.836 |
| SERV-CT，2标本/16视图 | 55.537 | 55.368 ± 2.351 | 22.212 ± 1.607 |
| DRENDS，5仿真器官组/15首帧 | 111.041 | 114.232 ± 1.805 | 93.784 ± 2.101 |
| DRENDS，4离体器官组/12首帧 | 100.131 | 99.157 ± 2.278 | 57.218 ± 4.548 |

结论：目标域监督明显降低了观察到的 SCARED/SERV-CT 平均误差；未宣称统计显著性；C3VD 平均表现回退且适配的种子波动较大。新 DRENDS 仍存在严重迁移失败，不支持“设备无关的准确毫米深度”。没有把改善幅度转述为临床尺寸精度。

## 新对象与设备证据

公开来源：[DRENDS 数据](https://zenodo.org/records/17598453)、[官方处理代码](https://github.com/L0za007/DRENDS)、[2026 年论文](https://doi.org/10.1038/s41597-026-07775-2)。下载 DRENDS.zip、Phantom_Sequences.zip、ExVivo_Sequences.zip，核对官方 MD5。按模态分目录解压；同名序列不会在正式分析中混用。官方处理固定提交为 077ec79fd4c414508bea1de1ef1e14b94911c942。

5 个仿真组为 Colon/Intestine/Stomach/Liver/Pancreas；4 个离体组为 Colon/Intestine/Stomach/Pancreas。每组 Ext/Med/High 三个照明序列仅取首帧。器官/配置是统计分组，不当作供体身份。公开设备描述为 da Vinci Si 下 Ikegami 720p 与 Panasonic 1080i 两套相机配置；缺少序列号，不能证明两台新增独立设备，也无法确认跨论文供体完全独立。

首帧选择在评分前固定，27个全部保留。Seq_Pattern 是标定棋盘，不属于预定27个实验序列，已在协议实施修订中记录排除原因。首帧结果不代表非刚体时序鲁棒性。

## 新参考深度的可靠性与差异

按作者的标定、补偿变换、workspace、normal/view过滤及5 mm遮挡阈值，将ToF点云映射至整流左相机。原始毫米XYZ先转米，再计算相机轴向Z，最后输出毫米。

SAM2 Hiera Large使用作者首帧提示，通过image predictor生成工具掩膜；未声称精确复现其视频传播。CPU raycasting代替Linux离屏渲染，使用未归一化、z=1的射线，避免将欧氏射线距离误认为轴向Z。已通过2米平面几何测试，并检查全部27帧参考叠加图。该几何测试不等于真实采集标定精度验证。

官方滤波sigma参数命名为mm，但计算坐标为米。主分析保留官方代码设置；另以0.005米解释重建参考。在共同有效像素上，27帧平均参考差0.471 mm，单帧平均差最高0.974 mm；六模型各帧MAE变化最大0.687 mm。两种掩膜覆盖率也单独保存，不能把共同支持差异当作全部参考不确定性。工具提示主观性、ToF误差与配准残差仍未完全排除。

## 受控因子分析

1. **亮度×模糊**：gamma=1/1.5，Gaussian sigma=0/1.5 native pixels；同图、同GT、同mask，六模型全部运行。适配模型暗化主效应约 C3VD +0.10、SCARED +1.65、SERV-CT +2.65 mm。模糊很轻，尤其经过输入缩放后，不能由小效应推断一般模糊鲁棒性。该干预不等价于物理光照实验。
2. **输出上限**：原径向正则模型SCARED/SERV-CT从100放宽300 mm后MAE不变；因此不能将其55 mm量级失败解释为截断。适配模型DRENDS放宽后仍为93.765/57.109 mm，结论不变。
3. **GT支持范围**：另存1<GT<100 mm支持集，主分析始终保留全部有效GT，避免删掉深部难例获得更好数字。
4. **偏移/尺度成分**：原模型加入逐帧真值中位数偏移后，SCARED 37.115→9.900、SERV-CT 55.537→12.584 mm。此为不可部署的诊断，只说明误差中有较大全局偏移成分；不是新模型结果，也不证明偏移的物理成因。

## 复现入口

使用指定Python，按以下顺序；冻结/训练文件保留，不删除后重做协议：

```bash
/ssd-1/temp/kimicode/multmodel/venv-gpu/bin/python scripts/train_major_revision_baseline.py freeze
/ssd-1/temp/kimicode/multmodel/venv-gpu/bin/python scripts/train_major_revision_baseline.py cache
CUBLAS_WORKSPACE_CONFIG=:4096:8 /ssd-1/temp/kimicode/multmodel/venv-gpu/bin/python scripts/train_major_revision_baseline.py train
/ssd-1/temp/kimicode/multmodel/venv-gpu/bin/python scripts/acquire_major_revision.py
PYTHONPATH=/tmp/lesion_publication_deps /ssd-1/temp/kimicode/multmodel/venv-gpu/bin/python scripts/prepare_drends_firstframes.py Phantom
PYTHONPATH=/tmp/lesion_publication_deps /ssd-1/temp/kimicode/multmodel/venv-gpu/bin/python scripts/prepare_drends_firstframes.py ExVivo
/ssd-1/temp/kimicode/multmodel/venv-gpu/bin/python scripts/evaluate_major_revision.py features
/ssd-1/temp/kimicode/multmodel/venv-gpu/bin/python scripts/evaluate_major_revision.py inference
/ssd-1/temp/kimicode/multmodel/venv-gpu/bin/python scripts/evaluate_major_revision.py score
/ssd-1/temp/kimicode/multmodel/venv-gpu/bin/python scripts/diagnose_original_transfer.py
PYTHONPATH=/tmp/lesion_publication_deps /ssd-1/temp/kimicode/multmodel/venv-gpu/bin/python scripts/audit_drends_reference_sensitivity.py
/ssd-1/temp/kimicode/multmodel/venv-gpu/bin/python scripts/check_major_revision.py
/ssd-1/temp/kimicode/multmodel/venv-gpu/bin/python scripts/report_major_revision.py
```

当前工作目录已有freeze协议，第一条故意拒绝覆盖；已缓存阶段可复核运行。新机器需按随附环境清单恢复通用模型、SAM2权重、官方数据及项目隔离依赖。checkpoint有哈希，不含在轻量手稿包内。

证据目录 `outputs/major_revision/`：training_protocol.json、trained_models/、training_complete.json、factor_protocol.json、drends_processing_protocol.json、metric_per_frame.csv、metric_per_object.json、metric_summary.json、original_transfer_diagnostics.csv、reference_sensitivity.json、reference_model_sensitivity.json、validation_checks.json；两幅主图有PDF/SVG可编辑版本。

## 对 Major Revision 的诚实判断

已实质加强：独立训练而非借用有重叠的域适配权重；同更新预算的三种子对照；新增另一公开采集来源与多个器官组；可复核的控制实验与失败结果；摘要/正文/结论同步收窄。

仍需审稿人接受的范围：这是“校准迁移、有限监督适配与失败机制诊断”研究，而非新的通用SOTA或已验证的临床病灶测量系统。只有两个留出SCARED对象、两个SERV-CT标本，DRENDS没有可核实供体/设备序列号；需要更多有身份记录的对象及同一物理对象跨设备配对数据，才能完全解决独立性和设备因果归因。不能保证编辑给出Major Revision，更不能将此报告当成真实审稿意见。
