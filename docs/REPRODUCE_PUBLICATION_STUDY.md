# 本轮实验复现

本次已在 `/ssd-1/temp/kimicode/multmodel/venv-gpu` 实际执行。结果位置 `outputs/publication_study`。不要删除冻结文件后在原目录重跑；最终集已被观察，任何复跑都是复现，不再是新的盲测。

## 独立目录复跑

以下入口复制源代码，共享现有数据、权重和旧 Small 原始预测缓存，不覆盖原始结果。需 CUDA 显卡、完整现有数据/模型文件以及下述辅助依赖。

```bash
cd /ssd-1/temp/kimicode/multmodel/lesion_3d_measurement
/ssd-1/temp/kimicode/multmodel/venv-gpu/bin/python scripts/reproduce_publication_study.py --destination /tmp/publication_replay_01 --run
```

省略 `--run` 只准备目录并写入 replay.json。目标目录必须不存在。辅助依赖安装位置为 `/tmp/lesion_publication_deps`，清理 /tmp 后需恢复：

```bash
/ssd-1/temp/kimicode/multmodel/venv-gpu/bin/python -m pip install --target /tmp/lesion_publication_deps --no-deps -r outputs/publication_study/aux_requirements.txt
```

这些辅助包支持 EndoDAC 原始模块导入；未替换指定 venv 的 Torch/NumPy。完整版本快照见 `outputs/publication_study/environment.json`。GPU 内核及跨版本差异可能带来数值漂移，不承诺跨硬件逐 bit 一致。

## 已实际执行的顺序

在尚无冻结结果的新工作目录，核心命令为：

```bash
/ssd-1/temp/kimicode/multmodel/venv-gpu/bin/python scripts/publication_inference.py --phase development
/ssd-1/temp/kimicode/multmodel/venv-gpu/bin/python scripts/publication_calibration_study.py --phase development
/ssd-1/temp/kimicode/multmodel/venv-gpu/bin/python scripts/publication_inference.py --phase final
/ssd-1/temp/kimicode/multmodel/venv-gpu/bin/python scripts/publication_calibration_study.py --phase final
/ssd-1/temp/kimicode/multmodel/venv-gpu/bin/python scripts/build_publication_analysis.py
```

开发执行结束生成 frozen_heads.json，最终执行验证源码 SHA；存在冻结头/最终结果时拒绝覆盖。开发 Small 预测复用既有冻结骨干缓存；Large、EndoDAC 与最终集经过实际推理。`publication_anchor_sensitivity.py` 是冻结后开发集探索，不影响最终模型。

## 数据和权重来源

- C3VD 官方：https://durrlab.github.io/C3VD/ 。本轮三个序列的选取、下载地址、每个成员 CRC/SHA 在 `*_selection.json` 与 `*_manifest.json`。cecum 数字 ID 配对变更见 selection_amendment.json。
- C3VDv2：现有 native_subset 与 robust_development 的 download_manifest.json；固定镜像版本与本地文件指纹可追踪，但未与不可访问的原仓库逐字节比对。
- EndoDAC：https://github.com/BeileiCui/EndoDAC ，源码固定提交 22ec911b270d056fdd7f430474e4be407cce8287，权重及 SHA 在 datasets/external_models/EndoDAC/manifest.json；加载和预处理检查已留档。
- Small/Large 使用项目现有本地 Depth Anything V2 权重。数据/模型未复制进源码快照；共享目录是复跑的必要输入。

## 论文生成

```bash
/ssd-1/temp/kimicode/multmodel/venv-gpu/bin/python scripts/write_calibration_manuscript.py
/home/suyuxiong/miniforge3/bin/tectonic manuscript/calibration_generalization.tex --keep-logs
```

生成 `manuscript/calibration_generalization.pdf`。论文主表读取最终 JSON，正文为审阅过的本轮叙述；新实验后必须同步复核正文，不能只刷新表格。
