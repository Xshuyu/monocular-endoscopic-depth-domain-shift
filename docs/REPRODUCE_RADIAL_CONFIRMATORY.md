# 径向校准补强实验复现

指定环境：`/ssd-1/temp/kimicode/multmodel/venv-gpu/bin/python`。

## 推荐：保留原结果的独立目录复跑

```bash
cd /ssd-1/temp/kimicode/multmodel/lesion_3d_measurement
/ssd-1/temp/kimicode/multmodel/venv-gpu/bin/python scripts/reproduce_radial_confirmatory.py --destination /tmp/radial_replay_01 --run
```

目标目录必须不存在。省略 `--run` 只生成隔离目录及命令清单。共享本地数据、权重和现有开发集骨干预测；会在独立输出目录重新拟合、重新推理新测试 RGB 并评分，不覆盖本轮结果。数据已被观察，复跑不能称新盲测。

## 本轮实际流程

1. 先保存 `outputs/radial_confirmatory/protocol.json`（候选、采样、序列名、终点和分层规则）。
2. `scripts/run_radial_confirmatory.py development`：6×5 嵌套 LOSO，四种校准家族，固定总预算 16384。保存拟合系数、训练均值/尺度、候选分数和冻结 SHA。
3. `scripts/run_radial_confirmatory.py acquire`：冻结后从官方 ZIP 取四序列各12帧，CRC/SHA校验，保存选取/下载清单。完整 ZIP 不下载。复跑入口复用这些文件，不必再次下载。
4. `scripts/run_radial_confirmatory.py inference`：哈希检查本地 Small 权重、检查图像重复；只读取 RGB 推理，不用 GT 选参。
5. `scripts/run_radial_confirmatory.py final`：验证冻结代码哈希，保存评分开始记录；仅一次最终评分，已存在 test_summary.json 时拒绝覆盖。
6. `scripts/analyze_radial_confirmatory.py`：逐视频配对与半径×深度分析，无重拟合。
7. `scripts/revise_cmig_radial.py`：读取结果生成修订稿/投稿信/亮点。正文针对本轮解释，未来改变实验后须人工同步审阅正文，不能只更新主表。
8. `/home/suyuxiong/miniforge3/bin/tectonic manuscript_cmig/main_cmig.tex --keep-logs`：编译 PDF。

所有 Python 命令都以前述 venv 可执行文件运行。分析使用固定种子20260927，网络不训练；5个解析/单位测试验证参数恢复、正则化收缩、输出边界及尺寸接口。环境和源码快照随产物留档。

## 拟合定义

两家族分别用 d 和 (d, d*r², r²) 作为非截距特征。训练数据场景均权标准化，最小化场景平均逆深度平方误差，加 λ 倍标准化非截距系数平方和；截距不惩罚。正则化网格为1e-8至1e-1的8个十倍步长，LS固定0。内层以深度MAE选参。该定义不是原稿所称、但缺乏运行证据的“原始特征仅惩罚径向项 λ=1e-4”。

## 尺寸接口

```bash
/ssd-1/temp/kimicode/multmodel/venv-gpu/bin/python scripts/predict_radial_measurement.py --relative native_relative.npy --heads outputs/radial_confirmatory/frozen_heads.json --method radial_ridge --mask lesion_mask.npy --rays calibrated_rays.npy --output /tmp/radial_measurement_case01
```

上述三个 `.npy` 是实际输入要求，不是项目已存在文件。射线须来自对应相机标定并匹配图像网格。输出为研究用途的轴向深度和MAD-PCA 1–99%范围，不是临床卡尺真值。真实物理/临床尺寸验证仍需采集数据，见 `RADIAL_PHYSICAL_SIZE_VALIDATION_PROTOCOL.md`。
