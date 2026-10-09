# SCARED 官方测试数据下载与冻结迁移验证

已完成两个精简测试包及官方代码下载，并从完整归档提取 dataset_8/9 的全部10个实测左关键帧、XYZ标签和相机标定。未下载完整训练集或全部插值视频；未训练模型、未拟合新校准头。

## 主要结果

单位：mm。每帧按全部有限XYZ且Z>0的原生有效像素计算MAE，先对每对象5帧取平均，再对2对象等权平均。

|方法|dataset_8|dataset_9|平均MAE|解释|
|---|---:|---:|---:|---|
|small_affine_ls|48.849|31.381|40.115|冻结C3VD校准迁移|
|small_affine_ridge|48.849|31.381|40.115|冻结C3VD校准迁移|
|small_radial_ls|44.644|28.579|36.611|冻结C3VD校准迁移|
|small_radial_ridge|45.356|28.873|37.115|冻结C3VD校准迁移|
|large_affine_ls|45.295|29.204|37.250|冻结C3VD校准迁移|
|large_radial_ls|42.244|28.081|35.163|冻结C3VD校准迁移|
|endodac_affine_ls|43.740|23.146|33.443|训练清单存在对象重叠，仅供参考|
|endodac_radial_ls|42.374|23.504|32.939|训练清单存在对象重叠，仅供参考|

预设主要对比：Small radial ridge 相对 affine LS 的MAE差为 -3.000 mm（下降 7.48%），两个对象均改善。但径向正则化结果比未正则化 radial LS 高 0.503 mm，不能将优势归因于正则化。Small radial ridge 平均有符号误差为 -36.343 mm，明显低估深度；尚不支持高精度毫米测量或病灶尺寸有效性。

## 下载、身份和公平性依据

- 官方仓库 https://huggingface.co/datasets/maxhallan7/scared ，固定 revision c1246e4f9f52e1ffd51fa45f8544288771785189。test_dataset_8.zip 为252229192字节，test_dataset_9.zip 为396748898字节，两者以及code.zip均与官方LFS SHA256一致且ZIP CRC检查通过。
- 完整dataset_8.zip/9.zip约51GB，因此使用HTTP Range只提取30个必要成员，约182MB传输；成员CRC与ZIP目录一致，另存本地SHA256。没有下载整个大归档，因此不声称验证了大归档整体SHA256。
- 原论文 https://arxiv.org/abs/2101.01133 第II节说明每个dataset对应一个猪对象；单位按原论文毫米深度指标与官方XYZ/Z表示读取。旧README的三个dataset描述与现行发布数量不符，采用论文和实际文件目录，保留原文快照。
- EndoDAC本地发布版 train_files.txt 包含dataset8共3134行、dataset9共2737行（总15351行）。这与此前依据常见挑战划分推测只训练1–7不一致，现已更正。实际权重逐样本训练历史无法独立核查，因此保守标为对象重叠风险，不能用来支撑公平的独立泛化排名。
- DA V2基础模型预训练数据不能完全审计；只能确认本项目未用SCARED进行拟合。对象级外部测试不等于基础预训练绝对无重叠。
- 新增的是两个SCARED测试对象。SERV-CT与SCARED跨研究供体关系未获得直接证明，因此分数据集报告，不直接宣称合计四个完全独立供体。

## 数据质量与评估边界

- 保留原生1024×1280图像，读取float32 XYZ的第三通道作为轴向深度，不使用点到相机的欧氏距离；预测插值回原生RGB尺寸，未重采样真值。
- 所有头保持原始1–100 mm输出范围；主结果不按真值上限删像素，另报告预先约定的1<GT<100支持域。两种支持域不是两个模型选优机会。
- 参照存在超出100mm的有效值以及少量极端值，最大约6567mm。reference_tail_audit.json记录每帧分位数和超界比例；没有根据观察到的误差事后删除这些点。
- XYZ经提供的K和畸变投影后的残差p95约1.14–1.79像素。这是内部投影一致性检查，不代表相机真实标定误差或物理深度精度。
- 只有两个对象，不进行像素级显著性检验，不把帧或随机种子充当独立样本。
- 仅实测关键帧、单目冻结校准迁移；官方挑战还包括插值视频和双目方法。因此这些数字不能直接与挑战排行榜比较，不声称复现官方排行榜或达到SOTA。

## 复现命令

```bash
/ssd-1/temp/kimicode/multmodel/venv-gpu/bin/python scripts/acquire_scared_test.py
/ssd-1/temp/kimicode/multmodel/venv-gpu/bin/python scripts/acquire_scared_keyframes.py
/ssd-1/temp/kimicode/multmodel/venv-gpu/bin/python scripts/run_scared_validation.py inference
/ssd-1/temp/kimicode/multmodel/venv-gpu/bin/python scripts/run_scared_validation.py evaluate
/ssd-1/temp/kimicode/multmodel/venv-gpu/bin/python scripts/report_scared_validation.py
```

在项目根目录运行。evaluate为保护已评分结果拒绝覆盖现有summary.json；现有结果可直接运行report脚本独立重算核对。下载可复用已校验文件；推理复用缓存。若开展新协议，应使用独立输出目录并先锁定方案。

关键文件：scripts/acquire_scared_test.py（下载与协议）、scripts/acquire_scared_keyframes.py（按范围提取）、scripts/run_scared_validation.py（推理与评分）、src/radial_affine_calibration.py（原冻结算法）、scripts/report_scared_validation.py（独立复算）。

新数据报告存于outputs/scared_validation。所有160行逐帧MAE已独立复算，汇总和原冻结头哈希通过检查。未更改论文正文，也未把深度验证当成真实病灶尺寸验证。
