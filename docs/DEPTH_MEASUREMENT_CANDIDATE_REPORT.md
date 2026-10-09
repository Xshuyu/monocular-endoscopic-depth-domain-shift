# 深度估计与毫米尺寸测量：候选方法第一轮验证

日期：2026-09-27。结论：候选方法已实现并运行，但没有超过当前项目深度基线，也没有证据支持 SOTA 或真实病灶尺寸精度提升。旧算法和论文稿未被替换。

## 算法与公平比较

代码 `src/calibrated_depth_measurement.py`：将校准帧中的预测按数值排序，分为64个等样本量分组；各组取预测中位数和参考深度中位数，再用加权单调回归约束毫米映射。神经网络相对逆深度采用递减映射，光度深度采用递增映射；超出标定范围时使用端点值。两者在毫米深度空间进行凸组合。这是标准方法组成的校准候选，不宣称新颖性。

只使用原校准集12帧，每帧8192个有效像素拟合；验证集12帧在神经网络权重0、0.25、0.5、0.75、1中选取帧平均MAE最小者。验证集最终选择1，即完全弃用光度分支。拟合和选择完成后保存 frozen_calibration.json，再读取本轮测试真值。未微调或训练神经网络；复用已校验SHA256的原始网络预测缓存。

此处测试集为此前已分析过的72帧、6视频、1个c2解剖模型。参数没有在测试帧上拟合，但研发已知旧测试表现，因而本次只能标记为探索性复用测试。不能称为新的盲测或独立外部验证。未来方法选择必须有新的最终留出数据。

## 实测结果

以视频等权汇总；RMSE为逐帧RMSE平均到视频后再平均，不是像素合并RMSE。使用原始uint16轴向Z参考，统一[1,100]mm预测范围，无逐测试帧真值对齐。

| 方法 | 测试MAE mm↓ | 测试RMSE mm↓ | AbsRel↓ | δ<1.25↑ |
|---|---:|---:|---:|---:|
| 原光度方法 | 11.735 | 15.202 | 0.593 | 28.82% |
| DA V2 Small + 原逆深度仿射标定 | 11.084 | 14.073 | 0.572 | 30.91% |
| 单调神经深度/验证集选出的候选 | 11.136 | 13.629 | 0.633 | 31.98% |
| 单调光度分支 | 18.443 | 23.010 | 1.151 | 25.85% |

候选减原神经基线的MAE差为+0.052 mm，序列bootstrap描述性95%区间[-0.534,0.637]mm。只有一个测试解剖模型，不能解释为患者总体置信区间。主指标MAE没有改善，AbsRel变差；RMSE虽有改善，不能事后更换主指标来宣布胜出。

验证集MAE从6.240下降到4.307mm，但该改善未迁移到旧测试集。校准/验证都来自c1 ascending，测试来自c2且拍摄条件不同；这与标定泛化不足相符，但当前设计无法单独归因于解剖、污物或其他成像变化。单调映射的端点饱和和不同数据中的尺度关系也应在开发集上分析，不能依据测试逐帧调参。

## 尺寸模块已实现的部分

`measure_axial_depth` 接收毫米轴向深度、病灶mask以及真实标定的每像素相机射线；通过 ray/ray_z * Z 反投影，使用径向MAD过滤和PCA，输出1–99%分位范围及有效/保留点比例。不会默认使用140度视场角。

`scripts/predict_calibrated_measurement.py` 将保存的深度校准与该模块连接，保存深度NPY及测量JSON。两分支与融合结果的尺寸范围只表示组件敏感性，不能称为统计置信区间或误差界。

该终点是稳健分位几何范围，不是临床卡尺完整直径，也不是测地线长度。四项单元测试通过，另有一次命令行端到端解析几何检查通过；这些证明接口和单位处理正确，不证明真实息肉测量准确。当前子集没有可用的病灶mask、尺寸GT和已核实的对应相机射线，故未计算真实病灶MAE、ICC或伪造尺寸提升。

## 公开强基线核对

- [EndoStreamDepth作者仓库](https://github.com/MedICL-VU/EndoStreamDepth)：提供C3VD不同划分的模型与序列推理方案。其论文协议不是当前C3VDv2子集协议，数值不能直接横比。官方setup脚本固定torch2.4及xformers，另需flash-attn和本地Mamba构建；当前指定环境torch2.5.1，尚未验证兼容性。本轮未安装或执行此模型。
- [PPSNet作者仓库](https://github.com/yahskapar/PPSNet)：提供预训练模型和评估代码，原评估涉及相机校准与去畸变。不能直接把原生畸变图像送入该协议后称为公平复现。本轮未运行。
- [Depth Any Endoscopy论文官方页面](https://openaccess.thecvf.com/content/CVPR2026/html/Shao_Depth_Any_Endoscopy_Towards_Self-Supervised_Generalizable_Depth_Estimation_in_Monocular_CVPR_2026_paper.html)：列出自监督泛化方法及作者代码入口。已确认相关工作存在，尚未完成其权重和本项目协议的复现。

以上是需要纳入的强基线，不是对唯一“当前SOTA”的认证。新候选要先完成同数据、同GT对齐规则、同有效区域和尺度条件的比较，再谈超越。

## 运行命令

在项目根目录执行；输出目录必须尚不存在。

```bash
/ssd-1/temp/kimicode/multmodel/venv-gpu/bin/python -B tests/test_calibrated_depth_measurement.py
/ssd-1/temp/kimicode/multmodel/venv-gpu/bin/python -B tests/run_calibrated_depth_candidate.py --output /tmp/calibrated_depth_repeat
```

有真实标定和mask时的应用命令（这里的路径是调用模板，不是已取得的真实病灶数据）：

```bash
/ssd-1/temp/kimicode/multmodel/venv-gpu/bin/python -B scripts/predict_calibrated_measurement.py \
  --relative relative_depth.npy --photometric photometric_depth_mm.npy \
  --calibration outputs/calibrated_depth_candidate/frozen_calibration.json \
  --mask lesion_mask.npy --rays calibrated_rays.npy --output /tmp/lesion_measurement
```

后续重点：建立包含强基线的统一推理协议；增加不同解剖/采集条件的开发数据；在真正未观察过的数据上锁定最终比较；获得实际相机标定、病灶尺寸标签与对象对应关系。不能靠反复试验已观察的c2测试集补足这些证据。
