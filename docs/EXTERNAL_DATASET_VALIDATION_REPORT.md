# 外部数据检索与实测验证报告

## 实际完成及边界

已完成公开数据检索、许可和格式审计、原格式C3VDv2深度验证、真实预训练基线推理，以及180个几何一致的解析表面案例。**尚未完成真实病灶毫米尺寸验证或标注者ICC**：没有取得可核实的物理/临床息肉标签包。

## 数据集选择与可用性

| 数据源 | 官方依据 | 实际取得/审计 | 决定 |
|---|---|---|---|
| C3VDv2 | https://durrlab.github.io/C3VDv2/ ; DOI 10.7281/T1/JC64MK；CC BY 4.0 | BeileiCui/C3VDv2镜像的96帧，1350×1080、uint16参考深度 | 接受为有来源记录的物理模体深度试验子集；未与不可访问的官方文件逐字节核对 |
| C3VDv2另一镜像 | kangqi-ni/c3vdv2 | 96帧的深度为8位、540×675，未提供转换说明 | 排除毫米评估，不猜测其深度单位 |
| 原C3VD | https://durrlab.github.io/C3VD/ ; CC BY-NC-SA 4.0 | 官方Drive返回Quota exceeded HTML，而非ZIP | 未计入实验 |
| PolypSense3D | https://github.com/HNUicda/PolypSense3D ; DOI 10.7910/DVN/LKDIEK；DOI许可CC0 | 镜像7z约1.1GB，SHA256已校验，但仅含虚拟部分；30个P1–P30尺寸行 | 缺物理/临床包；虚拟帧和Polyp_ID对应关系及深度整数到米的完整映射未核实，不计算伪毫米误差 |
| Polyp-Size-Dataset | https://github.com/Scarlett213/Polyp-Size-Dataset | 仓库主要提供预处理和交叉验证划分；本次未取得原始测量数据 | 未计入实验 |
| C3VDReg | https://github.com/linzhe001/C3VDReg | 登记/定位任务，原有接口的tiny样本不具备完整参考条件 | 不替代病灶直径验证 |

PolypSense3D的公开issue提出标签与论文对应关系疑问，这是待核实的第三方报告，不直接认定所有标签错误。我们独立确认的是当前取得的包只有虚拟数据，且说明未建立可审计的帧→对象→测量终点链。相关记录： https://github.com/HNUicda/PolypSense3D/issues/2 。

官方Dataverse在本环境返回403；普通网页检索服务也失败。已改用作者GitHub、DOI注册元数据和公开镜像，访问记录保存在 `datasets/external_sources/`。未绕过登录、访问受限数据或发送邮件。

## 外部试验协议

- 校准：c1_ascending_t1_v1的12帧。验证：c1_ascending_t1_v2的12帧。
- 测试：c2的ascending、cecum、descending、rectum、transverse1、transverse2，共6视频×12帧=72帧。c2不参与任何尺度拟合。
- 每段按文件顺序均匀抽12帧，选择先于预测。原始深度按uint16×100/65535解码为相机Z轴毫米，0和65535排除。
- 保留原分辨率GT；只把预测插值到GT尺寸。不假设140°针孔内参，不把轴向深度当作射线距离。
- 光度原方法、30mm常数、校准集深度中位数、校准集拟合的光度比例、真实预训练Depth Anything V2 Small共同评估。最后者直接使用模型原始张量，不使用展示用8位深度图。
- 神经网络含 24,785,089 个参数，无训练、无微调；不是参数量/训练预算匹配的消融，只是外部方法比较。
- 光度比例和网络逆深度仿射系数只在校准集拟合；测试帧不做真值对齐或median scaling。所有预测使用预先固定的[1,100]mm范围，保留截断比例。
- 汇总以每帧指标平均到视频，再对视频等权平均。RMSE列在JSON中也是帧均值，而非所有像素合并后的RMSE。置信区间仅对6个视频重采样，不能视为患者总体区间。
- **独立测试解剖模型只有1个**；测试均为v3含污物场景，解剖与拍摄条件变化没有单独隔离。结论仅为外部物理模体pilot。

## 外部深度结果

| 方法 | 深度MAE / mm | 序列bootstrap 95%区间 | AbsRel | δ<1.25 |
|---|---:|---|---:|---:|
| Constant 30 mm | 14.575 | [13.724, 15.429] | 0.877 | 21.7% |
| Calibration median | 13.478 | [12.541, 14.506] | 0.745 | 25.0% |
| Photometric (original) | 11.735 | [10.856, 12.527] | 0.593 | 28.8% |
| Photometric + calibration scale | 15.774 | [14.323, 16.979] | 1.058 | 22.0% |
| Depth Anything V2 + calibration | 11.084 | [9.018, 13.177] | 0.572 | 30.9% |

学习型基线与原光度方法的逐视频MAE差平均为 -0.651 mm，描述性区间为 [-3.141, 1.762] mm。不据此宣称临床优势或普遍优胜。

原光度方法的校准比例为 1.4971，在新模型/污物条件下误差增加。它是需要保留的泛化失败结果，不在看到测试结果后重新选尺度。

**这里的11–16mm是深度误差，不是病灶主径误差，不能与此前3.006mm主径MAE直接比较。**

## 几何一致与无混杂的补充基准

新增圆盘、穹顶和鞍面，完全交叉4种尺寸、3种离群点水平和5个种子，共180案例、720条方法结果。先在毫米坐标生成表面，再施加已知旋转和平移。完整几何主径由模型支持点保证；1–99%参考范围在污染前的已知本征轴上计算，和完整范围分开保存。它消除了该补充试验中的形态—尺寸混杂，**没有声称修复或替换旧RGB生成器**。

| Surface | Added outlier fraction | PCA q | MAD-PCA q | SO(3) no filter | SO(3) MAD |
|---|---:|---:|---:|---:|---:|
| disc | 0% | 0.048 | 0.048 | 0.087 | 0.087 |
| disc | 5% | 26.872 | 0.043 | 27.032 | 0.300 |
| disc | 10% | 44.481 | 0.058 | 44.314 | 0.372 |
| dome | 0% | 0.027 | 0.027 | 0.064 | 0.064 |
| dome | 5% | 27.581 | 0.027 | 26.697 | 0.061 |
| dome | 10% | 43.648 | 0.029 | 42.846 | 0.067 |
| saddle | 0% | 0.030 | 0.030 | 0.060 | 0.060 |
| saddle | 5% | 27.951 | 0.030 | 25.886 | 0.062 |
| saddle | 10% | 44.338 | 0.030 | 44.678 | 0.059 |

表中是针对干净表面1–99%参考范围的MAE（mm）。不能将该分位数参考量称为临床卡尺主径。没有进行超参数选择。在本次9个形态×噪声汇总条件中，MAD-PCA的平均误差均低于采用相同MAD过滤的SO(3)方法，未观察到旋转优化的额外优势；这只是当前解析表面家族的结果。

## 仍需要用户/数据提供方资源才能完成

1. PolypSense3D物理与临床完整包、可核实的帧—对象映射、尺寸终点定义和相机标定。公开镜像只有虚拟数据，官方链接在本环境无法获取相应包。
2. 多观察者重复测量记录；这些不能从单个尺寸标签或模型预测推导出来。
3. 原始C3VDv2文件与镜像的官方校验，以提升来源证据；当前已验证的是镜像提交、LFS哈希和原格式/单位一致性。
4. 扩大外部物理模型和患者范围；当前六视频仍来自一个测试解剖模型。

## 复现命令

在项目目录中，用新的、尚不存在的输出目录执行：

```bash
/ssd-1/temp/kimicode/multmodel/venv-gpu/bin/python -B tests/c3vdv2_external_depth_validation.py --output /tmp/c3vdv2_depth_repeat
/ssd-1/temp/kimicode/multmodel/venv-gpu/bin/python -B tests/consistent_surface_reference_benchmark.py --output /tmp/consistent_surface_repeat
/ssd-1/temp/kimicode/multmodel/venv-gpu/bin/python -B tests/test_external_depth_validation.py
```

下载脚本：`scripts/download_external_validation_data.py c3vdv2_native`。模型来自 `depth-anything/Depth-Anything-V2-Small-hf` 的固定提交，保存在 `datasets/external_models/`。输入文件、模型和代码均有哈希记录。数据用途遵循原许可并引用原作者。
