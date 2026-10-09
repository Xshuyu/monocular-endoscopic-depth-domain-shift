# 强基线、新外部测试与临床数据获取

日期：2026-09-27。已完成DA V2 Large同协议推理，以及官方C3VD新序列上的冻结参数迁移测试。现有候选未达到更低主指标MAE，不能宣称超过SOTA。

## 同条件深度比较

| 方法 | 原C3VDv2测试MAE mm↓ | 新C3VD测试MAE mm↓ |
|---|---:|---:|
| 原光度方法 | 11.735 | 11.126 |
| DA V2 Small + 校准集逆深度仿射 | 11.084 | 11.521 |
| DA V2 Large + 校准集逆深度仿射 | 13.454 | 10.256 |
| 冻结的单调校准候选 | 11.136 | 11.723 |

Small为24,785,089参数，Large为335,315,649参数。这是不同容量公开预训练基线比较，不是参数预算匹配消融。Large是更大规模通用基线，不是已经认证的内镜任务SOTA。

两种模型在相同96帧、相同原分辨率GT、相同uint16×100/65535毫米Z解码下评估。每个模型仅在原c1校准集12帧、每帧8192像素上拟合逆深度仿射；测试不使用GT中位数缩放或尺度对齐。预测统一限制在[1,100]mm。直接使用原始预测张量，不使用可视化深度。原测试含72帧、6视频，但只有1个解剖模型，且此前已观察。

Large的官方固定版本为7581137eff8d4e94f6e796d3baea0e9fa79b22d2，safetensors已与官方LFS SHA256匹配；许可CC BY-NC 4.0。推理及数据检查均使用指定venv-gpu，未改变依赖或训练网络。现有脚本中的方法字段是通用depth_anything_v2名称，模型身份由protocol.json中的模型路径与清单哈希区分。

Large在旧测试集相比Small的MAE差为+2.370mm，视频bootstrap描述性95%区间[0.368,4.417]mm；该区间不能作为患者总体推断。

## 真正新增的数据测试

本次从官方链接成功获取原C3VD trans_t1_a归档，其中61个RGB帧。预先指定按文件顺序均匀抽取12帧，并在读取图像/GT前保存清单。与原C3VDv2全部96帧做字节哈希检查，无重复RGB。

新序列不参与拟合、超参数选择或候选修改。Small、Large和候选均直接沿用C3VDv2冻结标定。其结果说明：Large在这个新视频上更好，但候选依然没有MAE收益。仅一个新增物理模体视频，不能外推临床优势，也不能排除预训练语料重叠。

新序列的RMSE：光度15.181、Small15.136、Large14.048、候选12.816mm。候选RMSE更小而MAE/AbsRel更差，不能事后换主指标宣布胜出。RMSE口径为各帧RMSE的平均。

## 专用公开方法的复现核对

1. DAE作者仓库当前固定树快照e5ba8c5fb11c0ae7f5832436d197ab2d46353bff仍仅有“source code is coming soon”说明。此前论文页面声称提供代码入口，并不代表代码已经发布。本轮未运行DAE。
2. PPSNet公开学生权重下载入口可访问，但推理需要与去畸变图像一致的相机内参，当前C3VDv2子集尚未建立该预处理链。本轮未把它作为已复现模型。固定树快照f3839b4f8d53780feabed6392e175bb55f5acc59的test_ppsnet.py第276行调用scale_predictions(depth_gt, depth_est)，使用测试GT对齐尺度；第286行附近a1采用相对误差<0.1定义，也不同于标准δ<1.25。这些是评估任务差异，不能直接与绝对毫米误差横比，也不能仅凭此认定原论文存在不当行为。
3. EndoStreamDepth有权重及推理代码，但涉及序列输入、Mamba/flash-attn等依赖和作者指定环境。本轮未安装或执行它；不得把Large比较写成该方法复现。

来源：[DAE作者仓库](https://github.com/ShuweiShao/DAE)、[PPSNet作者代码](https://github.com/yahskapar/PPSNet/blob/f3839b4f8d53780feabed6392e175bb55f5acc59/test_ppsnet.py)、[EndoStreamDepth作者仓库](https://github.com/MedICL-VU/EndoStreamDepth)、[DA V2 Large官方模型](https://huggingface.co/depth-anything/Depth-Anything-V2-Large-hf)。

## 可复现命令

从项目根目录运行，评估输出目录必须尚不存在：

```bash
/ssd-1/temp/kimicode/multmodel/venv-gpu/bin/python -B scripts/download_large_depth_baseline.py
/ssd-1/temp/kimicode/multmodel/venv-gpu/bin/python -B tests/c3vdv2_external_depth_validation.py --model datasets/external_models/Depth-Anything-V2-Large-hf --output /tmp/large_depth_repeat
/ssd-1/temp/kimicode/multmodel/venv-gpu/bin/python -B scripts/evaluate_frozen_c3vd_transfer.py --output /tmp/frozen_transfer_repeat
```

迁移脚本使用已保存于outputs/c3vdv2_large_validation的Large标定，而不是/tmp下任意新标定；这是为了固定测试条件。所有输入均来自公开获取并校验的数据/权重。

八项已有必要测试通过，Small/Large协议关键字段一致，新序列12帧×4方法共48条结果无重复遗漏。结果、代码哈希和比较图位于outputs/next_validation、outputs/c3vdv2_large_validation和outputs/c3vd_frozen_transfer。未修改原论文数字以掩盖负结果。

## 已取得真实病灶尺寸标签

从官方Figshare取得Polyp-Size完整数据包（362,536,394字节），官方MD5 c4f2fb4b2bbc207b618a155f2b2eeac1验证通过，许可CC BY 4.0。42个视频全部可读取视频头；42行标签按Video_ID NN与VideoNN.mp4逐一对应，无缺失或重复ID。尺寸字符串带明确mm后缀，已解析成正数毫米值。

数据的Polyp_Size_Lables.csv文件实际为XLSX容器，已在保留原件的前提下按Excel结构解析，生成outputs/next_validation/polyp_size_labels_normalized.csv。新数据的42个视频全部登记为外部临床留出集，不用于训练或超参数选择；未在这些视频上运行模型预测。clinical_measurement_inputs.csv记录视频、尺寸GT及待补的mask/标定输入，不会用缺失值伪造测量。

[官方数据与论文](https://doi.org/10.1038/s41597-025-05251-x)说明标签取自切除后标本：三位观察者独立使用卡尺测量最大直径，取三次中的最大值；作者报告ICC=0.987，95%CI 0.979–0.993。它不是在体3D真值，也不能直接等同于当前算法的1–99%几何范围。形变和最大值汇总都可能影响与在体预测的比较。

实际归档只包含42个视频和标签工作簿，**没有病灶分割mask、相机内参/畸变标定、原始三观察者测量记录或患者ID字段**。因此本次补上了真实毫米尺寸标签与明确视频对应关系，但仍不能完整运行可辩护的3D毫米测量评估，也不能独立重算ICC或确认42个视频代表42位独立患者。设备型号字段不能替代几何标定。

临床数据下一步所需的具体输入已明确：可靠病灶mask或经核验的分割结果、与裁剪后视频像素坐标匹配的标定射线/尺度参照，以及与切除后最大径相匹配的测量终点。作者未公开的标定和观察者原始记录不能由现有标签反推。本轮未替换默认视场角来冒充标定，也未向数据作者发送消息。

数据审计脚本：scripts/audit_polyp_size_archive.py；官方数据来源与文件哈希：outputs/next_validation/*download.json；完整临床留出清单：outputs/next_validation/polyp_size_holdout_manifest.json。
