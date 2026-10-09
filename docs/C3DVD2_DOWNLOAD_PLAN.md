# C3VDv2 下载计划（MetricEndo 外部几何验证）

## 结论先行

你发的 `SmartWhatt/c3vdv2-SfM` 可以作为外部 3D/深度参考候选，但下载前要看它的数据卡/索引是否保留原始的 RGB、metric depth、camera pose、intrinsics 和 3D mesh。仅有 SfM 稀疏点云/相机轨迹不够验证逐像素毫米深度；若只有 RGB 或 pose，也不能验证 lesion-size MAE。

当前 HF 页面/API 在本执行环境网络不可达，本地也没有该仓库缓存，因此我暂时无法核对它逐文件的具体结构。以下下载建议基于官方 C3VDv2 数据说明；若 HF 派生版删减了文件，应按“必须项”核对，不满足则直接从官方 JHU Dataverse 获取注册视频包。

## 首轮最小下载清单

### 1. Registered Videos：定量几何验证主集

优先从 `registered_videos` 下载带逐帧像素真值的视频。建议先用 Data Summary Sheet 筛选：

- 至少 3 个不同 phantom segment/几何区域，按 segment 分组留出测试，避免相邻帧泄漏；
- 先选 `v1` 或 `v2` 的 clean、registration qualitative score 1/2 视频；
- 选择明确标记有 polyp 的 phantom sequence；
- 另外下载与部分 `v2` 同轨迹配对的 `v3` debris 视频，用作遮挡/反光/污物鲁棒性测试；v3 的 GT 是从 clean 配对视频复用的，应限定为几何静态场景，并按官方说明处理 debris 带来的 GT 不确定区域。

每个选中的 registered video 至少下载完整配套目录/文件：

| 文件 | 用途 | 优先级 |
|---|---|---|
| `rgb/NNNN.png` | 输入影像 | 必须 |
| `depth/NNNN_depth.tiff` | 独立 Z-depth 真值，16-bit，0--100 mm 线性编码 | 必须 |
| `pose.txt` | camera-to-world 位姿/坐标变换 | 必须 |
| `camera_intrinsics.txt` | 鱼眼相机内参及投影模型 | 必须 |
| `coverage_mesh.obj` | GT 表面几何/可见性参考 | 必须 |
| `normals/NNNN_normals.tiff` | 法向与曲率/表面朝向验证 | 推荐 |
| `occlusions/NNNN_occlusion.tiff` | 遮挡区域屏蔽及分层误差 | 推荐 |
| `C3VDv2_Data_Summary_Sheet_v1.xlsx` | 视频标签、polyps/debris、参数与配对关系 | 必须 |
| `optical_flow/NNNN_flow.tiff` | 时序/运动分析 | 可后补 |
| `diffuse/NNNN_diffuse.png` | 光照/反射分析 | 可后补 |

**首轮规模建议：** 先下载 8--12 个完整 registered clips，覆盖至少 3 个 phantom segments、clean/debris 配对和有/无 polyp 场景。不要随机拆帧；按 phantom segment 或整段视频做 train/validation/test 外部留出。

### 2. Polyp 尺寸验证还要额外确认病灶级真值

C3VDv2 的官方描述包含带 Paris 类别的 phantom lesions，并在数据表中列出 lesion major-axis diameters；但逐帧 depth GT 本身不是病灶卡尺标签。下载前须确认 Data Summary Sheet/3D model 是否能将**具体 lesion ID、尺寸、所在 mesh 区域**对应起来。

若能获得病灶级 mesh 标注/直径：

1. 从该病灶对应 mesh 生成与相机位姿一致的 lesion mask；
2. 用对应 RGB/depth/intrinsics 跑 MetricEndo；
3. 与 mesh 上独立测得的 3D 主轴、局部表面尺寸比较；
4. 按视频/phantom segment 分组，而不是按帧随机分割。

若无法得到 lesion-specific mesh/ROI/diameter：C3VDv2 仍可验证**深度图、反投影、相机几何、OBB 轴长的一般几何一致性**，但不能报告真实 polyp diameter MAE，也不能验证 Paris 分类。

## 现阶段先不要下载的类别

| 数据类别 | 原因 |
|---|---|
| Simulated Screening Videos | 官方说明主要提供 RGB 与 pose，没有逐像素 depth GT；适合轨迹/流程分析，不是像素级尺度精度主测试集 |
| Deformation v4 | 没有 pixel-wise GT，且变形时 mesh GT 为 undeformed model；只适合定性或专门的形变鲁棒性研究 |
| 全部 169 个 registered clips 一次性下载 | 首轮不需要；先用 Data Summary Sheet 选代表性/成对场景，确认解析器和单位无误后再扩展 |
| 仅下载 RGB 或仅下载 SfM sparse cloud | 缺少足够独立 metric reference，无法验证绝对尺度 |

## C3VDv2 depth 解码与相机模型

官方页面说明 depth TIFF 为相机 Z 轴深度，范围 0--100 mm，16-bit 线性编码，示例 16384 对应 25 mm。实际解码时要根据官方文件说明/元数据确认精确映射、无效值编码和 endian；不可默认把 16-bit 原始值直接当毫米。

C3VDv2 使用约 170° 鱼眼/球面全向相机模型，并提供 intrinsics。MetricEndo 当前 `CameraIntrinsics.from_fov(..., 140°)` 的 pinhole 模型不能直接用于 C3VDv2 原始图像。外部测试前必须实现官方投影/反投影模型，或者只在明确裁剪且经过重标定的区域评测，并在方法中限定适用范围。

## 许可与引用

官方 C3VDv2 页面标注 CC BY 4.0；原始 C3VD v1 页面标注 CC BY-NC-SA 4.0。HF 派生仓库可能有不同条款，必须以数据卡和其上游来源为准。论文应引用 C3VDv2 数据描述论文及数据 DOI，并注明使用的数据版本、视频 ID、下载日期和筛选规则。

## 下载后先运行的审计

本项目适配脚本：`tests/c3vd_reference_audit.py`。完整 reference 数据到位后按 README/manifest 路径运行：

```bash
PYTHONPATH=/ssd-1/temp/kimicode/multmodel \
/ssd-1/temp/kimicode/multmodel/venv-gpu/bin/python \
tests/c3vd_reference_audit.py \
  --dataset-root /path/to/C3VDv2 \
  --manifest /path/to/selected_video_manifest.jsonl \
  --split test \
  --output outputs/c3vdv2_external_audit.json
```

审计通过后再做几何评测，按功能报告：

- 深度：valid-pixel AbsRel/RMSE(mm)、bias、分位数误差；遮挡/无效深度单独计数；
- back-projection：与 reference mesh 的 Chamfer distance、point-to-surface distance；
- OBB：在对应 lesion ROI 或 reference surface patch 上比较 axis lengths/方向；
- scale：按 working distance、工具姿态、视频质量标签分层；
- 统计单位：视频/phantom segment，不能把同一视频的帧当独立生物样本。

C3VDv2 不应替代患者级 lesion caliper/post-resection truth。它能为内镜几何和 depth pipeline 提供比 Kvasir-SEG 更独立的物理参考，但临床病灶尺寸结论仍需带病灶级物理真值的数据集或体模实验。
