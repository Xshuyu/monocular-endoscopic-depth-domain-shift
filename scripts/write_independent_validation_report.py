"""Render independent-object experiment with explicit clinical evidence limits."""
from pathlib import Path
import json,csv
R=Path(__file__).resolve().parents[1];O=R/'outputs/independent_validation';M=R/'manuscript_cmig'
a=json.loads((O/'summary.json').read_text());a={r['method']:r for r in a if r['support']=='all_reference'}
with (O/'per_object.csv').open() as f:v=[r for r in csv.DictReader(f) if r['support']=='all_reference']
order=['small_affine_ls','small_affine_ridge','small_radial_ls','small_radial_ridge','large_affine_ls','large_radial_ls','endodac_affine_ls','endodac_radial_ls']
labels=['DA V2 Small + affine LS','DA V2 Small + affine ridge','DA V2 Small + radial LS','DA V2 Small + radial ridge','DA V2 Large + affine LS','DA V2 Large + radial LS','EndoDAC + affine LS','EndoDAC + radial LS']
rows=[];md=[]
for name,label in zip(order,labels):
 x=a[name];e=[float(next(r['mae_mm'] for r in v if r['method']==name and r['object']==obj)) for obj in ['Experiment_1','Experiment_2']]
 rows.append(label+' & '+' & '.join(f'{z:.3f}' for z in [*e,x['mae_mm'],x['rmse_mm']])+r' \\')
 md.append('| '+label+' | '+' | '.join(f'{z:.3f}' for z in [*e,x['mae_mm'],x['rmse_mm']])+' |')
section=r'''\section{Independent-object and device transfer}
\subsection{Dataset, freezing and matched baselines}
We extended evaluation to SERV-CT, which provides CT-derived reference depth, rectified camera matrices and images of two ex vivo porcine specimens acquired with classic da Vinci endoscopes \citep{Edwards2022}. All eight left-camera views of each specimen were included. The independent experimental units are two specimens, not 16 images; hardware, anatomy, tissue properties and imaging geometry change together. This is a cross-device/domain stress test rather than an isolated estimate of a camera effect or a clinical lesion-size study.

Before downloading the new image archive, we froze eight heads and the evaluation protocol. The four original Small heads were unchanged. We added Depth Anything V2 Large and the released EndoDAC model \citep{Cui2024}, each with affine and radial least-squares calibration. All heads used the same 16,384 development calibration positions, six training scenes, scene weights and inverse-depth objective; feature transforms used training data only. No SERV-CT references were used for fitting, model selection or scale alignment. Official archive MD5 and extracted-file hashes were verified. A documented directory alias resolves the archive's \texttt{Reference\_CT} name versus the toolkit documentation's \texttt{Ground\_truth\_CT}; data values were not changed.

EndoDAC uses its released architecture and complete checkpoint, with strict parameter loading and author-compatible RGB preprocessing. Its inverse-depth feature is calibrated using the same downstream protocol as the generic backbones. This is a comparison of frozen features under our metric-calibration protocol, not reproduction of the original paper's test-time scale-aligned scores. Small, Large and EndoDAC have approximately 24.8, 335.3 and 99.1 million parameters, respectively. Their pretraining/adaptation resources are unequal; only the downstream calibration budget is matched. No neural weights are trained in this study, and foundation-model pretraining overlap is not independently excluded.

\subsection{Units, support and camera consistency}
SERV-CT reference PNG values are divided by 256 to recover axial millimetres. Primary evaluation excludes the official blue outside-reference region and nonpositive reference depth, retaining monocular-visible occluded/non-overlap regions. Every method is scored on identical native-grid pixels. Secondary analyses use the official nonoccluded overlap mask and, separately, reference depths strictly between 1 and 100 mm. These masks and the unchanged prediction range of $[1,100]$ mm were specified before acquisition. We report frame means within each specimen followed by equal-specimen means, without significance claims at $N=2$.

The 16 supplied calibration records provide rectified $P_1$, $P_2$ and $Q$ matrices. Rays are derived from the inverse left-camera intrinsic matrix; applying distortion correction again to already rectified images would be inappropriate. Across deterministic sampled pixels, the largest frame-wise 95th percentile difference between reference axial depth and disparity reprojected with $Q$ is 0.015 mm. This is internal encoding/calibration consistency, not a newly measured physical calibration accuracy or an estimate of the CT registration uncertainty.

\subsection{Results and transfer limits}
\begin{table}[htbp]\centering\small
\caption{Independent SERV-CT transfer, all-reference support. MAE and mean per-frame RMSE in mm. Each specimen contributes eight views; only two independent specimens are represented.}\label{tab:servct}
\resizebox{\textwidth}{!}{\begin{tabular}{lrrrr}\toprule
Method & Specimen 1 MAE & Specimen 2 MAE & Mean MAE & RMSE \\\midrule
@@ROWS@@
\bottomrule\end{tabular}}
\end{table}

Table~\ref{tab:servct} shows substantial error for every transferred calibration. EndoDAC with affine LS has the lowest aggregate MAE among the eight specified heads (53.295 mm), compared with 56.245 mm for Small affine LS and 55.537 mm for Small radial ridge. Small radial ridge improves specimen 1 but worsens specimen 2 relative to Small affine LS. Adding radial terms to EndoDAC increases its aggregate MAE from 53.295 to 53.669 mm. The radial effect is therefore not uniformly beneficial across backbones and specimens.

Reference depth spans 38.398--194.039 mm; the equal-frame mean proportion above the 100 mm output bound is 20.34\%. However, on the prespecified 1--100 mm reference subset, MAE remains 52.253 mm for Small affine LS, 50.625 mm for Small radial ridge and 50.329 mm for EndoDAC affine LS. The unchanged range limit alone therefore cannot account for the observed error. Mean signed errors are strongly negative, indicating depth underestimation. These results demonstrate a failure of direct C3VD-calibration transfer to this different device/domain, rather than evidence of deployable cross-device accuracy. Broader independent-object evaluation remains necessary; the original frozen transfer results have not been retroactively corrected.
'''.replace('@@ROWS@@','\n'.join(rows))
(M/'independent_validation.tex').write_text(section)
p=R/'scripts/revise_cmig_radial.py';s=p.read_text()
s=s.replace('Controlled Ablation and Frozen Phantom Transfer Evaluation','Controlled Ablation and Cross-Device Transfer Evaluation')
if 'independent_validation.tex' not in s:s=s.replace(r'\section{Implications for lesion measurement}',r'\input{independent_validation.tex}'+'\n\n'+r'\section{Implications for lesion measurement}')
s=s.replace('Evaluation excludes test-reference scale alignment and reports video-level and joint radius--depth diagnostics.','A further frozen comparison uses 16 views of two SERV-CT specimens from a different endoscope system, adding EndoDAC and a larger generic backbone. Evaluation excludes test-reference scale alignment and reports group-level and radius--depth diagnostics.')
s=s.replace('Previously reported fixed-coefficient gains are reproducible, but their original fitting provenance remains unresolved.','On SERV-CT, all eight prespecified heads have MAE above 53 mm, indicating poor direct transfer to the new device/domain.')
s=s.replace('The results are limited to phantom axial depth.','The results concern phantom and ex vivo axial depth; direct cross-device transfer remains inaccurate.')
s=s.replace('with repeated segment geometry and no new-device validation.','with repeated segment geometry. The additional device/domain test contains only two ex vivo specimens and confounds hardware with anatomy and acquisition conditions.')
s=s.replace('A held-out camera and independently sampled physical objects are stronger tests than more frames from the same videos.','The additional held-out device/domain and two independent physical specimens expose severe transfer error, but cannot isolate which acquisition factor causes it.')
s=s.replace('the present comparisons do not establish superiority over contemporary domain-specific depth systems.','the matched EndoDAC comparison does not establish superiority over domain-specific depth systems or comprehensive contemporary SOTA coverage.')
s=s.replace('A clinical sizing claim would require paired physical size references, calibrated rays, lesion masks or endpoints, object-level splits and raw repeated-observer measurements.','A clinical sizing claim would require paired physical size references, calibrated rays, lesion masks or endpoints, object-level splits and raw repeated-observer measurements. An audit of the locally obtained 42-video Polyp-Size archive confirmed millimetre size labels, but found no camera calibration, lesion masks, patient identifiers or raw observer measurements. The post-resection maximum-caliper endpoint also differs from the implemented percentile extent. Consequently no clinical sizing error or recomputed observer ICC is reported. Camera matrices from SERV-CT cannot substitute for calibration of those clinical videos.')
s=s.replace('Public dataset sources are C3VD and C3VDv2 \\citep{Bobrow2023,Golhar2026};','Public depth dataset sources are C3VD, C3VDv2 and SERV-CT \\citep{Bobrow2023,Golhar2026,Edwards2022};')
s=s.replace('Radial features improved development MAE, but the selected ridge regularization did not preserve that advantage on new sequences. Spatial and depth-conditioned failures, and the stronger newly fitted affine comparator, limit a claim of general superiority.','Radial features improved development MAE, but the selected ridge regularization did not preserve that advantage on new sequences. Independent ex vivo device/domain transfer produced large errors for all compared heads. Spatial failures and the matched EndoDAC comparison limit a claim of general superiority.')
refs=r'''\bibitem[Edwards et al.(2022)]{Edwards2022} Edwards, P.J.E., Psychogyios, D., Speidel, S., Maier-Hein, L., Stoyanov, D., 2022. SERV-CT: A disparity dataset from cone-beam CT for validation of endoscopic 3D reconstruction. Medical Image Analysis 76, 102302. \url{https://doi.org/10.1016/j.media.2021.102302}.
\bibitem[Cui et al.(2024)]{Cui2024} Cui, B., Islam, M., Bai, L., Wang, A., Ren, H., 2024. EndoDAC: Efficient Adapting Foundation Model for Self-Supervised Depth Estimation from Any Endoscopic Camera. arXiv:2405.08672. \url{https://arxiv.org/abs/2405.08672}.
'''
if '{Edwards2022} Edwards' not in s:s=s.replace(r'\end{thebibliography}',refs+r'\end{thebibliography}')
p.write_text(s)
report='''# 独立对象、领域基线和尺寸证据补充报告

## 完成状态

1. **已实际完成新的独立对象/设备深度验证**：SERV-CT，两个独立离体猪标本，每个八个左视图。与结肠体模使用不同手术内镜系统；硬件、组织和场景同时变化，不能分离“相机本身”的因果效应。N=2 仅为小样本迁移压力测试，不是16个对象或临床病例。
2. **已实际完成统一协议领域基线**：官方 EndoDAC 冻结模型，以及 DA V2 Large。每个新增骨干均比较 affine/radial LS；加上原 Small 四模型，共八组、48次实际网络推理。没有训练神经网络。
3. **真实病灶毫米尺寸与多标注者一致性尚未完成**：42个 Polyp-Size 视频具备毫米标签，但对应标定、掩膜、病例ID及原始逐观察者记录缺失。已输出缺项表、采集模板和经过解析测试的一致性分析工具，没有生成假 ICC 或尺寸误差。

## 冻结和公平性

- 下载新图像前冻结代码、协议、校准头和权重/输入哈希。
- 新增骨干使用原52帧、6个场景的相同16384像素标定预算、场景均权、训练专属标准化和逆深度最小二乘。原 Small 四头不变。
- 所有方法在相同像素上评估：SERV-CT uint16/256毫米；保留单目可见遮挡区域，排除无参考区域；统一原生网格和[1,100]mm预测范围。
- 按视图→标本→两个标本等权聚合。没有测试参考中位数缩放，没有利用结果更换数据或重新拟合。
- EndoDAC是领域适配的公开基线，不宣称当前SOTA。它的作者预处理和完整权重严格载入；结果衡量“EndoDAC特征+本项目标定”，不是其原论文采用不同尺度处理的数字复现。
- Small/Large/EndoDAC参数量分别为24,785,089 / 335,315,649 / 99,091,748。骨干训练资源不等，公平性仅指下游校准和测试规则。

## 全部主要结果

| 方法 | 标本1 MAE | 标本2 MAE | 等对象 MAE | RMSE |
|---|---:|---:|---:|---:|
'''+ '\n'.join(md)+'''

单位均为mm。结论是明显的直接迁移失败，不能包装成成功的跨设备部署。径向项的收益也没有跨骨干、跨标本一致成立。

参考深度范围38.398–194.039mm；平均20.34%的有效像素超过原100mm上限。但预先指定的1–100mm子集上，Small affine、Small radial ridge、EndoDAC affine的MAE仍为52.253、50.625、50.329mm，说明范围上限不是全部原因。强烈负偏差显示系统性低估；不能仅凭本试验确定是焦距、组织纹理还是标定分布造成。

## 相机与参照检查

官方36.8MB数据包MD5核验通过，170个文件保留SHA256。16组P1/P2/Q矩阵均可读取；校正焦距约934.69–996.40像素，双目基线约5.45–5.50mm。由视差和Q重建的深度与提供的深度图一致，逐帧95百分位差最大约0.015mm。此数值仅表示同一发布数据内部的一致性，不是我们重新标定的精度，也不能替代数据论文报告的CT配准不确定性。有效像素规则与官方工具逐像素一致。

## 第三项为何不能报“完成”

官方 Polyp-Size 的42个毫米标签已验证一一对应视频，但本地发布包不包含所需配套资料。现有 PolypSense3D 镜像仍只有虚拟归档；此次官方Dataverse API访问返回403，不能据论文声称物理/临床文件已取得。没有将文献中的ICC当作本项目重算结果，也没有借用SERV-CT的相机参数处理另一个设备的视频。

要完成真实病灶尺寸验证，需要同一对象对应的相机内参/畸变/裁剪映射、掩膜或已定义端点、病例/对象编号、明确测量时点的毫米参照，以及每位观察者每次测量的原始数值。内镜在体尺寸、切除后最大卡尺尺寸和PCA百分位尺寸不可默认等价。

## 文件和复现

- 核心脚本：`scripts/run_independent_servct.py`，阶段 `freeze` → `acquire` → `inference` → `evaluate`。已冻结输出拒绝覆盖；完整重跑须使用干净的项目副本和输出目录。官方元数据为 `outputs/independent_validation/servct_metadata.json`。
- 官方归档命名 `Reference_CT` 与文档 `Ground_truth_CT` 不同；`layout_adapter.json`记录评估前建立的相对目录别名。复现须保留该别名，未修改图像值。
- 审计：`scripts/audit_servct_and_size_readiness.py`。
- 临床输入模板：`outputs/independent_validation/physical_size_readiness/`。
- 一致性工具：`scripts/analyze_observer_measurements.py`；完整交叉标注矩阵，不插补缺失值；报告绝对一致ICC(A,1)、ICC(A,k)、按患者聚类的描述性bootstrap及逐标注者偏差。ICC(A,k)对应均值，不能解释为“三者最大值”的一致性。

使用统一环境 `/ssd-1/temp/kimicode/multmodel/venv-gpu/bin/python`。9项独立验证审计全部通过；3项一致性解析测试通过，空白真实数据模板被拒绝。

## 原始来源

- SERV-CT官方数据：https://doi.org/10.5522/04/26352199.v1 （CC BY 4.0）
- SERV-CT论文：https://doi.org/10.1016/j.media.2021.102302
- SERV-CT官方评估：https://github.com/surgical-vision/servcttk
- EndoDAC：https://github.com/BeileiCui/EndoDAC ；https://arxiv.org/abs/2405.08672
- Polyp-Size：https://doi.org/10.1038/s41597-025-05251-x
- ICC模型选择参考：https://doi.org/10.1016/j.jcm.2016.02.012
'''
(R/'docs/INDEPENDENT_OBJECT_BASELINE_SIZE_REPORT.md').write_text(report)
print('Wrote independent validation manuscript section and report')
