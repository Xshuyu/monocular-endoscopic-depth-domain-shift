"""Generate evidence-linked SCARED extension and synchronize both working drafts."""
from pathlib import Path
import json,csv,re,shutil,zipfile,datetime
import numpy as np
R=Path(__file__).resolve().parents[1];O=R/'outputs/scared_expansion'
summ=json.loads((O/'summary.json').read_text());paired=json.loads((O/'paired.json').read_text());objs=list(csv.DictReader((O/'per_subject.csv').open()))
def result(m,cohort='cumulative_seven',support='all_reference'):
 return next(x for x in summ if (x['method'],x['cohort'],x['support'])==(m,cohort,support))
def num(m,cohort='cumulative_seven'):return result(m,cohort)['mae_mm']
labels={'small_affine_ls':'Small + affine LS','small_affine_ridge':'Small + affine ridge','small_radial_ls':'Small + radial LS','small_radial_ridge':'Small + radial ridge','large_affine_ls':'Large + affine LS','large_radial_ls':'Large + radial LS','sgbm':'SGBM (stereo)','raft_sceneflow':'RAFT-Stereo (SceneFlow)'}
newdiff=paired[0]['mean_difference_mm'];alldiff=paired[1]['mean_difference_mm'];interval=paired[1]['exploratory_bootstrap_95ci']
section=r'''\section{Expanded SCARED object evaluation and stereo references}
\subsection{Sequential extension and source provenance}
The original SCARED publication describes each dataset as a single porcine subject \citep{Allan2021}. We first evaluated all ten measured left keyframes from datasets 8 and 9 using unchanged C3VD calibration heads. After observing those results, we specified an extension to all measured keyframes of datasets 1, 2, 3, 6 and 7, before downloading and scoring those five additional subjects. Datasets 4 and 5 were excluded on the basis of previously published calibration problems, not model performance. These five subjects belong to the original challenge training partition, but none was used to fit or select the frozen C3VD heads; their role here is external evaluation of this pipeline, not reuse of the official challenge test partition. The extension therefore contains 25 newly evaluated keyframes and 35 cumulative keyframes across seven dataset-defined subjects. The cumulative analysis is sequential and exploratory, not a newly blinded seven-subject confirmatory trial. Cross-study donor independence from SERV-CT has not been established, so the two datasets are not pooled as nine independent donors.

All six Small/Large heads retain their previous C3VD fits, normalization and prediction limits. No SCARED reference is used for fitting or test-time scale alignment. Native floating-point XYZ references provide axial depth via their Z channel. Primary support includes finite XYZ with positive Z; secondary support restricts reference Z to $(1,100)$ mm. All methods use the same reference pixels, and frame means are averaged within subject before equal-subject aggregation. Only measured keyframes are used, excluding kinematically interpolated video references. These results are not directly comparable with challenge scores computed over different image sets and stereo submission protocols.

Official test archives were checked against publisher LFS hashes. Required measured keyframes and calibration files were selectively extracted from pinned full archives using HTTP ranges; member CRC and local hashes were retained. Whole-archive SHA256 verification is not claimed for the partially downloaded archives. Foundation-model pretraining overlap cannot be fully audited. Moreover, the locally released EndoDAC training list includes datasets 8 and 9 (3,134 and 2,737 entries) as well as datasets 1--7. Its SCARED results are therefore excluded from independent-generalization comparisons; a challenge's nominal split cannot substitute for auditing the particular released model's training list.

\subsection{Unmatched-input geometric references}
We add the official SceneFlow checkpoint of RAFT-Stereo \citep{Lipson2021} and a fixed OpenCV SGBM configuration. The former is a learned stereo reference trained on synthetic scenes according to its release; the latter has no learned weights. Neither is adapted to SCARED in this experiment. Both use left and right RGB plus supplied stereo calibration, whereas the main method uses left RGB and its source-fitted mapping. They are therefore geometric references with additional input, not resource- or input-matched monocular competitors, and do not close the gap in nonoverlapping domain-specific monocular baselines.

Stereo pairs are rectified using the supplied intrinsics, distortion and relative pose. RAFT-Stereo uses half-resolution rectified images and 32 recurrent updates; disparities are rescaled to the native grid. SGBM uses 256 disparities, a $5\times5$ block, uniqueness ratio 10 and fixed speckle filtering. Disparity is converted using $Q$, inverse rectification returns axial depth to the original left-camera coordinates, and predictions are sampled on the native reference grid. Invalid correspondences are assigned 100 mm and all predictions clipped to $[1,100]$ mm, matching the inherited output range. Invalid pixels remain in primary scoring. Coverage and an identical intersection-of-stereo-valid-pixels analysis are secondary diagnostics; they do not replace the primary result. No reference-based scale correction is applied. Synthetic geometry checks verify conversion conventions but do not establish physical calibration accuracy.

\subsection{Results and interpretation}
\begin{table}[htbp]\centering\small
\caption{SCARED MAE in mm on all-reference support, with equal-subject weighting. Five new subjects are distinguished from two previously evaluated subjects and the cumulative seven. Stereo references have additional image/calibration input.}\label{tab:scared_expansion}
\begin{tabular}{lrrr}\toprule
Method & New five & Previous two & All seven \\\midrule
'''
for m,label in labels.items():section+=f'{label} & {num(m,"new_five"):.3f} & {num(m,"previous_two"):.3f} & {num(m):.3f} '+r'\\'+'\n'
section+=r'\bottomrule\end{tabular}\end{table}'+'\n\n'
section+=f'The prespecified Small radial-ridge minus affine-LS contrast is {newdiff:+.3f} mm on the five newly evaluated subjects and {alldiff:+.3f} mm cumulatively. Radial ridge improves MAE in {paired[0]["improved_subjects"]}/5 new subjects and {paired[1]["improved_subjects"]}/7 cumulative subjects. The exploratory subject-bootstrap 95\\% interval for the cumulative difference is [{interval[0]:+.3f}, {interval[1]:+.3f}] mm (10,000 draws, fixed seed); it does not account for prior method development or represent a population-level confirmation.\n\n'
section+=f'Across seven subjects, Small radial LS and radial ridge yield {num("small_radial_ls"):.3f} and {num("small_radial_ridge"):.3f} mm MAE. The signed error of radial ridge is {result("small_radial_ridge")["bias_mm"]:+.3f} mm. These data must be read alongside the phantom and SERV-CT results; a reduction relative to one affine calibration does not establish accurate absolute depth or a consistent advantage of regularization. RAFT-Stereo and SGBM have valid-correspondence coverage of {100*result("raft_sceneflow")["coverage"]:.1f}\\% and {100*result("sgbm")["coverage"]:.1f}\\% on primary reference pixels. Their additional geometric information prevents attributing cross-method differences solely to algorithm quality.\n\n'
section+=r'''\begin{figure}[htbp]\centering
\includegraphics[width=0.96\textwidth]{figures/scared_subject_results.pdf}
\caption{SCARED subject-level MAE on identical all-reference support. Left: three frozen Small calibration heads. Right: Small radial ridge and two stereo references with additional input. Subjects 8 and 9 were observed before the five-subject extension; curves connect categorical identifiers only.}\label{fig:scared_subjects}
\end{figure}

The expanded dataset improves object coverage but remains limited to seven dataset-defined subjects in one public acquisition resource. Reference uncertainty, inherited clipping, sequential design, unverified foundation pretraining and the absence of a clean domain-adapted monocular comparator limit inference. None of these experiments provides physical lesion diameter, repeat-observer agreement or clinical measurement validity.
'''
# Save a snapshot once before synchronizing drafts.
backup=O/'manuscripts_before_expansion.zip'
if not backup.exists():
 with zipfile.ZipFile(backup,'w',zipfile.ZIP_DEFLATED) as z:
  for folder in ['manuscript_cmig','manuscript_mbec']:
   for p in (R/folder).glob('*.tex'):z.write(p,str(p.relative_to(R)))
# Publication figure from actual subject results.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':9,'pdf.fonttype':42,'svg.fonttype':'none'})
fig,axes=plt.subplots(1,2,figsize=(10,3.5),sharey=True)
subjects=[1,2,3,6,7,8,9]
for ax,methods in zip(axes,[['small_affine_ls','small_radial_ls','small_radial_ridge'],['small_radial_ridge','sgbm','raft_sceneflow']]):
 for m in methods:
  values=[float(next(x for x in objs if int(x['subject'])==n and x['method']==m and x['support']=='all_reference')['mae_mm']) for n in subjects]
  ax.plot(range(7),values,marker='o',label=labels[m])
 ax.set_xticks(range(7),[str(n)+('*' if n in [8,9] else '') for n in subjects]);ax.set_xlabel('SCARED subject (* previously observed)');ax.grid(alpha=.2);ax.legend(fontsize=7)
axes[0].set_ylabel('Subject mean absolute error (mm)');fig.tight_layout()
figdir=O/'figures';figdir.mkdir(exist_ok=True)
for ext in ['pdf','svg']:fig.savefig(figdir/f'scared_subject_results.{ext}')
plt.close(fig)
abstract=("Metric interpretation of monocular endoscopic depth depends on calibration that may not transfer between acquisition domains. We assess a four-coefficient radial inverse-depth mapping against global affine calibration, separating spatial terms from ridge regularization under matched calibration budgets. Six development scenes support grouped selection. Frozen heads are tested on four colon-phantom sequences, two SERV-CT specimens and a sequentially expanded SCARED cohort comprising seven subjects and 35 measured keyframes. No target-reference scale alignment is used. Small affine and radial ridge yield 8.794 and 8.899 mm MAE on the four phantom sequences; all eight evaluated heads exceed 53 mm on SERV-CT. "
+f"Across SCARED subjects, Small affine, radial least squares and radial ridge yield {num('small_affine_ls'):.3f}, {num('small_radial_ls'):.3f} and {num('small_radial_ridge'):.3f} mm, respectively. "
+"SceneFlow-trained RAFT-Stereo and SGBM provide geometric references with additional stereo/calibration input, not matched-input monocular comparisons. Released EndoDAC training lists overlap SCARED subjects, precluding independent-generalization claims for that model there. A separate post-hoc SERV-CT experiment reduces Small affine MAE to 19.408 mm through calibration on one specimen and evaluation on the other. Spatial benefits vary across datasets, and reference-assisted recalibration remains exploratory. The results characterize calibration-transfer limitations and do not establish clinical lesion-sizing accuracy or uniformly superior regularization.")
conclusion=("We provide controlled calibration ablations and frozen transfer evaluations across phantom and ex vivo datasets. Expanding SCARED evaluation to seven dataset-defined subjects increases object coverage while preserving the distinction between five newly evaluated and two previously observed subjects. Spatial calibration effects vary across datasets; the results do not establish uniformly beneficial regularization or clinically accurate absolute depth. Stereo references show the role of additional geometric input but cannot replace matched-input monocular baselines, and released EndoDAC training overlap limits its use for SCARED generalization claims. Post-hoc SERV-CT recalibration is promising within two observed specimens, not evidence of generalization to further objects. The contribution is an auditable study of calibration transfer and its limits, without claims of clinical lesion measurement or comprehensive state-of-the-art superiority.")
for folder,file in [('manuscript_cmig','main_cmig.tex'),('manuscript_mbec','main_mbec.tex')]:
 d=R/folder;(d/'scared_expansion.tex').write_text(section)
 for ext in ['pdf','svg']:shutil.copy2(figdir/f'scared_subject_results.{ext}',d/'figures'/f'scared_subject_results.{ext}')
 p=d/file;s=p.read_text()
 if r'\input{scared_expansion.tex}' not in s:s=s.replace(r'\section{Discussion and limitations}',r'\input{scared_expansion.tex}'+'\n\n'+r'\section{Discussion and limitations}')
 start=s.index(r'\begin{abstract}')+len(r'\begin{abstract}');end=s.index(r'\end{abstract}',start);s=s[:start]+'\n'+abstract+'\n'+s[end:]
 start=s.index(r'\section{Conclusion}')+len(r'\section{Conclusion}');end=s.index(r'\section*{Data and code availability}',start);s=s[:start]+'\n'+conclusion+'\n\n'+s[end:]
 s=s.replace('Public depth dataset sources are C3VD, C3VDv2 and SERV-CT \\citep{Bobrow2023,Golhar2026,Edwards2022}', 'Public depth dataset sources are C3VD, C3VDv2, SERV-CT and SCARED \\citep{Bobrow2023,Golhar2026,Edwards2022,Allan2021}')
 s=s.replace("The additional device/domain test contains only two ex vivo specimens and confounds hardware with anatomy and acquisition conditions.","SERV-CT contains two ex vivo specimens; the sequential SCARED extension adds seven dataset-defined subjects. Hardware changes remain confounded with anatomy and acquisition conditions, and cross-study donor identities are not verified.")
 if r'\bibitem[Allan' not in s:s=s.replace(r'\end{thebibliography}',r'''\bibitem[Allan et al.(2021)]{Allan2021} Allan, M., McLeod, J., Wang, C.C., et al., 2021. Stereo Correspondence and Reconstruction of Endoscopic Data Challenge. arXiv:2101.01133. \url{https://arxiv.org/abs/2101.01133}.
\bibitem[Lipson et al.(2021)]{Lipson2021} Lipson, L., Teed, Z., Deng, J., 2021. RAFT-Stereo: Multilevel Recurrent Field Transforms for Stereo Matching. International Conference on 3D Vision. \url{https://arxiv.org/abs/2109.07547}.
\end{thebibliography}''')
 p.write_text(s)
report=['# 对象扩展、无 SCARED 训练的双目参考与论文修订','', '已完成新增五个SCARED对象的实测关键帧验证；与此前两个对象合计七对象35帧，按原论文dataset级身份统计。不同研究供体关系未证实，不与SERV-CT简单合并宣称九个独立供体。','', '|方法|新增5对象 MAE|此前2对象 MAE|累计7对象 MAE|','|---|---:|---:|---:|']
for m,label in labels.items():report.append(f'|{label}|{num(m,"new_five"):.3f}|{num(m,"previous_two"):.3f}|{num(m):.3f}|')
report+=['',f'预设主对比（Small径向正则化−仿射LS）新增五对象差值 {newdiff:+.3f} mm；累计七对象 {alldiff:+.3f} mm。负值才表示改善。累计对象bootstrap探索性95%区间 [{interval[0]:+.3f}, {interval[1]:+.3f}] mm。不得用已观察两个对象后的扩展称为全新七对象盲测。','', '已加入官方 SceneFlow RAFT-Stereo 与无学习参数 SGBM。它们使用左右目与相机标定，比单目方法多输入信息，只能单列为几何参考。领域强基线问题部分解决：有无SCARED适配的强双目参考，但仍没有核查合格的、同等输入的领域适配单目基线；未将这项缺口伪装为完成。','', '真实毫米病灶尺寸参照、临床重复标注和ICC仍未完成；本轮未进行神经网络训练。扩展只评实测关键帧，不按测试结果选择lambda或调整裁剪上限。','', 'RA​FT权重来源：https://github.com/princeton-vl/RAFT-Stereo （官方SceneFlow checkpoint）；预训练来源依据发布说明，不能声称审计了原作者全部训练日志。','', '论文更新：两个版本均新增SCARED实验、基线输入差异、对象级图表、训练重叠说明，重写摘要与结论。主要叙事仍是校准迁移的作用及局限，不声称毫米临床精度或SOTA。','', '复现（项目根目录，统一指定Python）：','```bash','/ssd-1/temp/kimicode/multmodel/venv-gpu/bin/python scripts/acquire_scared_expansion.py','/ssd-1/temp/kimicode/multmodel/venv-gpu/bin/python scripts/run_scared_expansion.py mono','/ssd-1/temp/kimicode/multmodel/venv-gpu/bin/python scripts/run_scared_expansion.py stereo','/ssd-1/temp/kimicode/multmodel/venv-gpu/bin/python scripts/run_scared_expansion.py score','/ssd-1/temp/kimicode/multmodel/venv-gpu/bin/python scripts/revise_scared_expansion_manuscript.py','```','score保护已有结果，不允许直接覆盖；RAFT源码和权重可通过 scripts/acquire_raft_reference.py 恢复，opt_einsum==3.4.0 安装在 third_party/raft_runtime_deps 隔离目录。扩展的5对象属于原挑战训练分区，但未用于本研究冻结C3VD头的拟合；不能将它们声称为官方挑战测试集。若运行旧稿生成器，最后必须重新运行本修订脚本，再编译PDF。']
(R/'docs/SCARED_EXPANSION_AND_MANUSCRIPT_REVISION.md').write_text('\n'.join(report)+'\n');print('Revised both manuscripts; abstract words:',len(abstract.split()))
