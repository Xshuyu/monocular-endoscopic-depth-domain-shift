"""Generate manuscript supplement and reviewer assessment from sensitivity outputs."""
from pathlib import Path
import json,csv
R=Path(__file__).resolve().parents[1];O=R/'outputs/radial_sampling_sensitivity';M=R/'manuscript_cmig'
a=json.loads((O/'summary.json').read_text())
with (O/'test_video_influence.csv').open() as f:influence=list(csv.DictReader(f))
methods=['affine_ls','affine_ridge','radial_ls','radial_ridge']
rows=[]
for b in [4096,16384,65536]:
 vals=[next(x for x in a if x['budget']==b and x['method']==m) for m in methods]
 rows.append(f'{b:,}'+' & '+' & '.join(f"${x['mean_sampled_mae_mm']:.3f}\\pm{x['seed_sd_mm']:.3f}$" for x in vals)+r' \\')
tex=r'''\subsection{Post-hoc sampling and test-composition sensitivity}
After inspecting the primary results, we performed a separate sensitivity study restricted to development fitting. Ten random seeds (202609280--202609289) and total training budgets of 4,096, 16,384 and 65,536 pixels were evaluated in the same six outer scene folds, giving 720 fits across four methods. Sampling was without replacement within each training scene from the original 8,192-valid-pixel-per-frame pool, with equal scene quotas. Families shared identical samples; larger budgets extended the smaller sampled subsets within a seed and fold. Feature normalization was recomputed from each training sample only. Each ridge method retained its original outer-fold selected regularization; this analysis therefore conditions on that choice rather than repeating the full nested selection procedure.

Evaluation used the fixed 8,192-pixel pool in each held-out development frame and averaged over frames and scenes. These sampled endpoints differ from the full-native primary endpoint. The reported standard deviation is over ten sampling seeds, not a confidence interval, patient uncertainty estimate, or repeated backbone-training result. No new heads were selected for external testing.
\begin{table}[htbp]\centering\small
\caption{Post-hoc development sensitivity: sampled MAE in mm, mean $\pm$ sampling-seed SD. All six held-out scene scores contribute to each seed mean.}\label{tab:sampling}
\resizebox{\textwidth}{!}{\begin{tabular}{rrrrr}\toprule
Pixels/head & Affine LS & Affine ridge & Radial LS & Radial ridge \\\midrule
@@ROWS@@
\bottomrule\end{tabular}}
\end{table}

Radial LS and radial ridge improve on affine LS in all ten seed means at each budget. At 16,384 pixels, the paired radial-ridge minus affine-LS difference is $-1.619\pm0.057$ mm across seeds. Radial-ridge sampled MAE changes only from 10.732 to 10.732 mm (rounded) when the budget increases from 16,384 to 65,536, while its seed SD falls from 0.024 to 0.018 mm. Within this fixed development pool, random pixel selection is therefore unlikely to explain away the mean development benefit. This result does not establish stability across new scenes or undo the observed transfer failure.

We separately recomputed the frozen test comparison after omitting each video, without refitting. The remaining-three-video mean difference versus affine LS ranges from $-1.265$ to $+1.526$ mm for radial LS and from $-1.009$ to $+1.698$ mm for radial ridge. Omitting either cecum video yields a positive difference (worse radial error); omitting sigmoid or transverse yields a negative difference. These are composition diagnostics, not confidence limits or a rule for excluding videos. All four videos remain in the primary result. Together, the analyses distinguish pixel-sampling stability within development from sensitivity to the anatomical composition of the test set.
'''.replace('@@ROWS@@','\n'.join(rows))
(M/'sampling_results.tex').write_text(tex)
p=R/'scripts/revise_cmig_radial.py';s=p.read_text()
anchor=r'\section{Implications for lesion measurement}'
if 'sampling_results.tex' not in s:s=s.replace(anchor,r'\input{sampling_results.tex}'+'\n\n'+anchor)
old='Our earlier fixed-coefficient evaluation reported improvement on 92 already observed frames. Reproduction confirmed these scores, but the supplied scoring script contained constants rather than their fitting procedure. It was therefore insufficient to verify how the coefficients or regularization had been selected. The present study establishes a separate, explicitly defined fitting protocol. We retain the historical constants as secondary controls, without retroactively assigning them the provenance of the new experiment.'
new='The central question is whether an inexpensive spatial calibration term improves depth beyond a matched global affine model and whether that improvement survives scene transfer. We separate three issues: the contribution of radial features, the incremental effect of ridge regularization, and dependence on scene composition. Historical fixed heads are retained only as secondary controls because their original fitting records are unavailable. The primary comparison uses a fully specified fitting procedure and identical calibration budgets.'
s=s.replace(old,new)
anchor='A clinical sizing claim would require paired physical size references'
s=s.replace(anchor,'The post-hoc sampling study shows that a reproducible development gain can coexist with uncertain transfer: increasing the number of calibration pixels is not equivalent to increasing anatomical or device diversity. The leave-one-video-out analysis is especially sensitive to which segments are represented. Neither diagnostic establishes the cause of the shift; matched camera, texture and depth-distribution experiments are needed to separate these factors. A four-parameter extension remains methodologically modest, and the present comparisons do not establish superiority over contemporary domain-specific depth systems.\n\n'+anchor) if 'A four-parameter extension remains' not in s else s
p.write_text(s)
report='''# 审稿视角评估及补充实验（2026-09-28）

## 判断

实验有意义：它检验低维空间标定在内镜深度中的收益，并显示开发集收益不一定迁移。这个结果有工程与方法学价值，尤其能防止把径向相关性解释成已证明的光学机制。

若我以当前证据审阅 CMIG 常规原创研究稿，我会倾向拒稿或要求实质性补充后重新投稿，而不是直接录取。这是审稿判断，不是对编辑决定的预测。主要障碍是创新幅度有限、外部样本仅四个相关视频、收益随解剖组成改变、缺乏同一最终测试协议上的强领域基线，以及没有真实病灶尺寸终点。诚实呈现负结果可以提高可信度，但不能自动弥补贡献和验证不足。

CMIG 官方定位强调医学影像中的计算与可视化研究。主题匹配不等于创新和实验证据已达到录取标准。参考：https://shop.elsevier.com/journals/computerized-medical-imaging-and-graphics/0895-6111 （2026-09-28 查阅）。

## 本次实际完成的优化

1. 开发集采样稳定性：10 个随机种子 × 3 档标定像素预算 × 6 个外层场景划分 × 4 个方法，共 720 次低维拟合。没有训练神经网络。
2. 各方法用相同像素、训练折单独拟合标准化，测试场景不参与拟合。正则强度固定为原外层折的选择；不是重新执行 10 次嵌套搜索。
3. 逐个去掉一个视频，重算冻结测试结果，检验少量视频构成对平均结论的影响；没有据此删去任何视频。
4. 主文新增采样预算表、后验分析设计与局限；引言聚焦科学问题，讨论区分开发集采样稳定性与跨域泛化。
5. 保留全部原始主实验结果和冻结代码。新增结论明确标为事后分析，不声称新的盲测。

## 新实验结果

开发集 16,384 像素预算下，10 次采样的等场景抽样 MAE（均值±种子标准差）：

| 方法 | mm |
|---|---:|
| Affine LS | 12.351 ± 0.045 |
| Affine ridge | 12.312 ± 0.046 |
| Radial LS | 10.803 ± 0.031 |
| Radial ridge | 10.732 ± 0.024 |

注意：评估是固定每帧 8192 有效像素的抽样终点，不是主文全原生有效像素终点；不可拿 10.732 与主文 10.800 解释为算法精度提升。所有预算与种子下，径向方法的六场景平均值均优于仿射，但该结果只说明固定开发数据上的采样稳定性。

逐视频移除后，Radial ridge 相对 Affine LS 的差值范围为 −1.009 至 +1.698 mm。去掉任一盲肠视频，径向方法更差；去掉乙状或横结肠视频，径向方法更好。主结论依赖测试构成，不能仅依据整体均值声称优越。

## 仍必须补的证据，按优先级

| 优先级 | 工作 | 验收标准/限制 |
|---|---|---|
| 1 | 新相机或独立物理模型的未见数据 | 冻结方案后一次评估，按物体/设备分组；增加帧数不能替代增加独立对象。当前测试已看过，不能重新命名为盲测。 |
| 2 | 领域专用强基线与可竞争的标定方案 | 同一图像、深度单位、相机处理、有效区域、校准数据预算；报告训练/参数资源差异；领域基线需先冻结后评估新的未见集。 |
| 3 | 真实物体尺寸终点 | 标定射线、真实毫米参照、独立对象划分、重复标注；根据实际端点评估 MAE、偏差和一致性，不引用别人的 ICC 充当本实验结果。 |
| 4 | 更有依据的方法改进 | 在开发数据评估局部深度分布与相机径向项的混淆、正则网格边界及拟合/选择损失不一致；任何新算法仍需新的独立验证，不能挑当前测试最优者。 |
| 5 | 真实部署成本与失效检测 | 相同硬件实测时延/内存，检查极端深度、裁剪变化、掩膜误差；理论 O(HW) 不等于已达到实时。 |

本次完成的是实验可靠性、可复现性与论文叙事的优化；没有宣称新的算法超越基线，也没有把尚未取得的数据、物理测量或临床一致性写成完成。

## 复现

脚本：`scripts/strengthen_radial_review.py`。使用 `/ssd-1/temp/kimicode/multmodel/venv-gpu/bin/python` 执行。输出默认 `outputs/radial_sampling_sensitivity`，拒绝覆盖已有目录。该目录包含 protocol、720 个模型记录、逐折/逐种子结果、汇总、配对差、视频影响分析、可编辑 SVG 图以及校验结果。
'''
(R/'docs/REVIEWER_ASSESSMENT_AND_OPTIMIZATION.md').write_text(report)
print('Manuscript supplement and reviewer report written')
