"""Write exploratory supplement and MBEC subscription-route working draft."""
from pathlib import Path
import json,re,csv
R=Path(__file__).resolve().parents[1];O=R/'outputs/object_disjoint_calibration';M=R/'manuscript_cmig'
a=json.loads((O/'summary.json').read_text())
def get(model,head,budget,cap):return next(x for x in a if (x['model'],x['head'],x['budget'],x['cap_mm'])==(model,head,budget,cap))
rows=[];md=[]
for m,label in [('small','DA V2 Small'),('large','DA V2 Large'),('endodac','EndoDAC')]:
 for head,hlabel in [('affine_ls','Affine LS'),('radial_ls','Radial LS')]:
  old=get(m,'source_'+head,0,100);new=get(m,'adapted_'+head,16384,100);wide=get(m,'adapted_'+head,16384,250)
  rows.append(f"{label}, {hlabel} & {old['mae_mm']:.3f} & ${new['mae_mm']:.3f}\\pm{new['seed_sd_mm']:.3f}$ & ${wide['mae_mm']:.3f}\\pm{wide['seed_sd_mm']:.3f}$ "+r'\\')
  md.append(f"| {label} + {hlabel} | {old['mae_mm']:.3f} | {new['mae_mm']:.3f} ± {new['seed_sd_mm']:.3f} | {wide['mae_mm']:.3f} ± {wide['seed_sd_mm']:.3f} |")
tex=r'''\subsection{Exploratory object-disjoint recalibration}
After observing the SERV-CT transfer failure, we evaluated whether reference-assisted recalibration on one specimen transfers to the other. This is a post-hoc experiment on already observed data, not a new blind validation. In each of two reciprocal folds, all eight views of one specimen supplied calibration and all eight views of the other supplied evaluation. Object and camera-configuration differences remain confounded. No evaluated object's pixels were used to fit its head or feature normalization.

For each frozen backbone, global affine and radial LS were fitted using 1,024, 4,096 or 16,384 total reference pixels. Five sampling seeds (202610010--202610014) drew equal quotas per training image without replacement from the fixed 8,192-valid-pixel pool per image. All methods shared the same sampled positions, with nested subsets across budgets. This produced 180 fitted heads. No regularization or model selection was performed; all budgets and seeds were retained. Evaluation used all native reference-valid pixels, averaging views within specimen and then specimens. Seed standard deviations quantify calibration-pixel sampling only and do not increase the independent sample size beyond two.

The primary prediction interval remained $[1,100]$ mm. A separate, prespecified-for-this-post-hoc-analysis $[1,250]$ mm sensitivity condition stayed within the reference encoding's representable range. Both unchanged source-domain heads and recalibrated heads were evaluated under each interval, so output-range changes were not conflated with recalibration. The original frozen transfer results were preserved.

\begin{table}[htbp]\centering\small
\caption{Object-disjoint recalibration at 16,384 pixels/head: equal-specimen MAE (mm). Recalibrated entries show mean $\pm$ SD across five sampling seeds; these are not population confidence intervals. Full budget and range results are provided with the executable analysis.}\label{tab:recalibration}
\resizebox{\textwidth}{!}{\begin{tabular}{lrrr}\toprule
Backbone/head & Source, cap 100 & Recalibrated, cap 100 & Recalibrated, cap 250 \\\midrule
@@ROWS@@
\bottomrule\end{tabular}}
\end{table}

Recalibration markedly reduces error in this two-object experiment (Table~\ref{tab:recalibration}). Under the unchanged 100 mm cap, Small affine LS falls from 56.245 to 19.408 mm, while EndoDAC affine LS falls from 53.295 to 17.691 mm. However, radial terms do not consistently outperform affine calibration: they slightly improve the aggregate Small and Large results and worsen EndoDAC. Allowing a 250 mm cap does not remove the residual errors. These results support the practical importance of domain-matched reference calibration within this dataset, but do not prove accurate clinical sizing, optimal calibration requirements, or reliable transfer to a third independent specimen. The two reciprocal folds are not independent replications of a clinical population.
'''.replace('@@ROWS@@','\n'.join(rows))
(M/'object_recalibration.tex').write_text(tex)
p=R/'scripts/revise_cmig_radial.py';s=p.read_text()
if 'object_recalibration.tex' not in s:s=s.replace(r'\input{independent_validation.tex}',r'\input{independent_validation.tex}'+'\n'+r'\input{object_recalibration.tex}')
p.write_text(s)
report='''# 补充实验与非OA投稿选择

## 已实际补做

SERV-CT两标本互换：用一个标本的全部8幅图像拟合，评估另一个标本的全部8幅图像，再反向。三种冻结骨干×两种标定头×两折×三档标定预算×五次随机采样，共180次拟合、3072条逐图像/范围条件评分。所有条件使用相同标定位置和原生有效像素。未训练神经网络。

这是看过原测试结果后设计的探索性分析，不是新的盲测。按对象划分防止当前拟合使用对应测试对象，但不能消除事后设计偏差。两个对象、两折不等于两个独立临床队列。

| 方法 | 原标定，100mm上限 | 跨对象重标定，100mm上限 | 跨对象重标定，250mm上限 |
|---|---:|---:|---:|
'''+ '\n'.join(md)+'''

单位mm，重标定结果为五种采样种子的均值±标准差。额外250mm范围仅为明确标注的敏感性条件，原模型也在相同范围重新评分，全部结果保留；不能把改范围当作算法改进。

结论：重新标定显著改善本数据集内的跨对象误差，但仍不足以支持临床尺寸测量；径向项没有稳定优于简单仿射。论文主线应聚焦“标定迁移失效、参考辅助重标定和空间项的适用边界”。

## 非OA选择

首选目标：Medical & Biological Engineering & Computing（MBEC），选择 **Subscription publishing model**。

官方出版说明明确这是混合期刊，可选择订阅路线，订阅路线不收APC。也就是说文章可以非OA发表，并非该期刊所有文章都非OA。这里确认的是无需OA文章处理费，不承诺任何情况下绝无其他费用；须核对机构/资助要求及最终投稿规则。

官方链接：
- https://link.springer.com/journal/11517/how-to-publish-with-us
- https://link.springer.com/journal/11517/aims-and-scope
- https://link.springer.com/journal/11517 （官网列SCIE）

选择依据：医学图像处理、医学成像和工程建模符合范围；稿件应突出严谨的工程验证和实际标定问题，而不是宣称新SOTA或临床尺寸精度。这是有条件的目标选择，不是录取保证。目前仅两个独立标本、方法贡献有限、重标定分析为事后设计，仍有拒稿风险。Medical Engineering & Physics 官方详情页本次返回403，未将其未经核实的收费政策作为推荐依据。

## 交付和剩余缺口

- `scripts/run_object_disjoint_calibration.py`及`outputs/object_disjoint_calibration/`保留协议、180个头、逐图像/逐对象记录和全部预算/范围汇总。
- 统一使用 `/ssd-1/temp/kimicode/multmodel/venv-gpu/bin/python`。
- `manuscript_mbec/main_mbec.tex/pdf`为面向MBEC的工作稿，尚未声称满足全部投稿格式与声明要求，也未提交。
- 还缺第三个及更多未参与设计的独立对象，用于冻结方案后的真正验证；真实病灶尺寸及原始观察者记录仍缺失。
'''
(R/'docs/NON_OA_SELECTION_AND_RECALIBRATION.md').write_text(report)
print('Generated recalibration section and non-OA selection report')
