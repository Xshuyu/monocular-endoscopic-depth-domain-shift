"""Build figures and revised drafts solely from the frozen multi-seed outputs."""
from pathlib import Path
import argparse
import hashlib
import json
import re
import shutil
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def interval(x): return f'[{x[0]:.3f}, {x[1]:.3f}]'

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=ROOT/'outputs/paper_strengthening')
    parser.add_argument('--update-manuscript',action='store_true')
    args=parser.parse_args(); out=args.run.resolve()
    summary=json.loads((out/'summary.json').read_text())
    protocol=json.loads((out/'protocol.json').read_text())
    manifest=json.loads((out/'manifest.json').read_text())
    if manifest['source_changed_during_run']: raise RuntimeError('Unstable source run')
    for name,h in manifest['outputs_sha256'].items():
        if sha(out/name)!=h: raise RuntimeError(f'Changed result: {name}')
    for name,h in manifest['source_sha256_at_start'].items():
        if sha(ROOT/name)!=h: raise RuntimeError(f'Changed source: {name}')
    cases=summary['n_unique_cases']; seeds=len(protocol['seeds'])
    table={(r['scale_mode'],r['method']):r for r in summary['summary']}
    scales=protocol['scale_modes']; methods=protocol['methods']
    labels=['PCA full','PCA q01–99','MAD–PCA q01–99','SO(3) no MAD','SO(3) always MAD','SO(3) adaptive MAD']
    short=['PCA full','PCA q','MAD-PCA q','SO3 never','SO3 always','SO3 adaptive']
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    colors=['#476B8A','#7FA8C9','#409A8A','#B88E4B','#BD705E','#76588A']
    for ax,scale in zip(axes.flat,scales):
        for i,method in enumerate(methods):
            r=table[scale,method]; lo,hi=r['mae_seed_block_bootstrap_95ci']
            ax.bar(i,r['mae_mm'],color=colors[i],width=.7)
            ax.plot([i,i],[lo,hi],color='black',lw=1.2)
            ax.plot([i-.08,i+.08],[lo,lo],color='black',lw=1.2)
            ax.plot([i-.08,i+.08],[hi,hi],color='black',lw=1.2)
        ax.set_xticks(range(6),short,rotation=25,ha='right',fontsize=9)
        ax.set_ylabel('Nominal major-diameter MAE (mm)')
        ax.set_title(scale.replace('_',' ').title()); ax.spines[['top','right']].set_visible(False)
    fig.suptitle(f'{cases} procedural scenes / {seeds} seeds; seed-block 95% intervals\nOracle depth is a separate reference condition',fontsize=12)
    fig.savefig(out/'multiseed_mae.pdf'); fig.savefig(out/'multiseed_mae.png',dpi=160); plt.close(fig)
    pairs=[r for r in summary['paired_comparisons'] if r['design']!='different_filter_policy_exploratory']
    fig,ax=plt.subplots(figsize=(10,5),layout='constrained')
    for i,r in enumerate(pairs):
        lo,hi=r['seed_block_bootstrap_95ci']
        ax.plot([lo,hi],[i,i],color='#476B8A',lw=2)
        ax.plot(r['mean_abs_error_difference_mm'],i,'o',color='#BD705E')
    ax.axvline(0,color='gray',ls='--'); ax.set_yticks(range(len(pairs)),[r['scale_mode']+' / '+r['design'] for r in pairs])
    ax.invert_yaxis(); ax.set_xlabel('SO(3) minus matched PCA absolute error (mm); negative favors SO(3)')
    ax.set_title('Matched filtering and reported quantiles; unadjusted seed-block 95% intervals')
    ax.spines[['top','right']].set_visible(False)
    fig.savefig(out/'matched_geometry.pdf'); fig.savefig(out/'matched_geometry.png',dpi=160); plt.close(fig)

    audit=json.loads((out/'dataset_audit.json').read_text())
    if audit['status']=='complete':
        base=ROOT/'datasets/kvasir-instrument/kvasir-instrument'
        for split,h in audit['split_file_sha256'].items():
            if sha(base/(split+'.txt'))!=h: raise RuntimeError('Source split changed')
        excluded=set(audit['suggested_train_exclusions_preserving_test'])
        train=(base/'train.txt').read_text().split()
        (out/'instrument_train_deduplicated.txt').write_text('\n'.join(s for s in train if s not in excluded)+'\n')
        shutil.copy2(base/'test.txt',out/'instrument_test_unchanged.txt')
    snapshot=out/'source_snapshot'/'lesion_3d_measurement'
    for name in manifest['source_sha256_at_start']:
        dest=snapshot/name; dest.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(ROOT/name,dest)
    link=snapshot/'datasets'
    if not link.exists(): link.symlink_to(ROOT/'datasets',target_is_directory=True)

    lines=['| Depth/scale condition | '+' | '.join(labels)+' |', '|---|'+'---:|'*6]
    for scale in scales:
        lines.append('| '+scale+' | '+' | '.join(f"{table[scale,m]['mae_mm']:.3f}" for m in methods)+' |')
    result_table='\n'.join(lines)
    lines=['| Scale | Matched filter | SO(3) − PCA MAE (mm) | 95% interval | Holm p |', '|---|---|---:|---|---:|']
    for r in pairs:
        lines.append(f"| {r['scale_mode']} | {r['design']} | {r['mean_abs_error_difference_mm']:.3f} | {interval(r['seed_block_bootstrap_95ci'])} | {r['wilcoxon_holm_12_comparisons_p']:.4f} |")
    paired_table='\n'.join(lines)
    fusion=table['fusion','so3_adaptive']; prior=table['prior','so3_adaptive']; tool=table['tool','so3_adaptive']
    fm=next(r for r in pairs if r['scale_mode']=='fusion' and r['design']=='matched_radial_MAD')
    chord=summary['rendered_truth_audit']['mean_absolute_chord_minus_nominal_mm']
    consistent=sum(r['wilcoxon_holm_12_comparisons_p']<.05 and r['mean_abs_error_difference_mm']<0 for r in pairs)
    worse=sum(r['wilcoxon_holm_12_comparisons_p']<.05 and r['mean_abs_error_difference_mm']>0 for r in pairs)
    conclusion=(f'Of eight comparisons with matched filtering and reported quantiles, {consistent} showed lower SO(3) error with Holm-adjusted p < 0.05, while {worse} showed higher error under the same criterion. '
                'This does not establish uniform superiority: effect sizes, signs and intervals must be considered in each condition. '
                'The optimization objective differs from the nominal diameter error, and scale error can compensate geometric bias.')
    command=f'/ssd-1/temp/kimicode/multmodel/venv-gpu/bin/python -B tests/paper_strengthening_benchmark.py --output /tmp/lesion_paper_repeat'
    report=f'''# 论文实验完善报告

本报告对应 `outputs/paper_strengthening/` 的固定协议运行，不混用旧版主基准数字。

## 已实施

- 主场景从单种子扩展到 {seeds} 个预先写入协议的种子，共 {cases} 个场景、{summary['n_method_rows']} 条方法结果。
- 新增两组严格匹配过滤策略和测量分位数的 PCA / SO(3) 比较；另保留原始自适应策略作探索比较。
- 逐例预测、按尺寸分层、混淆矩阵、各档召回率、CCC、真正的预测 R²、Bland–Altman 偏差和界限均有输出。
- 不确定性以整批随机种子为重采样单位；配对检验基于种子内平均误差差，12项比较统一Holm校正。区间本身未作多重比较校正。
- 场景生成使用局部随机数生成器，默认种子保持原场景不变，不重置调用者的全局随机状态。
- 修正旧统计函数对ICC的错误说明，保留接口兼容。没有把CCC称为标注一致性。
- 保留原始器械数据与划分，另输出建议去重划分（从训练集移除跨集完全重复图像，测试集不变）。未训练模型。
- 保存执行前协议、逐例结果、软件版本、代码与输出哈希和源代码快照。

## 结果：主径 MAE（mm）

{result_table}

## 匹配条件下的几何比较

{paired_table}

负值表示SO(3)误差更低；所有结果必须结合尺度条件解释，不因显著性筛选展示。

## 数据及真值边界

器械数据审计发现 {len(audit.get('byte_identical_cross_split_groups',[]))} 组训练/测试完全重复图像。建议划分仅解决字节相同的重复，不保证患者/视频级独立。

渲染表面的最大三维弦长与名义主径平均绝对差为 {chord:.3f} mm。两者是不同测量定义，本项用于揭示真值语义差异，未用其重新标定或修正算法。

主实验仍保留旧生成器的形态—尺寸混杂、共享光度假设和真值掩膜。多种子不能消除这些设计局限。Oracle还更换深度来源，不能解释为纯尺度因子。未获得真实病灶物理尺寸、多标注者重复测量或外部临床验证。

## 复现

从项目目录执行（新输出目录必须不存在）：

```bash
{command}
```

渲染报告：

```bash
/ssd-1/temp/kimicode/multmodel/venv-gpu/bin/python -B scripts/build_paper_strengthening_report.py
```

默认运行仅使用CPU光度回退路径，不下载权重、不训练。运行时约 {summary['seconds']/60:.1f} 分钟（本机本次结果）。每个方法使用同样的场景、掩膜和尺度输入，未进行模型选择。

`source_snapshot/lesion_3d_measurement/tests/paper_strengthening_benchmark.py` 保存了本次代码，可在以后代码改变时用于重放；数据链接仅供独立数据审计，主测量数据由种子生成。

## 仍需真实资源的工作

见 `docs/PHYSICAL_VALIDATION_PROTOCOL.md`。没有用合成数据替代医生标注ICC、真实毫米真值或患者级外部验证。当前主文是以本次数据为依据的修订稿，投稿前仍需完成相关工作引用、作者声明核实及真实验证决策。
'''
    (ROOT/'docs/PAPER_STRENGTHENING_REPORT.md').write_text(report)
    draft=f'''# MetricEndo: Separating Scale and Robust Geometry Errors in Procedural Monocular Lesion Measurement

Shuyu Xiong and Yongtao Deng

Revision scope: this draft replaces the untraceable primary numbers in the earlier manuscript. It reports the new frozen multi-seed run only. Older curvature, heat-method and real-image timing figures remain archived, but are not used to substantiate the claims below. Related-work citations and author declarations require author review before submission.

## Abstract

Monocular lesion measurement requires both a physical scale and a geometric extent estimator. We evaluate these stages in a fixed procedural benchmark, using an explicitly identified photometric depth fallback and ground-truth masks. Across {seeds} seeds and {cases} scenes, six geometry configurations were evaluated under three estimated-depth scale strategies and a separate oracle-depth reference. Filtering and reported endpoint quantiles were matched in direct PCA/SO(3) comparisons. With adaptive SO(3), nominal major-diameter MAE was {prior['mae_mm']:.3f} mm for the lesion-region distance prior, {tool['mae_mm']:.3f} mm for tool-only scaling and {fusion['mae_mm']:.3f} mm for equal fusion. These are procedural errors, not established clinical accuracy. The experiments quantify conditional trade-offs and expose limitations in scale calibration and ground-truth semantics; they do not demonstrate general SO(3) superiority or clinical validity.

## 1. Motivation and contribution

Physical size cannot be recovered from image pixels without scale assumptions. Within this framework, a working-distance prior and a tool-width cue can each be biased, while filtering and endpoint trimming can alter the geometric quantity reported. The contribution evaluated here is a reproducible separation of these effects, with explicit matched comparisons and negative results. Novelty relative to external methods has not been established by this experiment, and no state-of-the-art claim is made.

## 2. Methods

### 2.1 Inputs and depth/scale conditions

Each scene supplies RGB, dense depth, lesion mask, tool mask and nominal dimensions. The legacy renderer is sampled using seeds {protocol['seeds'][0]}–{protocol['seeds'][-1]}, with {protocol['samples_per_seed']} scenes per seed. No model is trained. The active depth path is the photometric fallback, not learned Depth Anything inference. Back-projection uses the same 640 x 480 pinhole model and 140-degree field-of-view assumption for every method. Estimated depth is calibrated using a 30-mm lesion-ROI median prior, a nominal 2.4-mm tool width, or fixed equal fusion. The oracle condition substitutes generated depth and is reported separately from deployable scale strategies.

### 2.2 Geometry comparisons

Six estimators are reported: PCA full extent, PCA 1st–99th-percentile extent, radial-MAD-filtered PCA with the same extent quantiles, and SO(3) optimization with no, always-on, or adaptive radial MAD filtering. The no-filter SO(3)/PCA and always-MAD SO(3)/PCA pairs share input support and reported quantiles. Adaptive SO(3) versus always-MAD PCA is exploratory because the filtering policies differ. The SO(3) orientation objective uses 2nd–98th-percentile box volume, while reported extents use the 1st–99th percentiles; optimizing volume need not minimize nominal diameter error. No boundary correction or fusion-weight tuning is performed.

### 2.3 Endpoints and statistics

The primary endpoint is absolute major-diameter error against nominal generator size, in millimetres. RMSE, signed bias, MRE, CCC, predictive R² and descriptive Bland–Altman limits are secondary outputs. Size-bin confusion matrices and recall describe dimension categories, not clinical risk. Seed-average MAEs and paired differences are resampled as whole blocks for 10,000 percentile bootstrap draws. Wilcoxon tests use the ten seed-average differences; Holm correction covers all twelve prespecified script comparisons. Intervals are unadjusted and conditional on this generator. The protocol was saved before this run, but this is an extension of an already explored synthetic framework, not a prospectively registered clinical study.

## 3. Results

### 3.1 Multi-seed nominal-diameter error

{result_table}

All values are MAE in mm. Adaptive SO(3) fusion MAE has a seed-block 95% interval of {interval(fusion['mae_seed_block_bootstrap_95ci'])} mm. Full per-seed results and all confidence intervals are in the run artifacts.

![Multi-seed errors](../outputs/paper_strengthening/multiseed_mae.png)

### 3.2 Matched geometric comparisons

{paired_table}

{conclusion}

![Matched differences](../outputs/paper_strengthening/matched_geometry.png)

### 3.3 Reference and dataset audit

The sampled rendered-surface maximum chord differed from nominal major diameter by {chord:.3f} mm on average in absolute value. These are distinct estimands; this diagnostic is not a replacement clinical reference. Three byte-identical train/test image groups were identified in the bundled instrument dataset; proposed train-only exclusions are supplied without modifying original splits. That dataset was not used to train the evaluated measurement pipeline.

## 4. Discussion and limitations

Scale-dependent error differences must not be interpreted as an isolated optimizer benefit. A geometric shrinkage bias may offset a positive scale bias, producing lower end-to-end error without more accurate geometry. The matched comparisons quantify this possibility but do not resolve calibration ambiguity. The prior/fusion findings are tied to the lesion-ROI protocol and must not be mixed with earlier whole-image-prior results.

The renderer couples morphology with size bins and uses lighting assumptions related to the photometric inverse. Masks are supplied as truth, and nominal dimensions are not guaranteed to match the rendered surface after projection and depth modification. Multiple seeds improve assessment of sampling variability but cannot remove these structural limitations. The ten blocks are synthetic repetitions, not patients or sites. No clinical millimetre accuracy, learned-depth performance, segmentation-to-measurement accuracy, inter-rater reliability, Paris classification or clinical risk benefit is established. Heat and curvature components are outside this primary endpoint and require separate validation.

## 5. Conclusion

The study provides a reproducible, explicitly scoped assessment of scale and robust geometry choices. Its supported contribution is conditional error analysis and matched comparison, rather than universal superiority or clinical deployment readiness. Physical phantom measurements, independently calibrated images and repeated reference annotations remain necessary to extend the claims.

## Reproducibility

Entry point: `tests/paper_strengthening_benchmark.py`. The run directory contains protocol, per-case CSV, per-seed/stratified statistics, dataset audit, environment versions, hashes and source snapshots. No external weights were downloaded. The report and figures are generated from these artifacts by `scripts/build_paper_strengthening_report.py`.
'''
    (ROOT/'manuscript/VERIFIED_REVISED_DRAFT.md').write_text(draft)
    if args.update_manuscript:
        (ROOT/'manuscript/MANUSCRIPT_CBM.md').write_text(draft)
        backup=ROOT/'outputs/paper_revision_backup_20260926/manuscript/main_cbm.tex'
        original=backup.read_text(); header=original.split('\\begin{abstract}')[0]
        old_title=header[header.index('\\title{'):header.index('\n\\author')]
        header=header.replace(old_title,'\\title{MetricEndo: Separating Scale and Robust Geometry Errors in Procedural Monocular Lesion Measurement}')
        abstract=draft.split('## Abstract\n\n')[1].split('\n\n## 1.')[0]
        def escape(s):
            for a,b in [('&','\\&'),('%','\\%'),('_','\\_'),('#','\\#')]: s=s.replace(a,b)
            return s.replace('SO(3)','SO(3)').replace('–','--').replace('−','$-$').replace('²','$^2$')
        body=header+'\\begin{abstract}\n'+escape(abstract)+'\n\\end{abstract}\n\\begin{keyword}monocular measurement \\sep scale calibration \\sep robust geometry \\sep procedural evaluation\\end{keyword}\n\\end{frontmatter}\n'
        # Convert this restricted, generated Markdown draft; tables are explicit below.
        prose=draft.split('## 1. Motivation and contribution\n\n',1)[1]
        body+='\\section{Motivation and contribution}\n'
        in_table=False
        for line in prose.splitlines():
            if line.startswith('|'):
                if not in_table:
                    is_pair='Matched filter' in line
                    columns=5 if is_pair else 7
                    body+='\\begin{table}[ht]\\centering\\scriptsize\n\\resizebox{\\textwidth}{!}{\\begin{tabular}{'+'l'*columns+'}\\toprule\n'
                    in_table=True
                if set(line.replace('|','').replace(':','').replace(' ',''))=={'-'}: continue
                cells=[escape(x.strip()) for x in line.strip('|').split('|')]
                body+=' & '.join(cells)+' \\\\\n'
                continue
            if in_table:
                body+='\\bottomrule\\end{tabular}}\\end{table}\n'; in_table=False
            if line.startswith('## '): body+='\\section{'+escape(re.sub(r'^\d+\.\s*', '', line[3:]))+'}\n'
            elif line.startswith('### '): body+='\\subsection{'+escape(re.sub(r'^\d+\.\d+\s*', '', line[4:]))+'}\n'
            elif line.startswith('!['):
                filename='matched_geometry' if 'Matched differences' in line else 'multiseed_mae'
                body+='\\begin{figure}[ht]\\centering\\includegraphics[width=\\textwidth]{../outputs/paper_strengthening/'+filename+'.pdf}\\caption{'+('Matched geometry differences; negative values favor SO(3). Intervals are unadjusted.' if filename=='matched_geometry' else 'Multi-seed nominal-diameter errors. Oracle depth is a separate reference condition.')+'}\\end{figure}\n'
            else: body+=escape(line.replace('`',''))+'\n'
        if in_table: body+='\\bottomrule\\end{tabular}}\\end{table}\n'
        declarations=original.split('\\section*{Ethics, Funding and Conflict of Interest}')[1].split('\\end{document}')[0]
        body+='\\section*{Ethics, Funding and Conflict of Interest}\n'+declarations+'\n\\end{document}\n'
        (ROOT/'manuscript/main_cbm.tex').write_text(body)
    artifact_hashes={str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'docs/PAPER_STRENGTHENING_REPORT.md',ROOT/'manuscript/VERIFIED_REVISED_DRAFT.md',Path(__file__).resolve()]}
    if args.update_manuscript:
        for name in ('manuscript/main_cbm.tex','manuscript/MANUSCRIPT_CBM.md'):
            artifact_hashes[name]=sha(ROOT/name)
    for p in out.glob('*'):
        if p.is_file() and p.name not in ('manifest.json','report_manifest.json'): artifact_hashes[str(p.relative_to(ROOT))]=sha(p)
    (out/'report_manifest.json').write_text(json.dumps(artifact_hashes,indent=2)+'\n')
    print(result_table); print(paired_table); print('Wrote report, figures and revised draft.')

if __name__=='__main__': main()
