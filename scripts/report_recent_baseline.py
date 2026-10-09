from pathlib import Path
import json,re,hashlib,collections
R=Path(__file__).resolve().parents[1];O=R/'outputs/recent_baselines'
s=json.loads((O/'pcc_summary.json').read_text());old=json.loads((R/'outputs/radial_confirmatory/test_summary.json').read_text());serv=json.loads((R/'outputs/independent_validation/summary.json').read_text())
lookup=lambda data,method:next(x['mae_mm'] for x in data if x['method']==method and x.get('support','all_reference')=='all_reference')
audit={}
for name in ['monopcc','endoufm']:
 p=O/(name+'_splits_endovis_train_files.txt');text=p.read_text();audit[name]={'training_lines':len(text.splitlines()),'training_subjects':sorted(set(map(int,re.findall(r'dataset_?(\d+)',text)))),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'default_test_median_scaling':True}
audit['delightmono']={'availability':'Official linked GitHub repository empty; API 409; no released implementation retrieved'}
(O/'availability_and_overlap_audit.json').write_text(json.dumps(audit,indent=2))
tex=r'''\subsection{Post-hoc contemporary monocular baseline}
To address the limited coverage of recent methods, we evaluated the official EndoDAC+PCC checkpoint linked by the MonoPCC authors \citep{Wang2025}. This is their PCC-on-EndoDAC variant, not the original MonoViT-based MonoPCC model. The 99.09-million-parameter network was loaded strictly, with the released checkpoint resolution of $256\times320$. We used the existing 52 development frames and identical 16,384 scene-balanced calibration samples to fit global and radial least-squares heads. Both heads used the same $[1,100]$ mm prediction bounds and native-pixel support as the existing comparisons. No neural-network retraining, target-reference median scaling or target-based parameter selection was performed. This controls calibration data and evaluation, but not original pretraining data or compute.

\begin{table}[htbp]
\centering\small
\caption{Added contemporary baseline under the common source-calibration protocol. MAE in mm; C3VD averages four videos (48 frames), SERV-CT averages two specimens (16 views). This is a post-hoc comparison on previously examined cohorts, not a fresh confirmatory test.}
\begin{tabular}{lrr}\toprule
Method & C3VD & SERV-CT \\\midrule
'''
for label,c,v in [('DA V2 Small + global LS',lookup(old,'affine_ls'),lookup(serv,'small_affine_ls')),('DA V2 Small + radial LS',lookup(old,'radial_ls'),lookup(serv,'small_radial_ls')),('DA V2 Small + radial ridge',lookup(old,'radial_ridge'),lookup(serv,'small_radial_ridge')),('EndoDAC+PCC + global LS',lookup([x for x in s if x['cohort']=='c3vd'],'pcc_affine_ls'),lookup([x for x in s if x['cohort']=='servct'],'pcc_affine_ls')),('EndoDAC+PCC + radial LS',lookup([x for x in s if x['cohort']=='c3vd'],'pcc_radial_ls'),lookup([x for x in s if x['cohort']=='servct'],'pcc_radial_ls'))]:tex+=f'{label} & {c:.3f} & {v:.3f} '+r'\\'+'\n'
tex+=r'''\bottomrule\end{tabular}\end{table}

The radial mapping also reduced MAE within the PCC-adapted backbone, but substantial SERV-CT transfer error remained. DA V2 Small with radial ridge had lower C3VD MAE than the added PCC radial baseline, while its SERV-CT MAE was slightly higher. These mixed outcomes support neither uniform superiority nor clinical metric accuracy. Published MonoPCC numbers use a different evaluation, including default test-reference median scaling, and are not directly comparable to this table.

A reproducibility audit further found all nine SCARED dataset identifiers in the public MonoPCC and EndoUFM training lists. We therefore did not treat SCARED as an independent-object test of these checkpoints. The repository linked by DeLightMono \citep{Ou2026} was empty at the audit date (2 October 2026), so no DeLightMono reproduction is claimed. Additional clean-split domain comparisons remain necessary. Absence of SERV-CT or C3VD identifiers from a SCARED training list is not proof that every stage of pretraining and model selection was independent of those datasets.
'''
for folder,file in [('manuscript_cmig','main_cmig.tex'),('manuscript_mbec','main_mbec.tex')]:
 d=R/folder;(d/'recent_baseline.tex').write_text(tex);p=d/file;t=p.read_text()
 if r'\input{recent_baseline.tex}' not in t:t=t.replace(r'\section{Discussion and limitations}',r'\input{recent_baseline.tex}'+'\n\n'+r'\section{Discussion and limitations}')
 p.write_text(t)
report='''# 近期强基线补充与公平性审查

本次实际完成官方 EndoDAC+PCC 组合模型推理：52 帧开发集、48 帧 C3VD 测试、16 帧 SERV-CT，共 116 帧；严格加载权重，无神经网络训练。不是原始 MonoViT 版 MonoPCC 复现，也不是重新复现其论文表格。使用已有 EndoDAC 官方模型实现和作者发布的 PCC 权重，全部状态键严格匹配；模型结构未改动。

## 统一与未统一的条件

统一：52 帧标定数据、确定性 16384 像素采样、开发集内标准化、全局/径向最小二乘、1–100 mm 输出范围、测试掩膜、指标和组级平均；不使用测试真值中位数缩放。

未统一：基础网络参数量、原始预训练数据与算力、模型特定输入预处理。测试对象此前已观察，新增实验属于事后比较，不应重新标称独立盲测。SERV-CT 仍只有两个标本。不能由这张表宣布全面超过 2026 SOTA。

## 实测 MAE（mm）

| 方法 | C3VD 4 视频 / 48 帧 | SERV-CT 2 标本 / 16 帧 |
|---|---:|---:|
'''
for m in ['pcc_affine_ls','pcc_radial_ls']:report+=f"| {m} | {lookup([x for x in s if x['cohort']=='c3vd'],m):.3f} | {lookup([x for x in s if x['cohort']=='servct'],m):.3f} |\n"
report+='''
现有 Small radial ridge 对应 8.899 / 55.537 mm；PCC radial LS 对应 9.523 / 55.264 mm。C3VD 上现有方法较低，SERV-CT 上 PCC 略低；双方外部尺度误差仍大。该对照补上近期方法实测，但不能消除所有方法学短板。

## 官方来源与可复现性

- MonoPCC（2025）：https://github.com/adam99goat/MonoPCC ，固定提交 ed2f13f7fd46c3bcc2369e78f8751688970605a1。README 提供 EndoDAC+PCC 权重；本地记录完整权重 SHA256。其训练清单覆盖 SCARED 1–9，默认评估使用真值中位数缩放。因此不在本次 SCARED 独立对象表中添加该权重。
- EndoUFM：https://github.com/RealMindyY/EndoUFM ，固定提交 3db0ec9302e4216ed5be0a400bf52403897f3a4f。同样存在 SCARED 对象重叠和默认真值缩放。已完成协议审查，未宣称已运行其模型。
- DeLightMono（2026）：https://github.com/ComgLq24/AAAI-2026-DeLightMono 。论文所链仓库在审查时为空，API 返回 409 Git Repository is empty。不能将不可复现写成性能不佳。
- 2026 年内参引导方法已在相关工作讨论；本轮检索未确认可运行的官方实现，未编造数值。

## 复现与证据

使用 /ssd-1/temp/kimicode/multmodel/venv-gpu/bin/python scripts/run_recent_pcc_baseline.py，依次 freeze、inference、calibrate、score。现有 freeze/calibrate/score 文件有防覆盖保护；复查可直接读取已保存预测与指标，不要删除冻结证据。推理日志 pcc_inference.log；逐帧 pcc_per_frame.csv；对象级 pcc_per_object.json；汇总 pcc_summary.json；pcc_protocol.json 保存权重和脚本哈希。评价包含原掩膜和 1–100 mm 次要支持集。

论文新增 recent_baseline.tex，补充复现细节、同口径数值以及负面结果。未改动原始算法、原冻结头和历史实验结果。若使用旧手稿生成器重建，先运行参考文献修订脚本，再运行 scripts/report_recent_baseline.py 恢复本段。
'''
(R/'docs/RECENT_BASELINE_VALIDATION.md').write_text(report)
print('Wrote manuscripts and audit report')
