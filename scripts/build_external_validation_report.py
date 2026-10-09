"""Render external-data audits, measured results and a manuscript supplement."""
import csv,hashlib,json,re,shutil,sys
from pathlib import Path
import cv2,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/c3vdv2_external_validation'
LABELS={'constant_30mm':'Constant 30 mm','constant_calibration_median':'Calibration median',
        'photometric_raw':'Photometric (original)','photometric_calibration_scale':'Photometric + calibration scale',
        'depth_anything_v2_calibration_inverse_affine':'Depth Anything V2 + calibration'}

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 s=json.loads((OUT/'summary.json').read_text());protocol=json.loads((OUT/'protocol.json').read_text())
 test=[x for x in s['summary'] if x['split']=='test'];by={x['method']:x for x in test}
 geo=json.loads((ROOT/'outputs/consistent_surface_reference/summary.json').read_text())
 videos=list(csv.DictReader((OUT/'per_video.csv').open()));scenes=sorted({x['scene'] for x in videos if x['split']=='test'})
 a={r['scene']:float(r['mae_mm']) for r in videos if r['split']=='test' and r['method']=='depth_anything_v2_calibration_inverse_affine'}
 b={r['scene']:float(r['mae_mm']) for r in videos if r['split']=='test' and r['method']=='photometric_raw'}
 diffs=np.array([a[k]-b[k] for k in scenes]);draws=np.random.default_rng(20260926).choice(diffs,(10000,len(diffs)),replace=True).mean(axis=1)
 paired={'comparison':'DA V2 calibrated minus original photometric depth MAE',
         'mean_sequence_difference_mm':float(diffs.mean()),'sequence_bootstrap_95ci':np.quantile(draws,[.025,.975]).tolist(),
         'n_sequences':6,'n_held_out_anatomies':1,'interpretation':'descriptive pilot; not a population-level clinical superiority test'}
 (OUT/'paired_diagnostic.json').write_text(json.dumps(paired,indent=2))
 fig,ax=plt.subplots(figsize=(9,4.6),layout='constrained')
 for i,row in enumerate(test):
  lo,hi=row['sequence_bootstrap_mae_95ci'];ax.barh(i,row['mae_mm'],color='#476B8A')
  ax.plot([lo,hi],[i,i],color='black',lw=1.4);ax.text(hi+.15,i,f"{row['mae_mm']:.2f}",va='center',fontsize=10)
 ax.set_yticks(range(len(test)),[LABELS[r['method']] for r in test]);ax.invert_yaxis()
 ax.set_xlabel('Mean sequence axial-depth MAE (mm)');ax.set_xlim(0,20)
 ax.set_title('C3VDv2 physical-phantom pilot: 72 test frames / 6 sequences\nOne held-out anatomy; descriptive sequence-bootstrap intervals',fontsize=11)
 ax.spines[['top','right']].set_visible(False)
 fig.savefig(OUT/'external_depth_mae.pdf');fig.savefig(OUT/'external_depth_mae.png',dpi=160);plt.close(fig)
 # Predefined examples: first selected test frame in the first and last test videos.
 data=ROOT/'datasets/external/C3VDv2_native_subset';manifest=json.loads((data/'download_manifest.json').read_text())
 frames=[next(r for r in manifest['frames'] if r['scene']==scene) for scene in (scenes[0],scenes[-1])]
 fig,axes=plt.subplots(2,4,figsize=(12,6),layout='constrained')
 fit=s['calibration']
 for i,row in enumerate(frames):
  rgb=cv2.cvtColor(cv2.imread(str(data/row['color_path'])),cv2.COLOR_BGR2RGB)
  raw=cv2.imread(str(data/row['depth_path']),-1);gt=raw.astype(float)*100/65535;valid=(raw>0)&(raw<65535)
  pred=np.load(OUT/'raw_predictions'/f"{row['scene']}_{row['frame_id']}.npz")
  den=fit['depth_anything_inverse_affine_a']*pred['relative_depth_anything']+fit['depth_anything_inverse_affine_b']
  da=np.clip(np.divide(1.,den,out=np.full_like(den,100.),where=den>0),1,100)
  axes[i,0].imshow(rgb);axes[i,0].set_title(row['scene']+' / '+row['frame_id'],fontsize=9)
  for j,(name,d) in enumerate([('Registered reference Z',gt),('Original photometric Z',pred['photometric']),('Calibrated DA V2 Z',da)],1):
   artist=axes[i,j].imshow(np.where(valid,d,np.nan),vmin=0,vmax=100,cmap='viridis');axes[i,j].set_title(name,fontsize=10)
  for ax in axes[i]:ax.axis('off')
 fig.colorbar(artist,ax=axes[:,1:].ravel().tolist(),label='Axial depth (mm)',shrink=.75)
 fig.savefig(OUT/'external_depth_examples.pdf');fig.savefig(OUT/'external_depth_examples.png',dpi=160);plt.close(fig)
 lines=['| 方法 | 深度MAE / mm | 序列bootstrap 95%区间 | AbsRel | δ<1.25 |','|---|---:|---|---:|---:|']
 english=['| Method | Depth MAE (mm) | Sequence-bootstrap 95% interval | AbsRel | Delta < 1.25 |','|---|---:|---|---:|---:|']
 for r in test:
  lo,hi=r['sequence_bootstrap_mae_95ci'];row=f"| {LABELS[r['method']]} | {r['mae_mm']:.3f} | [{lo:.3f}, {hi:.3f}] | {r['abs_rel']:.3f} | {100*r['delta1_fraction']:.1f}% |"
  lines.append(row);english.append(row)
 table='\n'.join(lines);en_table='\n'.join(english)
 gt=['| Surface | Added outlier fraction | PCA q | MAD-PCA q | SO(3) no filter | SO(3) MAD |','|---|---:|---:|---:|---:|---:|']
 for shape in ['disc','dome','saddle']:
  for noise in [0.,.05,.10]:
   values=[next(r['quantile_mae_mm'] for r in geo['summary'] if r['shape']==shape and r['outlier_fraction']==noise and r['method']==m) for m in ['pca_q','mad_pca_q','so3_never','so3_always']]
   gt.append('| '+shape+f' | {noise:.0%} | '+' | '.join(f'{v:.3f}' for v in values)+' |')
 geo_table='\n'.join(gt)
 report=f'''# 外部数据检索与实测验证报告

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
- 神经网络含 {s['model_parameter_count']:,} 个参数，无训练、无微调；不是参数量/训练预算匹配的消融，只是外部方法比较。
- 光度比例和网络逆深度仿射系数只在校准集拟合；测试帧不做真值对齐或median scaling。所有预测使用预先固定的[1,100]mm范围，保留截断比例。
- 汇总以每帧指标平均到视频，再对视频等权平均。RMSE列在JSON中也是帧均值，而非所有像素合并后的RMSE。置信区间仅对6个视频重采样，不能视为患者总体区间。
- **独立测试解剖模型只有1个**；测试均为v3含污物场景，解剖与拍摄条件变化没有单独隔离。结论仅为外部物理模体pilot。

## 外部深度结果

{table}

学习型基线与原光度方法的逐视频MAE差平均为 {paired['mean_sequence_difference_mm']:.3f} mm，描述性区间为 [{paired['sequence_bootstrap_95ci'][0]:.3f}, {paired['sequence_bootstrap_95ci'][1]:.3f}] mm。不据此宣称临床优势或普遍优胜。

原光度方法的校准比例为 {fit['photometric_scale']:.4f}，在新模型/污物条件下误差增加。它是需要保留的泛化失败结果，不在看到测试结果后重新选尺度。

**这里的11–16mm是深度误差，不是病灶主径误差，不能与此前3.006mm主径MAE直接比较。**

## 几何一致与无混杂的补充基准

新增圆盘、穹顶和鞍面，完全交叉4种尺寸、3种离群点水平和5个种子，共180案例、720条方法结果。先在毫米坐标生成表面，再施加已知旋转和平移。完整几何主径由模型支持点保证；1–99%参考范围在污染前的已知本征轴上计算，和完整范围分开保存。它消除了该补充试验中的形态—尺寸混杂，**没有声称修复或替换旧RGB生成器**。

{geo_table}

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
'''
 (ROOT/'docs/EXTERNAL_DATASET_VALIDATION_REPORT.md').write_text(report)
 supplement=f'''## External reference validation and reference-consistent geometry

### Dataset acquisition and exclusions

C3VDv2 provides real colonoscope images of silicone phantoms paired with registered 3D-model reference depth [1,2]. We acquired a fixed 96-frame native-format subset through the public BeileiCui/C3VDv2 mirror. The original Dataverse was inaccessible from this environment; revision and file hashes are preserved, but bytewise equivalence to primary repository files is unverified. A separate 8-bit, half-resolution mirror was rejected because its metric conversion was undocumented. A retrieved PolypSense3D archive contained only virtual data and was not used as physical lesion-size evidence [4].

### Frozen depth protocol and external comparator

Twelve c1 ascending frames were used for calibration and twelve additional c1 frames for validation. Six c2 sequences supplied 72 test frames, with twelve uniformly spaced frames per video. The c2 anatomy was excluded from calibration. References were decoded as uint16 × 100/65535 mm along camera Z; zero and saturated values were excluded. Predictions were resized to native 1350×1080 dimensions without resampling reference depth. No unverified pinhole intrinsics were used.

Comparators were a 30-mm constant, the calibration-set median, original photometric depth, calibration-scaled photometric depth, and pretrained Depth Anything V2 Small [3]. The last model contains {s['model_parameter_count']:,} parameters and was evaluated without training or fine-tuning, using raw network output rather than display-normalized depth. A photometric scale and an affine mapping from network output to inverse metric depth were fitted only on calibration pixels. No test-frame reference alignment was permitted. Predictions were uniformly clipped to a prespecified 1–100 mm interval, with clipping rates retained. This compares different resource histories; it is not a parameter-matched ablation.

### Results and limits

{en_table}

MAE is averaged equally across six sequence means. Intervals are descriptive bootstrap intervals over sequences, not patients. Calibrating photometric scale on c1 worsened c2 performance, showing a transfer failure that was retained without post-test adjustment. The mean learned-minus-original-photometric MAE difference was {paired['mean_sequence_difference_mm']:.3f} mm with a descriptive interval of [{paired['sequence_bootstrap_95ci'][0]:.3f}, {paired['sequence_bootstrap_95ci'][1]:.3f}] mm. No general superiority claim follows. These are axial-depth errors and must not be compared numerically with the earlier lesion-diameter errors.

![External depth errors](../outputs/c3vdv2_external_validation/external_depth_mae.png)

The six test sequences come from one held-out anatomy and all use debris conditions. Anatomy and appearance shifts are therefore not isolated. Reference depth derives from 2D–3D registration and carries registration uncertainty; it is not direct in-vivo depth metrology. The subset supplies no independently annotated lesion diameter or observer-reliability data. External testing establishes a limited physical-phantom depth pilot, not clinical size accuracy.

### Reference-consistent analytic geometry

A separate fully crossed experiment used three surface types (disc, dome, saddle), four major extents (4, 8, 16, 24 mm), three added-outlier fractions (0, 5, 10%) and five seeds: 180 cases. Millimetre-valued surfaces were transformed by known rotations and translations. Full extent and clean intrinsic-axis 1st–99th-percentile extent were stored as distinct references before contamination. Both matched pairs shared filtering and reported quantiles. This removes the old renderer's confounding and nominal/reference mismatch within this new component experiment, while leaving the original renderer and its limitations unchanged.

{geo_table}

Values are MAE against the clean intrinsic-axis quantile reference, in mm; they are not clinical caliper errors. Full-extent errors are retained separately. MAD-PCA had lower mean error than identically filtered SO(3) in all nine shape-by-contamination summaries; an additional optimizer benefit was not observed in these surface families. The experiment uses analytic point clouds rather than RGB images and does not validate segmentation or depth recovery.

### Sources

[1] Bobrow et al. Colonoscopy 3D video dataset with paired depth from 2D–3D registration. Medical Image Analysis, 2023. https://doi.org/10.1016/j.media.2023.102956

[2] Golhar et al. C3VDv2-Colonoscopy 3D video dataset with enhanced realism. Scientific Data, 2026. https://doi.org/10.1038/s41597-026-07471-1 ; dataset: https://doi.org/10.7281/T1/JC64MK

[3] Yang et al. Depth Anything V2, 2024. https://arxiv.org/abs/2406.09414 ; official model: https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf

[4] PolypSense3D dataset. https://doi.org/10.7910/DVN/LKDIEK ; author repository: https://github.com/HNUicda/PolypSense3D
'''
 (ROOT/'manuscript/EXTERNAL_VALIDATION_SUPPLEMENT.md').write_text(supplement)
 # Preserve the previous draft and make additions idempotent with markers.
 backup=ROOT/'outputs/external_dataset_audit/manuscript_before_external';backup.mkdir(parents=True,exist_ok=True)
 for name in ['main_cbm.tex','main_cbm.pdf','MANUSCRIPT_CBM.md']:
  original=ROOT/'manuscript'/name
  if original.exists() and not (backup/name).exists():shutil.copy2(original,backup/name)
 md=ROOT/'manuscript/MANUSCRIPT_CBM.md';text=md.read_text()
 text=re.sub(r'\n<!-- EXTERNAL_BEGIN -->.*?<!-- EXTERNAL_END -->\n','\n',text,flags=re.S)
 marker='## 4. Discussion and limitations'
 text=text.replace(marker,'<!-- EXTERNAL_BEGIN -->\n'+supplement+'\n<!-- EXTERNAL_END -->\n\n'+marker)
 # Existing abstract described only the procedural experiment; clearly append the pilot.
 addition=f' An external physical-phantom depth pilot on 72 test frames from one held-out C3VDv2 anatomy yielded axial-depth MAE of {by["photometric_raw"]["mae_mm"]:.2f} mm for original photometric depth and {by["depth_anything_v2_calibration_inverse_affine"]["mae_mm"]:.2f} mm for a calibration-fitted pretrained baseline; these are not lesion-diameter errors.'
 if addition not in text:text=text.replace('## 1. Motivation and contribution',addition+'\n\n## 1. Motivation and contribution',1)
 md.write_text(text)
 tex=ROOT/'manuscript/main_cbm.tex';body=tex.read_text()
 body=re.sub(r'\n% EXTERNAL_BEGIN\n.*?% EXTERNAL_END\n','\n',body,flags=re.S)
 def esc(t):
  for x,y in [('&','\\&'),('%','\\%'),('_','\\_'),('#','\\#')]:t=t.replace(x,y)
  return t.replace('×',r'$\times$').replace('–','--').replace('−','$-$').replace('²','$^2$')
 block='\n% EXTERNAL_BEGIN\n';in_table=False
 for line in supplement.splitlines():
  if line.startswith('|'):
   if not in_table:
    columns=len(line.strip('|').split('|'));block+='\\begin{table}[ht]\\centering\\scriptsize\\resizebox{\\textwidth}{!}{\\begin{tabular}{'+'l'*columns+'}\\toprule\n';in_table=True
   if set(line.replace('|','').replace(':','').replace(' ',''))=={'-'}:continue
   block+=' & '.join(esc(x.strip()) for x in line.strip('|').split('|'))+' \\\\\n';continue
  if in_table:block+='\\bottomrule\\end{tabular}}\\end{table}\n';in_table=False
  if line.startswith('## '):block+='\\section{'+esc(line[3:])+'}\n'
  elif line.startswith('### '):block+='\\subsection{'+esc(line[4:])+'}\n'
  elif line.startswith('!['):block+='\\begin{figure}[ht]\\centering\\includegraphics[width=\\textwidth]{../outputs/c3vdv2_external_validation/external_depth_mae.pdf}\\caption{External physical-phantom axial-depth pilot. Six sequences share one held-out anatomy.}\\end{figure}\n'
  else:
   # Keep URLs breakable in PDF without escaping their punctuation.
   pieces=re.split(r'(https?://[^\s;]+)',line)
   block+=''.join('\\url{'+x+'}' if x.startswith('http') else esc(x) for x in pieces)+'\n'
 block+='\n% EXTERNAL_END\n'
 body=body.replace('\\section{Discussion and limitations}',block+'\\section{Discussion and limitations}',1)
 if esc(addition) not in body:body=body.replace('\\end{abstract}',esc(addition)+'\n\\end{abstract}',1)
 tex.write_text(body)
 products=[ROOT/'docs/EXTERNAL_DATASET_VALIDATION_REPORT.md',ROOT/'manuscript/EXTERNAL_VALIDATION_SUPPLEMENT.md',md,tex,Path(__file__).resolve(),OUT/'paired_diagnostic.json',*OUT.glob('external_depth_*.png'),*OUT.glob('external_depth_*.pdf')]
 (ROOT/'outputs/external_dataset_audit/report_manifest.json').write_text(json.dumps({str(p.relative_to(ROOT)):sha(p) for p in products},indent=2))
 print(table);print('Paired diagnostic:',paired);print('Created external report, figures, manuscript supplement and updated main draft.')
if __name__=='__main__':main()
