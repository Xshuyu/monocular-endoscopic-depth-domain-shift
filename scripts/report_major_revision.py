"""Generate audited major-revision tables, figures and both manuscripts."""
from pathlib import Path
import json,csv,re,zipfile,sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[1];O=R/'outputs/major_revision';F=O/'figures';F.mkdir(exist_ok=True)
objs=json.loads((O/'metric_per_object.json').read_text());raw=list(csv.DictReader((O/'original_transfer_diagnostics.csv').open()));effects=json.loads((O/'factorial_effects.json').read_text());checks=json.loads((O/'validation_checks.json').read_text())
cohorts=[('c3vd','C3VD (4 videos)'),('scared','SCARED (2 held-out subjects)'),('servct','SERV-CT (2 specimens)'),('Phantom','DRENDS phantom (5 organ groups)'),('ExVivo','DRENDS ex vivo (4 organ groups)')]
def match(row,c):return row['cohort']==c if c not in ['Phantom','ExVivo'] else row['cohort']=='drends' and row['object'].startswith(c)
def stats(c,regime,bound=100):
 vals=[np.mean([r['mae_mm'] for r in objs if match(r,c) and r['regime']==regime and r['seed']==seed and r['factor']=='identity' and r['bound']==bound and r['support']=='all_reference']) for seed in [11,23,47]]
 return float(np.mean(vals)),float(np.std(vals,ddof=1)),list(map(float,vals))
def orig(c,bound=100,metric='mae_mm'):
 a=[r for r in raw if match(r,c) and r['method']=='small_radial_ridge' and int(r['bound'])==bound and r['support']=='all_reference'];return float(np.mean([np.mean([float(r[metric]) for r in a if r['object']==obj]) for obj in sorted({r['object'] for r in a})]))
summary=[]
for c,label in cohorts:
 a,b,_=stats(c,'source_only');v,w,_=stats(c,'adapted');summary.append({'cohort':c,'label':label,'original_radial_ridge_mm':orig(c),'source_mean_mm':a,'source_sd_seeds_mm':b,'adapted_mean_mm':v,'adapted_sd_seeds_mm':w,'adapted_minus_source_mm':v-a})
(O/'major_revision_table.json').write_text(json.dumps(summary,indent=2))
plt.rcParams.update({'font.size':10,'svg.fonttype':'none','pdf.fonttype':42})
fig,ax=plt.subplots(1,2,figsize=(11,4.4),gridspec_kw={'width_ratios':[1.4,1]})
x=np.arange(5);w=.25
ax[0].bar(x-w,[r['original_radial_ridge_mm'] for r in summary],w,label='Frozen radial ridge',color='#777777')
ax[0].bar(x,[r['source_mean_mm'] for r in summary],w,yerr=[r['source_sd_seeds_mm'] for r in summary],capsize=3,label='Source-only decoder',color='#4477aa')
ax[0].bar(x+w,[r['adapted_mean_mm'] for r in summary],w,yerr=[r['adapted_sd_seeds_mm'] for r in summary],capsize=3,label='Adapted decoder',color='#ee7733')
ax[0].set_xticks(x,['C3VD','SCARED\nheld out','SERV-CT','DRENDS\nphantom','DRENDS\nex vivo']);ax[0].set_ylabel('MAE (mm)');ax[0].set_title('(a) Transfer and adaptation');ax[0].legend(fontsize=8)
xs=np.arange(3)
for j,(metric,label,color) in enumerate([('mae_mm','Unaligned','#777777'),('oracle_offset_mae_mm','GT offset diagnostic','#228833'),('oracle_scale_mae_mm','GT scale diagnostic','#ccbb44')]):ax[1].bar(xs+(j-1)*w,[orig(c,metric=metric) for c in ['c3vd','scared','servct']],w,label=label,color=color)
ax[1].set_xticks(xs,['C3VD','SCARED','SERV-CT']);ax[1].set_ylabel('MAE (mm)');ax[1].set_title('(b) Original-head error decomposition');ax[1].legend(fontsize=8)
for a in ax:a.spines[['top','right']].set_visible(False)
fig.tight_layout();fig.savefig(F/'major_revision_results.pdf');fig.savefig(F/'major_revision_results.svg');fig.savefig(F/'major_revision_results.png',dpi=150);plt.close(fig)
fig,ax=plt.subplots(figsize=(8,3.7));x=np.arange(3)
for j,(key,label) in enumerate([('dark_main','Gamma darkening'),('blur_main','Gaussian blur'),('interaction','Interaction')]):
 vals=[next(r for r in effects if r['cohort']==c and r['regime']=='adapted')[key] for c in ['c3vd','scared','servct']];ax.bar(x+(j-1)*.24,[v['mean'] for v in vals],.24,yerr=[v['sd_seeds'] for v in vals],capsize=3,label=label)
ax.axhline(0,color='k',lw=.7);ax.set_xticks(x,['C3VD','SCARED held out','SERV-CT']);ax.set_ylabel('Paired change in MAE (mm)');ax.legend(fontsize=8);ax.spines[['top','right']].set_visible(False);fig.tight_layout();fig.savefig(F/'factorial_effects.pdf');fig.savefig(F/'factorial_effects.svg');plt.close(fig)
tex=r'''\section{Independent adaptation and expanded external validation}
\subsection{Independent training and matched-update control}
To avoid inheriting object overlap from publicly released surgical checkpoints, we trained a separate supervised domain-adaptation baseline from the generic Depth Anything V2 Small initialization \citep{Yang2024}. The 22,056,576-parameter backbone remained frozen; the 2,728,513-parameter neck and head were optimized, with the final scalar convolution reinitialized to predict log depth in millimetres. This is a low-data supervised adaptation control, not a reproduction of EndoDAC or an unsupervised adaptation claim. Generic pretraining overlap cannot be exhaustively audited.

Both regimes use AdamW (learning rate $10^{-4}$, weight decay 0.01), 600 updates, batch size two, gradient norm clipping at one, and seeds 11, 23 and 47. The per-image loss is Smooth L1 on log axial depth ($\beta=0.1$), averaged equally over the two images. RGB is normalized with the released processor and resized to $224\times280$; depth and masks use nearest-neighbour resampling for training only. Valid reference targets are clamped to 1--300 mm before taking logarithms. The source-only regime samples two of the existing 52 C3VD development frames per update. Adaptation replaces one with a uniformly sampled measured keyframe from SCARED subjects 1, 2, 3 or 6 (20 frames). Subject 7 is unused; subjects 8 and 9 are excluded from training. The final checkpoint is used without validation or test selection. Thus total updates and batch size are matched, but source exposure, target supervision and original calibration-model complexity are not identical. The four-coefficient model and the trained decoder must not be described as equal-capacity or equal-pixel-budget methods.

Training manifests, RGB/reference hashes, all six checkpoint hashes and learning curves are retained. No SCARED-adapted pretrained neural weights are used. Existing test cohorts were previously inspected in the project, so this is a prespecified training protocol within a post-hoc revision, not a newly blinded confirmation. Seed variability measures optimization variability, not patient-level uncertainty.

\subsection{Additional organ groups and acquisition configurations}
We acquired and checksum-verified the DRENDS phantom and ex vivo sequence archives \citep{LozaGalindo2026}. The selection is the first frame of every numbered nonconsecutive sequence: 15 phantom and 12 ex vivo observations, comprising five and four organ/setup groups with three illumination conditions per group. This adds distinct organ preparations and acquisition configurations rather than counting more frames as independent subjects. The published setups use Ikegami 720p and Panasonic 1080i endoscopic cameras with a da Vinci Si system. Serial-number independence and ex vivo donor identities are unavailable; nine organ/setup groups are not nine independent donors, and two configurations are not a verified count of two additional independent devices. Differences in anatomy, material, camera and illumination remain partly confounded. First-frame evaluation does not test temporal or deformation robustness.

Reference construction uses the released calibration and compensation transforms, workspace mask, normal/view filtering, and a 5-mm occlusion tolerance. Raw ToF XYZ values are converted from millimetres to metres for transformation, then rendered in the rectified left-camera coordinates as axial depth and converted back to millimetres. CPU raycasting uses unnormalized rays with $z=1$; a plane test verifies axial depth rather than Euclidean range. SAM2 Hiera Large uses the authors' first-frame prompts to exclude tools; this uses the image predictor, not full-video mask propagation. These documented implementation choices make this a new external validation protocol, not an exact reproduction of the published DRENDS benchmark. No model is adapted to DRENDS, and no per-sequence truth alignment is used.

The official spatial-filter parameter is named in millimetres although its function operates on metre coordinates. We retained the released settings for the primary reference and repeated processing with a 0.005-m range sigma. On common support, mean reference differences averaged 0.471 mm across 27 frames, and the maximum observed change in trained-model frame MAE was 0.687 mm. This sensitivity check does not validate ToF hardware accuracy, registration or human mask consistency; those remain reference limitations.

\subsection{Adaptation gains and remaining transfer failure}
\begin{table}[htbp]\centering\small
\caption{Primary unaligned MAE (mm), with predictions clipped to 1--100 mm and all valid reference depths retained. Trained-model values are mean $\pm$ sample SD across three seeds. Frames/conditions are averaged within groups before averaging groups. Different rows represent different grouping units, not pooled independent patients.}
\begin{tabular}{lrrr}\toprule
Cohort & Frozen radial ridge & Source only & Adapted \\\midrule
'''
labels=['C3VD, 4 videos','SCARED, 2 held-out subjects','SERV-CT, 2 specimens','DRENDS, 5 phantom groups','DRENDS, 4 ex vivo groups']
for label,r in zip(labels,summary):tex+=f"{label} & {r['original_radial_ridge_mm']:.2f} & ${r['source_mean_mm']:.2f}\\pm{r['source_sd_seeds_mm']:.2f}$ & ${r['adapted_mean_mm']:.2f}\\pm{r['adapted_sd_seeds_mm']:.2f}$ "+r'\\'+'\n'
tex+=r'''\bottomrule\end{tabular}\label{tab:independent_adaptation}\end{table}

Adaptation reduces held-out SCARED MAE from 36.72 to 13.67 mm and SERV-CT MAE from 55.37 to 22.21 mm, but increases C3VD MAE from 8.02 to 9.09 mm. The adapted model still has large DRENDS errors (93.78 mm phantom, 57.22 mm ex vivo). These results support a benefit from labeled target-domain exposure and reveal a source-domain trade-off; they do not establish general device invariance or clinically adequate metric accuracy. Allowing a 300-mm prediction ceiling gives adapted DRENDS MAEs of 93.76 and 57.11 mm, so the 100-mm ceiling is not the main explanation for these failures.

\begin{figure}[htbp]\centering
\includegraphics[width=\textwidth]{../outputs/major_revision/figures/major_revision_results.pdf}
\caption{Independent adaptation and transfer diagnostics. (a) Seed error bars describe training variability only; organ/setup groups are not independent donors. (b) Offset and scale diagnostics use test reference values and are explicitly non-deployable; they do not replace the unaligned primary results.}
\label{fig:major_revision}\end{figure}

\subsection{Controlled factors and limits of causal interpretation}
A paired $2\times2$ design applies gamma darkening ($\gamma=1$ or 1.5) and Gaussian blur ($\sigma=0$ or 1.5 native-image pixels) to identical C3VD, SCARED and SERV-CT frames, with geometry and reference masks unchanged. All six fixed models are evaluated under every condition. For errors $E_{00},E_{10},E_{01},E_{11}$, the darkening main effect is $[(E_{10}-E_{00})+(E_{11}-E_{01})]/2$, with the analogous blur effect; interaction is $E_{11}-E_{10}-E_{01}+E_{00}$. These are image-intervention effects, not isolated physical device effects. The blur is mild, especially after input resizing; absence of a large effect cannot establish general blur robustness.

For the adapted model, darkening increases MAE by approximately 0.10 mm on C3VD, 1.65 mm on held-out SCARED and 2.65 mm on SERV-CT. Those changes are appreciably smaller than the original inter-domain errors. For the original radial ridge head, increasing the output ceiling from 100 to 300 mm leaves SCARED and SERV-CT MAEs unchanged. A separate non-deployable diagnostic subtracts a reference-derived per-frame median residual: MAE then drops from 37.11 to 9.90 mm on SCARED and from 55.54 to 12.58 mm on SERV-CT. This indicates a large offset component, but does not prove whether anatomy, camera geometry, depth distribution or appearance caused it. We do not infer a physical lighting or distortion mechanism from the radial coefficients.

\begin{figure}[htbp]\centering
\includegraphics[width=.85\textwidth]{../outputs/major_revision/figures/factorial_effects.pdf}
\caption{Paired factorial effects for the independently trained adapted decoder. Error bars are sample SD over three seeds; the same frames and reference masks are used across the four conditions.}
\end{figure}
'''
abstract=r'''Metric calibration of monocular endoscopic depth can fail across acquisition domains. We examine four-coefficient radial inverse-depth calibration through matched calibration ablations, independent supervised adaptation and controlled transfer diagnostics. Frozen affine and radial-ridge heads yield 8.794 and 8.899 mm MAE on four colon-phantom videos, but exhibit large external errors. To remove overlap inherited from public surgical checkpoints, we independently train a metric decoder from a generic frozen backbone with three seeds, comparing source-only training against matched-update adaptation using 20 measured frames from four SCARED subjects. Subjects 8 and 9 remain excluded from training. Adaptation reduces their MAE from 36.72 to 13.67 mm and SERV-CT MAE from 55.37 to 22.21 mm, while C3VD MAE increases from 8.02 to 9.09 mm. New DRENDS evaluation spans five phantom and four ex vivo organ/setup groups under two camera configurations; adapted errors remain 93.78 and 57.22 mm. These groups are not verified independent donors or device serials. Paired darkening--blur interventions, prediction-range sensitivity and explicitly reference-assisted error decomposition show that output clipping alone does not explain transfer failure and that global depth offsets are substantial. Existing-cohort comparisons are post-hoc; DRENDS reference processing and its sensitivity are documented. The results support conditional adaptation gains with source-domain trade-offs, not uniformly superior radial regularization, device-invariant metric depth or clinical lesion-sizing accuracy.'''
conclusion=r'''We provide an auditable study of monocular endoscopic calibration transfer, independently trained supervised adaptation and controlled failure diagnostics. Radial calibration is an inexpensive empirical extension with dataset-dependent benefit, not a demonstrated physical correction. Object-disjoint adaptation improves held-out SCARED and external SERV-CT performance but sacrifices some source-domain accuracy and remains inadequate on the newly evaluated DRENDS configurations. Matched image interventions and reference-assisted diagnostics identify sensitivity and offset components without establishing a unique physical cause. The evidence is strengthened by three-seed controls, new organ/setup groups, explicit reference processing and open failure cases; it remains limited by small subject counts, uncertain donor/device independence, historical test exposure and absence of clinical size/observer validation. No comprehensive state-of-the-art or clinical measurement claim is made.'''
backup=O/'manuscripts_before_major_revision.zip'
if not backup.exists():
 with zipfile.ZipFile(backup,'w',zipfile.ZIP_DEFLATED) as z:
  for folder in ['manuscript_cmig','manuscript_mbec']:
   for f in (R/folder).glob('*.tex'):z.write(f,f.relative_to(R))
for folder,file in [('manuscript_cmig','main_cmig.tex'),('manuscript_mbec','main_mbec.tex')]:
 d=R/folder;(d/'major_revision.tex').write_text(tex);p=d/file;t=p.read_text();t=re.sub(r'\\begin\{abstract\}.*?\\end\{abstract\}',lambda _:r'\begin{abstract}'+'\n'+abstract+'\n'+r'\end{abstract}',t,flags=re.S)
 if r'\input{major_revision.tex}' not in t:t=t.replace(r'\section{Discussion and limitations}',r'\input{major_revision.tex}'+'\n\n'+r'\section{Discussion and limitations}')
 t=t.replace('No neural weights are trained.','For this calibration ablation no neural weights are trained; the separately reported adaptation experiment trains a metric decoder.')
 t=t.replace('The contributions are an auditable four-way calibration ablation, scene-grouped development selection, frozen evaluation on newly acquired sequences, and radius-by-depth diagnostics.','The contributions are an auditable four-way calibration ablation, scene-grouped selection, frozen transfer evaluation, independently trained object-disjoint supervised adaptation, new organ/setup validation and paired failure diagnostics.')
 t=re.sub(r'\\section\{Conclusion\}.*?(?=\\section\*\{Data and code availability\})',lambda _:r'\section{Conclusion}'+'\n'+conclusion+'\n\n',t,flags=re.S)
 t=t.replace('The contribution here is a controlled study of a small spatial calibration model and its transfer limits, not a new foundation architecture or a comprehensive comparison with current endoscopic depth systems.','The contribution is a controlled study of spatial calibration, independently trained adaptation and transfer limits; it is not a new foundation architecture or a comprehensive comparison with current endoscopic depth systems.')
 t=t.replace('The local research package includes selection manifests, hashes, frozen heads, per-frame and per-video results, joint region tables and executable scripts.','The local research package includes selection manifests, hashes, frozen heads, per-frame and per-video results, independently trained checkpoints and learning curves, DRENDS processing records, paired interventions, reference-sensitivity checks and executable scripts.')
 t=re.sub(r'\\title\{[^}]*\}',lambda _:r'\title{Monocular Endoscopic Depth under Domain Shift: Calibration, Supervised Adaptation and Controlled Failure Analysis}',t)
 p.write_text(t)
print(json.dumps(summary,indent=2))
