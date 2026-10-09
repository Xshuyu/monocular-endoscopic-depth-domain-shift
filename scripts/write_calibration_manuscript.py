"""Write an evidence-limited manuscript from the completed locked experiment."""
from pathlib import Path
import json,csv
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/publication_study';MAN=ROOT/'manuscript'
dev={r['method']:r for r in json.loads((OUT/'development_summary.json').read_text())};fin={r['method']:r for r in json.loads((OUT/'final_summary.json').read_text())}
frozen=json.loads((OUT/'frozen_heads.json').read_text())
paired=json.loads((OUT/'paired_video_diagnostics.json').read_text())['comparisons']
labels={'constant_median':'Calibration median','photometric_raw':'Photometric','small_single_scene_affine':'DA V2 Small, one-scene affine','small_affine':'DA V2 Small, multi-scene affine','small_monotone':'DA V2 Small, monotone bins','selected_calibration':'DA V2 Small, selected spline','large_affine':'DA V2 Large, multi-scene affine','endodac_affine':'EndoDAC, multi-scene affine'}
order=list(labels)
rows=[]
for m in order:
 d=dev['nested_selected' if m=='selected_calibration' else m];f=fin[m]
 rows.append(f"{labels[m]} & {d['mae_mm']:.3f} & {f['mae_mm']:.3f} & {f['rmse_mm']:.3f} & {f['abs_rel']:.3f} & {100*f['delta1_fraction']:.1f} " + chr(92)*2)
table='\n'.join(rows)
with (OUT/'final_per_video.csv').open() as f:video=list(csv.DictReader(f))
seqrows=[]
for s in ['cecum_t1_a','desc_t4_a','sigmoid_t1_a']:
 values=[float(next(r['mae_mm'] for r in video if r['scene']==s and r['method']==m)) for m in ['small_affine','small_monotone','selected_calibration','large_affine','endodac_affine']]
 seqrows.append(s.replace('_',r'\_')+' & '+' & '.join(f'{v:.3f}' for v in values)+r' \\')
seqtable='\n'.join(seqrows)
text=r'''\documentclass[10pt]{article}
\usepackage[margin=0.85in]{geometry}
\usepackage[T1]{fontenc}
\usepackage{lmodern,amsmath,amssymb,booktabs,graphicx,microtype}
\usepackage[hidelinks]{hyperref}
\usepackage{enumitem}
\graphicspath{{../outputs/publication_study/figures/}}
\title{Calibration Objectives and Domain Shift in Monocular Endoscopic Depth:\\A Controlled Phantom Study}
\author{}
\date{Research draft --- 27 September 2026}
\begin{document}
\maketitle
\begin{abstract}
Metric interpretation of monocular endoscopic depth depends on calibration, yet a lower development error need not imply reliable transfer. We evaluate calibration objectives, scene coverage and frozen depth backbones in a controlled phantom study. Six C3VDv2 scene groups (52 frames) support nested leave-scene-out model selection. Calibration uses a fixed budget of 16,384 reference pixels, while final metrics retain native-resolution reference maps and prohibit test-ground-truth scale alignment. Three previously unscored official C3VD videos (36 frames) form the locked final transfer set. A validation-selected monotone spline yields 8.668\,mm mean absolute error (MAE), compared with 9.090\,mm for matched multi-scene affine calibration, 8.470\,mm for a simpler binned monotone mapping and 8.359\,mm for a calibrated larger backbone. The selected spline improves on affine calibration in only one of three final videos. Its paired video-level difference is $-0.422$\,mm, with a descriptive bootstrap interval spanning zero. EndoDAC performs best among the compared systems in development MAE but does not retain this ranking on final transfer. Equal-budget calibration coverage also has different effects in development and final evaluation. These results motivate reporting calibration support, scene-level effects and metric trade-offs alongside aggregate errors. They do not establish state-of-the-art performance or clinical lesion-sizing accuracy.
\end{abstract}

\section{Introduction}
Monocular depth backbones recover geometric cues that may support endoscopic reconstruction and measurement, but their outputs do not automatically provide transferable millimetre scale. Phantom datasets with registered reference depth make quantitative evaluation possible \cite{c3vd,c3vdv2}. General-purpose models such as Depth Anything V2 \cite{dav2}, as well as endoscopy-adapted models such as EndoDAC \cite{endodac}, provide distinct starting points for this evaluation.

Three choices can complicate comparisons: fitting a calibration objective that differs from the reported metric; changing calibration scene coverage at the same time as the algorithm; and using reference information from test images to resolve scale ambiguity. Scale-aligned relative-depth evaluation is a legitimate task, but it differs from producing metric depth without test reference access. We study the latter setting.

Our contribution is an auditable experimental analysis, not a new foundation model. We combine (i) scene-grouped nested calibration selection, (ii) an equal-pixel-budget coverage comparison, (iii) a specialist endoscopic backbone under the same downstream metric-depth rules, and (iv) locked cross-dataset evaluation with scene-level and depth-stratified reporting. The constrained spline is a conventional mathematical construction used as a candidate calibrator, not a claimed novel optimization principle.

Related diagnostic work on monocular polyp sizing identifies limitations of metric scale and segmentation robustness \cite{xiong}. Our narrower endpoint is axial depth in physical phantoms rather than clinical size classification or lesion diameter. We do not claim priority for identifying scale ambiguity or for showing that segmentation matters.

\section{Data and evaluation boundary}
\subsection{Development data}
Development uses 52 native-resolution frames from six scenes of C3VDv2 anatomy c1: ascending (12 frames), cecum, rectum, sigmoid1, transverse1 and transverse2 (eight frames each). All frames within a scene remain together. The ascending sequence is a different acquisition condition from the debris-containing sequences used for other regions, so anatomical location and acquisition condition are not independently manipulated. All six scenes belong to one anatomy; they are not six patients.

These development images had been used in preceding exploratory experiments. Nested validation limits within-run tuning leakage, but it cannot erase prior design choices informed by these data. Development findings are therefore exploratory. C3VDv2 files come from a fixed public mirror revision; local hashes and native encoding are verified, while bytewise equivalence to the unavailable primary repository remains unverified.

\subsection{Locked final transfer data}
Before the present development run, we recorded three previously unscored official C3VD sequences: \texttt{cecum\_t1\_a}, \texttt{desc\_t4\_a} and \texttt{sigmoid\_t1\_a}. Twelve paired frames per video were selected uniformly by numeric frame identifier. The cecum archive uses unpadded RGB identifiers and padded depth identifiers; a documented file-pairing amendment matched numeric identifiers and excluded identifiers lacking a reference file, without consulting pixel values or model errors. Selected archive members were fetched through public HTTP byte ranges and checked using ZIP CRC and saved SHA256 digests. The complete archives were not downloaded or hashed.

All calibration choices were frozen before final prediction and reference scoring. Exact RGB hashes were checked against the previously used image collections, with no duplicate found. This detects byte duplicates, not near duplicates or possible overlap with foundation-model pretraining data. The final set contains three videos from physical colon models, not three independent patients. The locally recorded protocol is not a formal prospective clinical registration.

\subsection{Reference and metrics}
The reference is registered axial camera depth $Z$, decoded as $100d/65535$\,mm from unsigned 16-bit values. Values 0 and 65535 are excluded. The reference is not resized. Prediction maps are resized to reference dimensions, and all metric predictions are constrained to $[1,100]$\,mm. No per-test-image median scaling or reference-based affine fitting is performed.

The primary endpoint is per-frame MAE averaged within each video and then equally across videos. RMSE is the average of per-frame root-mean-square errors, not a pooled-pixel RMSE. Secondary endpoints are absolute relative error and $\delta_1=\Pr(\max(\hat z/z,z/\hat z)<1.25)$. Registered phantom reference depth has its own alignment uncertainty and is not a direct clinical lesion measurement.

\section{Models and calibration}
\subsection{Frozen backbones and fair comparisons}
We compare Depth Anything V2 Small (24,785,089 parameters), Large (335,315,649 parameters), and EndoDAC (99,091,748 parameters). No neural network is trained or fine-tuned in this study. The public EndoDAC depth checkpoint is loaded strictly after removing only the non-model metadata \texttt{height}, \texttt{width} and \texttt{use\_stereo}. Its original DV-LoRA architecture and author preprocessing are retained. RGB preprocessing was independently checked against the author's torchvision procedure. The supplied checkpoint is the author's endoscopy-trained model; it is not retrained for this experiment.

EndoDAC uses RGB values in $[0,1]$, a $256\times320$ input resize and the model's internal $224\times280$ resize. Its predicted disparity is converted using the author's fixed inverse-depth range before downstream calibration. The author's optional test-reference median scaling is disabled in our protocol. Thus, our results are a calibrated transfer comparison, not a numerical replication of the original EndoDAC benchmark. The backbones differ in architecture, input resolution and pretraining history; this is not a compute-matched architecture ablation.

The simple controls are a calibration-depth median and the project's fixed photometric estimator. The latter uses intensity inversion and normalization and is a heuristic, not an independently validated physical illumination model. Every fitted head uses 16,384 development reference pixels. Scene allocations differ by at most one pixel, and within-scene selection is deterministic and approximately frame balanced. The one-scene coverage control uses the lexicographically first available training scene with the same total budget; it is not chosen using test performance.

\subsection{Calibration objectives}
The inverse-affine baseline fits $1/z\approx ar+b$ by scene-weighted least squares, where $r$ is the backbone's relative inverse-depth prediction. Nonpositive calibrated inverse depth is mapped to the common maximum prediction of 100\,mm. A simple alternative forms 64 ordered prediction bins, summarizes each by medians, applies weighted monotone regression, and linearly interpolates the resulting mapping.

For the spline candidate, twelve prediction quantiles define a piecewise-linear basis $B$. Node values $v$ are optimized subject to
\begin{equation}
1\le v_j\le100,\qquad v_{j+1}\le v_j.
\end{equation}
The fitting objective is
\begin{equation}
\frac{1}{S}\sum_{s=1}^{S}\frac{1}{n_s}\sum_{i\in s}
\left[\sqrt{((Bv)_i-z_i)^2+\epsilon^2}-\epsilon\right]
+\lambda\,\operatorname{mean}((D_2v)^2),\quad \epsilon=0.5\ \mathrm{mm}.
\end{equation}
$D_2$ is a second difference of node values. On nonuniform quantile knots it is a discrete roughness penalty, not an exact physical-curvature integral. The problem is solved with constrained numerical optimization. Out-of-range input predictions use endpoint values.

The fixed candidate grid compares raw and per-frame 5--95\% normalized inputs; $\lambda\in\{0,0.02,0.2\}$; and either mean scene loss or an entropic scene-risk aggregation with temperature 2\,mm. Together with inverse-affine and binned monotone calibration, this produces 14 candidate heads. All choices are limited to development data.

\subsection{Nested selection and frozen final calibration}
Each of six outer folds withholds one complete scene. Within its five training scenes, five inner leave-scene-out folds select a head by equal-scene sampled MAE. Inner scoring uses 8,192 valid reference pixels per frame; outer scoring uses all valid native-resolution pixels. Fixed baseline heads are fitted on the same outer training data and pixel budget.

After outer evaluation, six-fold development-only scores select the final head, which is fitted on all six development scenes with the same 16,384-pixel budget. The selected final configuration uses raw inputs, mean scene aggregation and $\lambda=0.02$. The final three-video set is scored once, without revising the head. A separate, explicitly exploratory development-only analysis evaluates all five eligible single-scene anchors in each outer fold; it is not used to alter the frozen final model.

\subsection{Failure analysis and uncertainty}
Predefined diagnostic depth bands are $(0,10]$, $(10,30]$ and $(30,100)$\,mm. Bright-neutral pixels satisfy minimum RGB channel $\ge220$ and maximum channel $\ge245$; dark pixels have maximum channel $<20$. These are image proxies, not annotated reflections or clinical artifact labels. Region errors are computed within frames, then averaged by video; support counts are retained. Input predictions outside calibration support and predictions at clipping bounds are also recorded.

Paired differences use video means, with 20,000 bootstrap resamples of videos. With only six development groups and three final videos, the intervals are descriptive conditional summaries, not patient-population confidence statements or evidence of statistical significance. They are not simultaneous intervals for multiple comparisons.

\section{Results}
\subsection{Aggregate performance and ranking changes}
Table~\ref{tab:main} reports the complete comparison. The selected calibration row uses outer-fold-specific inner selections for development and the single frozen head for final transfer; these are related procedures, not identical fitted parameters. EndoDAC has the lowest development MAE among these comparisons, while Large has the lowest final MAE. This ranking change cautions against selecting a generally superior backbone from a single calibration domain.

\begin{table}[htbp]\centering\small
\caption{Depth results under the common no-test-alignment protocol. All MAE and RMSE values are in mm. Final $\delta_1$ is a percentage.}\label{tab:main}
\begin{tabular}{lrrrrr}\toprule
Method & Dev MAE & Final MAE & RMSE & AbsRel & $\delta_1$\\\midrule
__MAIN_TABLE__
\bottomrule\end{tabular}
\end{table}

\begin{figure}[htbp]\centering
\includegraphics[width=\linewidth]{comparison.pdf}
\caption{Development and final transfer comparison. Development selected calibration uses nested outer-fold predictions; final selected calibration is frozen before reference scoring. Differences between panels also reflect dataset shift.}
\end{figure}

\subsection{Scene-level effects and calibration coverage}
The selected spline reduces final aggregate MAE from 9.090 to 8.668\,mm relative to matched Small inverse-affine calibration, a 4.65\% reduction. However, Table~\ref{tab:scene} shows improvement only in the cecum video and deterioration in the other two. The paired mean difference is $-0.422$\,mm, with descriptive 95\% bootstrap interval $[-4.228,1.940]$\,mm. The simpler binned monotone mapping reaches 8.470\,mm, so these results do not establish an advantage for the more elaborate spline.

At the same pixel budget, multi-scene Small affine calibration gives 9.090\,mm final MAE versus 11.317\,mm from the one-scene control. The paired difference is $-2.227$\,mm with descriptive interval $[-6.187,3.633]$\,mm. In development the direction reverses: 12.361 versus 11.132\,mm. This is evidence of context dependence, not a general law that adding scenes improves calibration. The exploratory anchor sensitivity confirms that the identity of the single calibration scene matters; it does not retroactively change the primary control.

\begin{table}[htbp]\centering\small
\caption{Final per-video MAE. All configurations were fixed before these results were computed.}\label{tab:scene}
\begin{tabular}{lrrrrr}\toprule
Video & Small affine & Monotone & Spline & Large affine & EndoDAC\\\midrule
__SCENE_TABLE__
\bottomrule\end{tabular}
\end{table}

\begin{figure}[htbp]\centering
\includegraphics[width=.95\linewidth]{final_sequence_results.pdf}
\caption{The aggregate spline improvement is concentrated in one final sequence. Reporting only the overall mean would hide regressions elsewhere.}
\end{figure}

\subsection{Metric and depth-range trade-offs}
On the final set, selected calibration improves MAE and mean frame RMSE relative to Small affine calibration, but increases AbsRel from 0.420 to 0.469 and slightly lowers $\delta_1$. Optimizing millimetre residuals therefore does not establish improvement on all error measures.

Development out-of-fold diagnostics show the same type of trade-off. For depths above 30\,mm, selected calibration has region MAE 14.732\,mm versus 19.660\,mm for Small affine calibration. For depths at most 10\,mm, it worsens from 10.702 to 13.358\,mm. EndoDAC reaches 5.797\,mm in the near band. These observations are compatible with heterogeneous range-dependent errors; they do not identify a unique causal failure mechanism.

Dark-proxy region errors are 21.298\,mm for selected calibration and 16.150\,mm for Small affine calibration, but only 20,365 valid pixels across 52 development frames support this proxy. Small and uneven regional support limits interpretation. Bright-neutral proxy errors do not support a claim that the spline specifically solves specular reflection.

\begin{figure}[htbp]\centering
\includegraphics[width=.9\linewidth]{development_depth_strata.pdf}
\caption{Predefined depth-band diagnostics from development out-of-fold predictions. Region metrics are frame- and video-averaged, not pooled-pixel errors.}
\end{figure}

\section{Discussion}
The data support three restricted conclusions. First, calibration selection can yield a favorable aggregate error while harming individual scenes and depth ranges. Second, a larger calibration scene set at a fixed sample budget can improve one transfer setting and worsen another. Third, a specialist backbone that ranks first in development need not rank first after transfer under a different metric calibration task.

These observations favor explicit reporting of calibration resources, scene-level differences and protocol boundaries. They do not show that specialist models are generally inferior, that monotone calibration universally transfers, or that scene coverage is unhelpful. The small study cannot disentangle anatomy, texture, debris, camera behavior and training-domain effects. Our adapted EndoDAC evaluation also changes its scale-recovery task relative to its published benchmark, which is essential context for interpreting its ranking.

\subsection{Limitations and clinical boundary}
The development set contains one anatomical model, and final videos are physical phantom sequences rather than patients. The study uses 88 sampled frames, not the full public datasets. Repeated development use can influence model-family design despite nested selection, and future confirmation requires additional sequestered anatomical models and acquisition conditions. Reference registration error and unverified pretraining overlap remain limitations. The equal-budget one-scene control uses one predefined anchor rule; exploratory anchor variation is not a substitute for a larger prospective coverage study.

No clinical polyp-size accuracy, surface geodesic accuracy, observer ICC, uncertainty calibration, or autonomous clinical safety is established. Although 42 publicly available Polyp-Size videos and millimetre labels were acquired and audited separately \cite{polypsize}, the package lacks the masks and geometric calibration needed by the current 3D measurement pipeline. Its post-resection caliper endpoint is different from an in-vivo geometric quantile extent. Those videos are excluded from all depth claims and remain unused for model fitting or selection.

\section{Reproducibility and scope of contribution}
The release artifacts include the locally locked protocol, pairing amendment, data and model hashes, strict checkpoint loading audit, fold-specific calibration heads, selected final head, per-frame and per-video metrics, region support counts, descriptive paired differences and source snapshots. Software checks cover fitting-budget equality, held-group exclusion, calibration properties and reference units. We retain negative comparisons and do not replace the original experiments with only favorable results.

This manuscript is a controlled pilot study and an evaluation workflow. It does not introduce a demonstrably state-of-the-art algorithm. A submission should retain this scope and be assessed for whether the controlled findings add sufficient value beyond existing diagnostic studies, rather than relabeling calibration components as a new foundation model.

\begin{thebibliography}{9}
\bibitem{c3vd} T. L. Bobrow et al. Colonoscopy 3D video dataset with paired depth from 2D--3D registration. \emph{Medical Image Analysis}, 2023. \href{https://doi.org/10.1016/j.media.2023.102956}{doi:10.1016/j.media.2023.102956}.
\bibitem{c3vdv2} M. Golhar et al. C3VDv2---Colonoscopy 3D video dataset with enhanced realism. \emph{Scientific Data}, 2026. \href{https://doi.org/10.1038/s41597-026-07471-1}{doi:10.1038/s41597-026-07471-1}. Dataset: \href{https://doi.org/10.7281/T1/JC64MK}{doi:10.7281/T1/JC64MK}.
\bibitem{dav2} L. Yang et al. Depth Anything V2. 2024. \href{https://arxiv.org/abs/2406.09414}{arXiv:2406.09414}.
\bibitem{endodac} B. Cui et al. EndoDAC: Efficient Adapting Foundation Model for Self-Supervised Depth Estimation from Any Endoscopic Camera. \emph{MICCAI}, 2024, pp. 208--218. \href{https://doi.org/10.1007/978-3-031-72089-5_20}{doi:10.1007/978-3-031-72089-5\_20}.
\bibitem{xiong} X. Xiong et al. Understanding Model Behavior in Monocular Polyp Sizing. 2026. \href{https://arxiv.org/abs/2605.20461}{arXiv:2605.20461}.
\bibitem{polypsize} Y. Song et al. Polyp-Size: A Precise Endoscopic Dataset for AI-Driven Polyp Sizing. \emph{Scientific Data}, 2025. \href{https://doi.org/10.1038/s41597-025-05251-x}{doi:10.1038/s41597-025-05251-x}.
\end{thebibliography}
\end{document}
'''.replace('__MAIN_TABLE__',table).replace('__SCENE_TABLE__',seqtable)
# Table row terminators must have two literal TeX backslashes.
text=text.replace(' \\\n', ' \\\\\n') if False else text
(MAN/'calibration_generalization.tex').write_text(text)
print('Wrote evidence-limited manuscript')
