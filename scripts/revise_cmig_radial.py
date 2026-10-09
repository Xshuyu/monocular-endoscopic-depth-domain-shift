"""Rebuild the CMIG draft from auditable calibration evidence, preserving authors."""
import json,csv,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/radial_confirmatory';MAN=ROOT/'manuscript_cmig'
old=(OUT/'manuscript_cmig_before/main_cmig.tex').read_text()
authors=old[old.index('\\author['):old.index('\\begin{abstract}')]
def load(n):return json.loads((OUT/n).read_text())
dev={r['method']:r for r in load('development_summary.json')};test={r['method']:r for r in load('test_summary.json')};frozen=load('frozen_heads.json');paired=load('paired_diagnostics.json')['comparisons'];comparison=next(r for r in paired if r['phase']=='test' and r['a']=='radial_ridge' and r['b']=='affine_ls');hist=next(r for r in paired if r['phase']=='test' and r['a']=='historical_radial')
methods=['affine_ls','affine_ridge','radial_ls','radial_ridge'];labels={'affine_ls':'Global affine, LS','affine_ridge':'Global affine, ridge','radial_ls':'Radial affine, LS','radial_ridge':'Radial affine, ridge','historical_affine':'Historical fixed affine','historical_radial':'Historical fixed radial'}
row=[]
for m in methods:
 d,t=dev[m],test[m];row.append(f"{labels[m]} & {d['mae_mm']:.3f} & {t['mae_mm']:.3f} & {t['rmse_mm']:.3f} & {t['abs_rel']:.3f} & {100*t['delta1_fraction']:.1f} "+chr(92)*2)
with (OUT/'test_per_video.csv').open() as f:video=list(csv.DictReader(f))
seq=[]
for s in sorted({r['scene'] for r in video}):seq.append(s.replace('_',r'\_')+' & '+' & '.join(f"{float(next(r['mae_mm'] for r in video if r['scene']==s and r['method']==m)):.3f}" for m in methods)+r' \\')
with (OUT/'test_strata_summary.csv').open() as f:regions=list(csv.DictReader(f))
rr=[]
for rb,name in enumerate(['Center','Middle','Edge']):
 for db,dname in [(-1,'All'),(0,'0--10'),(1,'10--30'),(2,'30--100')]:
  a=next((r for r in regions if r['method']=='affine_ls' and int(r['radius_bin'])==rb and int(r['depth_bin'])==db),None);b=next((r for r in regions if r['method']=='radial_ridge' and int(r['radius_bin'])==rb and int(r['depth_bin'])==db),None)
  if a is None or b is None:
   rr.append(f'{name} & {dname} & 0 & -- & -- & -- '+chr(92)*2);continue
  rr.append(f"{name} & {dname} & {int(a['pixels']):,} & {float(a['mae_mm']):.3f} & {float(b['mae_mm']):.3f} & {float(b['mae_mm'])-float(a['mae_mm']):+.3f} "+chr(92)*2)
ci=comparison['video_bootstrap_descriptive_95ci'];delta=comparison['mean_delta_mm'];percent=100*(1-test['radial_ridge']['mae_mm']/test['affine_ls']['mae_mm'])
text=r'''\documentclass[preprint,12pt,authoryear]{elsarticle}
\usepackage{amsmath,amssymb,graphicx,booktabs,hyperref,lineno,geometry}
\geometry{a4paper,margin=1in}
\setlength{\emergencystretch}{2em}
\hypersetup{hidelinks}
\graphicspath{{../outputs/radial_confirmatory/figures/}}
\journal{Computerized Medical Imaging and Graphics}
\begin{document}
\begin{frontmatter}
\title{Radial Affine Calibration of Monocular Endoscopic Depth: Controlled Ablation and Cross-Device Transfer Evaluation}
@@AUTHORS@@
\begin{abstract}
\textbf{Background and objective:} Relative monocular depth requires calibration before millimetre interpretation. A global mapping may leave position-dependent errors, but apparent improvements on repeatedly observed data do not establish generalization or their physical cause. We examine whether radial spatial terms and ridge regularization improve frozen-backbone metric depth.
\textbf{Methods:} A four-coefficient inverse-depth mapping augments a global affine head with squared image radius and its interaction with predicted depth. We compare affine and radial models, each with and without ridge regularization, using identical calibration pixels and train-only feature standardization. Six C3VDv2 development scenes (52 frames) support nested leave-scene-out selection. Heads are frozen before acquisition and scoring of four previously unscored C3VD sequences (48 frames). A further frozen comparison uses 16 views of two SERV-CT specimens from a different endoscope system, adding EndoDAC and a larger generic backbone. Evaluation excludes test-reference scale alignment and reports group-level and radius--depth diagnostics.
\textbf{Results:} Development MAE is 12.361\,mm for global affine least squares, 12.323\,mm for affine ridge, 10.867\,mm for radial least squares and 10.800\,mm for radial ridge. On the newly acquired sequences, the corresponding errors are @@TESTVALUES@@\,mm. Radial ridge improves on affine least squares in @@WINS@@ of four videos; the paired video-mean difference is @@DELTA@@\,mm, with descriptive bootstrap interval [@@LOW@@, @@HIGH@@]\,mm. These intervals characterize this small phantom collection, not independent patients. On SERV-CT, all eight prespecified heads have MAE above 53 mm, indicating poor direct transfer to the new device/domain.
\textbf{Conclusion:} The controlled ablation separates spatial calibration from regularization and provides an auditable assessment of transfer. The results concern phantom and ex vivo axial depth; direct cross-device transfer remains inaccurate. They neither establish a causal optical-distortion correction nor validate clinical lesion diameter, resection margins or state-of-the-art performance.
\end{abstract}
\begin{keyword}
Monocular endoscopy \sep Metric depth \sep Radial calibration \sep Ridge regression \sep Phantom validation \sep Domain shift
\end{keyword}
\end{frontmatter}
\linenumbers

\section{Introduction}
Quantitative interpretation of monocular endoscopy depends on the relationship between relative image-based depth and physical geometry. Dense phantom reference datasets permit depth evaluation under real acquisition optics, while clinical lesion dimensions additionally require calibrated camera rays, lesion boundaries and a suitable physical reference \citep{Bobrow2023,Golhar2026}. These endpoints must not be conflated: lower axial-depth error does not necessarily imply lower error in a three-dimensional diameter.

Modern relative-depth backbones, including Depth Anything V2 \citep{Yang2024}, offer a useful starting point. A scalar affine calibration in inverse-depth space can resolve an empirical scale and offset within a calibration domain. It does not model position-dependent residuals. Adding squared image radius provides a low-dimensional spatial extension, but the radius may correlate with depth, anatomy or framing. An observed benefit therefore does not by itself establish correction of lens distortion or illumination falloff.

The central question is whether an inexpensive spatial calibration term improves depth beyond a matched global affine model and whether that improvement survives scene transfer. We separate three issues: the contribution of radial features, the incremental effect of ridge regularization, and dependence on scene composition. Historical fixed heads are retained only as secondary controls because their original fitting records are unavailable. The primary comparison uses a fully specified fitting procedure and identical calibration budgets.

The contributions are an auditable four-way calibration ablation, scene-grouped development selection, frozen evaluation on newly acquired sequences, and radius-by-depth diagnostics. We do not claim invention of ridge regression, a new depth backbone, or a completed clinical measurement system. The radial form is treated as a testable empirical calibration model.

\subsection{Workflow overview}
Figure~\ref{fig:architecture} summarizes the implemented workflow. Development reference depth is used for calibration and grouped selection; new-sequence reference depth enters only the evaluation branch. The optional measurement interface requires additional calibrated rays and lesion masks and has not undergone clinical size validation.

\begin{figure}[p]
\centering
\includegraphics[width=\textwidth]{figures/radial_architecture.pdf}
\caption{Code-derived architecture of the primary calibration study. (A) Scene-balanced development data support four matched calibration families, grouped selection and final head freezing. The monocular backbone remains frozen. (B) Newly acquired RGB frames pass through the frozen backbone and calibration; test reference depth is used only for scoring and diagnostic strata. (C) Dashed arrows denote an optional size-measurement interface requiring calibrated camera rays and lesion masks; this branch is analytically tested but not clinically validated. Historical fixed-coefficient controls are omitted for clarity.}
\label{fig:architecture}
\end{figure}

\section{Data and evidence boundaries}
\subsection{Development and previously observed data}
Development contains 52 frames from six C3VDv2 c1 scene groups: ascending (12 frames), cecum, rectum, sigmoid1, transverse1 and transverse2 (eight each). All frames of a scene stay in the same fold. These images had informed earlier exploratory work; nested selection does not undo that history. The groups do not represent six independent patients or six separate full-colon anatomies. Acquisition conditions also differ between the ascending and other scenes, confounding anatomy with condition.

The historical evaluation contains 72 C3VDv2 frames from six c2 videos, eight further c2 sigmoid frames, and 12 C3VD transverse frames. All were observed before the present design. We describe them as reused exploratory data. The previously completed three-video evaluation is also excluded from the new test set. C3VDv2 inputs originate from a fixed public mirror with recorded file hashes; equivalence to inaccessible primary archive bytes was not independently established.

\subsection{New sequence evaluation}
Before fitting, the protocol specifies \texttt{cecum\_t2\_a}, \texttt{cecum\_t3\_a}, \texttt{sigmoid\_t2\_a} and \texttt{trans\_t2\_a}. Twelve matched RGB/reference identifiers are selected uniformly from each official archive by numeric identifier. Selection records precede extraction; acquisition follows head freezing. Selected ZIP members undergo CRC and local SHA256 checks. Complete archives are not downloaded or hashed. No sequence is replaced based on performance.

These are previously unscored sequences in this project, not new patients or a different-camera validation. Two videos depict cecum with different textures; four videos consequently span only three anatomical segment families. Exact RGB hashes are checked against prior evaluation inputs, but this cannot exclude near-duplicate geometry, correlated trajectories or overlap with foundation-model pretraining. C3VD records a clinical colonoscope imaging high-fidelity phantoms with reference views registered to known geometry \citep{Bobrow2023}; these are not in-vivo CT-registered lesion measurements.

\section{Calibration methods}
\subsection{Frozen depth and radius}
Depth Anything V2 Small is loaded from the project's hashed local checkpoint and kept in evaluation mode. No neural weights are trained. Development reuses stored frozen-backbone predictions; new-test RGB frames undergo actual inference without reading reference depth values. The backbone's scalar output $d$ is treated as a relative inverse-depth feature. The prediction is bilinearly resized to the native reference grid; the reference is not resized.

For a grid of width $W$ and height $H$, define
\begin{equation}
r^2(u,v)=\frac{(u-W/2)^2+(v-H/2)^2}{(W/2)^2+(H/2)^2}.
\end{equation}
This uses the geometric image center, not an independently estimated optical principal point. It is not a calibrated ray-angle coordinate. The radial model is
\begin{equation}
\hat q(u,v)=(a_0+a_1r^2)d(u,v)+b_0+b_1r^2,\qquad \hat q\approx 1/Z.
\end{equation}
The affine model sets $a_1=b_1=0$. Positive inverse predictions are inverted and clipped to $[1,100]$\,mm; nonpositive values map to 100\,mm. This shared boundary rule is reported alongside its activation fraction.

\subsection{Explicit fitting objective}
The non-intercept features are $x=d$ for affine calibration and $x=(d,dr^2,r^2)$ for radial calibration. Feature means $\mu$ and standard deviations $\sigma$ are computed using training weights only. Degenerate standard deviations are replaced by one. With $\tilde x=(x-\mu)/\sigma$, we solve
\begin{equation}
\min_{\beta,c}\frac{1}{S}\sum_{s=1}^{S}\frac{1}{n_s}\sum_{i\in s}
(\tilde x_i^\top\beta+c-1/Z_i)^2+\lambda\|\beta\|_2^2.
\label{eq:fit}
\end{equation}
The intercept is unpenalized. Ridge penalizes all standardized non-intercept features, consistently in both families. Least squares uses $\lambda=0$. An augmented least-squares solve provides the coefficients; means, scales, standardized coefficients and equivalent raw four-coefficient fields are saved.

This is a new, fully specified convention. It is not the radial-only unstandardized penalty claimed without fitting records in the earlier draft. Consequently its numerical $\lambda$ values cannot be compared directly to the historical claim of $10^{-4}$.

Every fitted head uses 16,384 reference pixels in total. Pixels are allocated equally across training scenes up to integer rounding; deterministic sampling draws from a pool of 8,192 valid pixels per frame. The same training pixels are used for all four families. No test image statistics are used to fit feature normalization or coefficients.

\subsection{Scene-grouped selection and freezing}
Six outer leave-scene-out folds evaluate development behavior. In each outer fold, five inner folds select regularization independently for each ridge family. The fixed grid is $\{10^{-8},10^{-7},10^{-6},10^{-5},10^{-4},10^{-3},10^{-2},10^{-1}\}$. Selection minimizes equal-scene sampled depth MAE, while fitting minimizes inverse-depth squared error. This objective mismatch is intentional and explicit; outer evaluation uses all valid native pixels.

Final ridge choices use six-fold development-only scores, followed by fitting on all six development scenes with the unchanged total pixel budget. The selected values are $10^{-8}$ for affine ridge and $10^{-1}$ for radial ridge. Both lie on the search boundary. No search expansion or refitting follows new-test results. Source hashes, protocol hash and coefficients are recorded before acquisition of new test frames. The test scorer rejects an existing final result.

\subsection{Execution and computational scope}
The calibration head has two fitted coefficients in the affine case and four in the radial case; these counts exclude the frozen neural backbone. Calibration is a small linear least-squares problem, rather than neural-network training. For $N$ calibration samples and $p$ augmented features ($p=2$ or $4$), a dense least-squares solve has order $Np^2$ work for fixed small $p$. Application to an $H\times W$ prediction requires order $HW$ scalar operations. These are algorithmic counts, not measured latency, memory or real-time guarantees.

The executable sequence is: (1) obtain frozen development predictions and valid reference pixels; (2) construct identical scene-balanced samples for each family; (3) estimate feature normalization using the training fold only; (4) fit candidate heads and select ridge strength on held-out development scenes; (5) refit on all development scenes and freeze coefficients and provenance; and (6) apply the frozen mapping to new predictions before reading reference values for scoring. Appendix~\ref{app:implementation} provides the final coefficients and maps supporting implementations to their evidence boundaries.

\subsection{Metrics and uncertainty}
Unsigned 16-bit references are decoded as $Z=100d_{16}/65535$\,mm; codes 0 and 65535 are excluded. The primary metric is native valid-pixel MAE averaged within each frame, then video, then equally over videos. Secondary metrics are mean per-frame RMSE, AbsRel and $\delta_1=\Pr[\max(\hat Z/Z,Z/\hat Z)<1.25]$. Mean per-frame RMSE is not pooled-pixel RMSE. No test-reference median scaling or affine alignment is permitted.

Paired differences are computed per video. Fixed-seed 20,000-draw video bootstrap intervals are descriptive; no frame-wise significance claim or patient-population coverage claim is made. Grouping the two cecum videos provides an additional anatomical-segment sensitivity analysis, still with only three groups. Its equal-segment mean difference for radial ridge versus affine is +1.156 mm, illustrating the influence of repeated segment weighting.

Mechanism diagnostics use radius bands $[0,0.33)$, $[0.33,0.66)$ and $[0.66,1]$, crossed with reference depth bands $(0,10]$, $(10,30]$ and $(30,100)$\,mm. Region summaries retain support counts and exclude frames without support. Depth stratification reduces one obvious confound but does not identify a causal optical mechanism.

\section{Results}
\subsection{Controlled calibration ablation}
Table~\ref{tab:main} separates the spatial feature extension from regularization. In nested development, adding radial terms to least squares reduces MAE by 1.494\,mm; ridge contributes a further mean difference of 0.067\,mm. The latter is small and inconsistent across scenes. Radial ridge improves on global affine least squares in five of six development scenes, not all six.

\begin{table}[htbp]\centering\small
\caption{Common-backbone, common-budget calibration ablation. Development uses nested outer-fold models; test uses frozen final models. MAE/RMSE in mm, $\delta_1$ in percent.}\label{tab:main}
\resizebox{\textwidth}{!}{\begin{tabular}{lrrrrr}\toprule
Method & Dev MAE & Test MAE & Test RMSE & AbsRel & $\delta_1$ \\\midrule
@@TABLE@@
\bottomrule\end{tabular}}
\end{table}

On new sequences, radial ridge has a paired video-mean difference of @@DELTA@@\,mm versus affine least squares, with descriptive interval [@@LOW@@, @@HIGH@@]\,mm and improvement in @@WINS@@/4 videos. @@DIRECTION@@ Radial ridge has slightly higher test MAE than affine least squares, and its RMSE and threshold accuracy are also worse, despite lower AbsRel. Unregularized radial calibration has slightly lower MAE than affine but higher RMSE. Both radial variants improve the two cecum videos and worsen sigmoid and transverse. Ridge is worse than unregularized radial calibration in all four test videos. Thus, development-selected regularization does not retain an advantage on new data. Table~\ref{tab:video} reports every video rather than only the pooled average.

\begin{table}[htbp]\centering\small
\caption{New-sequence per-video MAE (mm).}\label{tab:video}
\resizebox{\textwidth}{!}{\begin{tabular}{lrrrr}\toprule
Video & Affine LS & Affine ridge & Radial LS & Radial ridge \\\midrule
@@SEQ@@
\bottomrule\end{tabular}}
\end{table}
\begin{figure}[htbp]\centering\includegraphics[width=\textwidth]{test_per_video.pdf}
\caption{Frozen calibration comparison on all four newly acquired sequences. Two cecum textures are not independent anatomical subjects.}\end{figure}

\subsection{Historical fixed coefficients}
The supplied historical radial head reproduces 9.763\,mm frame-weighted MAE versus 10.514\,mm for its affine control on 92 reused frames, with video-mean improvement in all eight videos. This is a reproducible fixed-head observation. It does not establish how the original fit or $\lambda$ was obtained. On the present new sequences, the historical affine and radial heads give @@HISTAFF@@ and @@HISTRAD@@\,mm, respectively, with improvement in @@HISTWINS@@/4 videos. These controls have unresolved fitting provenance and are excluded from the common-budget ablation.

\subsection{Radius and depth-conditioned residuals}
Figure~\ref{fig:regions} and Table~\ref{tab:regions} describe where the radial mapping changes errors. Comparisons within depth bands are necessary because central pixels and peripheral pixels may have different depth distributions. Support varies substantially across joint bins; zero-support cells are shown as dashes, not zero error. Radial ridge reduces marginal edge MAE from 6.948 to 4.827 mm while increasing center MAE from 14.498 to 16.866 mm. Within the far-depth edge bin, error instead increases from 8.365 to 17.918 mm; that cell has support in only two videos. There are no valid center pixels in the 0--10 mm bin. These differences show that the effect is not uniform across radius and depth. A reduction in peripheral MAE is compatible with empirical spatial-bias compensation; it is not proof of vignetting correction, and a scalar-depth remapping does not itself undistort image coordinates or calibrate camera rays.

\begin{figure}[htbp]\centering\includegraphics[width=\textwidth]{test_radius_depth.pdf}
\caption{New-test radius-stratified MAE, shown marginally and within reference depth bands. Center, middle and edge correspond to fixed normalized-radius intervals. Affine LS (blue solid line, open circles) and affine ridge (orange dashed line, crosses) nearly coincide: the maximum absolute difference across these strata is below $10^{-6}$ mm. Distinct line styles and markers indicate both methods without displacing their values.}\label{fig:regions}\end{figure}
\begin{table}[htbp]\centering\small
\caption{Region support and paired aggregate error. Delta is radial ridge minus affine LS. All errors in mm; pixel counts describe support, not independent sample size.}\label{tab:regions}
\resizebox{\textwidth}{!}{\begin{tabular}{llrrrr}\toprule
Radius & Depth (mm) & Pixels & Affine LS & Radial ridge & Delta \\\midrule
@@REGIONS@@
\bottomrule\end{tabular}}
\end{table}

\input{sampling_results.tex}

\input{independent_validation.tex}
\input{object_recalibration.tex}

\section{Implications for lesion measurement}
The radial head is connected to an optional measurement interface that requires a lesion mask and actual calibrated per-pixel camera rays. For axial depth $Z$, a ray $\mathbf r$ is back-projected as $\mathbf P=Z\mathbf r/r_z$. The fixed endpoint is radial-MAD-filtered PCA 1st--99th percentile extent. It is not a full caliper diameter or a tissue-surface resection margin. Analytical tests verify recovery of a known field and consistency of the axial-depth/ray interface; they are not physical or clinical accuracy validation.

The repository's separate 320-case synthetic experiment and 720-point-cloud shape experiment do not use this radial head. They therefore cannot substantiate end-to-end sizing benefits. The former uses nominal generator dimensions, supplied masks and a photometric fallback; its four size bins are not clinical stages. In the latter rerun, fixed and adaptive boundary corrections have aggregate errors 2.268 and 2.292\,mm versus 1.346\,mm for the uncorrected estimator. The corrections are not uniformly beneficial.

Earlier descriptions of the repository's geometric components also require care. Its current orientation optimizer minimizes trimmed log-volume with radial MAD filtering, not Huber M-estimation. Its legacy heat-distance module combines a heat-kernel estimate and graph shortest paths, rather than implementing all steps of the standard Heat Method \citep{Crane2013}. A separate standard implementation elsewhere in the repository does not establish which implementation produced a historical result. Large estimated geodesic-to-chord ratios on uncalibrated clinical images cannot validate physical dimensions without independent reference measurements. We consequently exclude those demonstrations from evidence for clinical measurement accuracy.

\subsection{How depth error propagates into size}
Let $\mathbf s_i=\mathbf r_i/r_{z,i}$ be a fixed calibrated ray and let $L=\|Z_i\mathbf s_i-Z_j\mathbf s_j\|$ denote a two-endpoint chord. With exact rays and fixed endpoints, the reverse triangle inequality gives
\begin{equation}
|\hat L-L|\leq |\hat Z_i-Z_i|\,\|\mathbf s_i\|+
|\hat Z_j-Z_j|\,\|\mathbf s_j\|.
\end{equation}
Thus depth errors at the actual measurement endpoints, weighted by ray direction, matter directly. A lower image-average depth MAE does not ensure a smaller diameter error. Under a common multiplicative depth error $\hat Z=(1+\epsilon)Z$ with $1+\epsilon>0$, fixed-endpoint chord length scales by the same factor. Spatially varying radial errors need not behave as a common scale factor. The bound does not include ray-calibration error, mask uncertainty, or changes to filtering and PCA axes; it is not an accuracy guarantee for the implemented percentile endpoint. These additional effects require paired physical size evaluation.

\section{Discussion and limitations}
The four-way design isolates an empirical spatial extension under a common backbone and calibration budget. It avoids attributing gains from different data amounts, test-reference alignment or a larger neural model to the calibration formula. Nevertheless, the study is small and exploratory in its development history. There are only six c1 development scenes and four new C3VD videos, with repeated segment geometry. The additional device/domain test contains only two ex vivo specimens and confounds hardware with anatomy and acquisition conditions. The datasets' registered depth references have uncertainty, and pretraining overlap is not fully auditable.

Regularization strength depends on feature scaling, sample weighting and loss normalization. Reporting only a numerical $\lambda$ is insufficient. Here these definitions and fitted transforms are saved explicitly; the boundary selections also caution against claiming an optimal regularizer. The fitting loss emphasizes inverse-depth accuracy whereas model selection targets depth MAE. Other endpoints may rank the methods differently.

Image-centered radial coordinates assume a fixed crop and framing. They may encode camera position, anatomy or depth distribution instead of optical physics. Establishing an optical mechanism would require independently calibrated distortion, controlled illumination and camera/texture variation. The additional held-out device/domain and two independent physical specimens expose severe transfer error, but cannot isolate which acquisition factor causes it.

The post-hoc sampling study shows that a reproducible development gain can coexist with uncertain transfer: increasing the number of calibration pixels is not equivalent to increasing anatomical or device diversity. The leave-one-video-out analysis is especially sensitive to which segments are represented. Neither diagnostic establishes the cause of the shift; matched camera, texture and depth-distribution experiments are needed to separate these factors. A four-parameter extension remains methodologically modest, and the matched EndoDAC comparison does not establish superiority over domain-specific depth systems or comprehensive contemporary SOTA coverage.

A clinical sizing claim would require paired physical size references, calibrated rays, lesion masks or endpoints, object-level splits and raw repeated-observer measurements. An audit of the locally obtained 42-video Polyp-Size archive \citep{Song2025} confirmed millimetre size labels, but found no camera calibration, lesion masks, patient identifiers or raw observer measurements. The post-resection maximum-caliper endpoint also differs from the implemented percentile extent. Consequently no clinical sizing error or recomputed observer ICC is reported. Camera matrices from SERV-CT cannot substitute for calibration of those clinical videos. Agreement statistics must specify their model and endpoint. Public segmentation masks or literature-reported ICC values cannot replace this evidence. No conclusions about clinical risk categories or treatment decisions follow from the present depth results.

\section{Conclusion}
We provide a reproducible radial-affine fitting procedure, controlled ablations and a frozen new-sequence phantom evaluation. @@CONCLUSION@@ The contribution is a bounded empirical study of spatial depth calibration, with its provenance, uncertainty and downstream measurement limits made explicit.

\section*{Data and code availability}
Public depth dataset sources are C3VD, C3VDv2 and SERV-CT \citep{Bobrow2023,Golhar2026,Edwards2022}; applicable source licenses remain in force. The local research package includes selection manifests, hashes, frozen heads, per-frame and per-video results, joint region tables and executable scripts. At this draft stage, no public code-archive DOI is claimed. The historical fixed-head fitting record remains unavailable. A public archival release and complete environment specification should accompany submission.

\section*{Declaration of competing interest}
The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.
\section*{Funding}
This work was supported by the Science and Technology Research Program of Chongqing Municipal Education Commission (Grant No. QN202404018).

\input{implementation_appendix.tex}

\begin{thebibliography}{00}
\bibitem[Bobrow et al.(2023)]{Bobrow2023} Bobrow, T.L., Golhar, M., Vijayan, R., Akshintala, V.S., Garcia, J.R., Durr, N.J., 2023. Colonoscopy 3D video dataset with paired depth from 2D--3D registration. Medical Image Analysis 90, 102956. \url{https://doi.org/10.1016/j.media.2023.102956}.
\bibitem[Golhar et al.(2026)]{Golhar2026} Golhar, M.V., Galeano Fretes, L.S., Ayers, L., Akshintala, V.S., Bobrow, T.L., Durr, N.J., 2026. C3VDv2 -- Colonoscopy 3D video dataset with enhanced realism. Scientific Data 13, 1273. \url{https://doi.org/10.1038/s41597-026-07471-1}.
\bibitem[Yang et al.(2024)]{Yang2024} Yang, L., Kang, B., Huang, Z., Zhao, Z., Xu, X., Feng, J., Zhao, H., 2024. Depth Anything V2. arXiv:2406.09414. \url{https://arxiv.org/abs/2406.09414}.
\bibitem[Crane et al.(2013)]{Crane2013} Crane, K., Weischedel, C., Wardetzky, M., 2013. Geodesics in heat: A new approach to computing distance based on heat flow. ACM Transactions on Graphics 32(5), Article 152. \url{https://doi.org/10.1145/2516971.2516977}.
\bibitem[Edwards et al.(2022)]{Edwards2022} Edwards, P.J.E., Psychogyios, D., Speidel, S., Maier-Hein, L., Stoyanov, D., 2022. SERV-CT: A disparity dataset from cone-beam CT for validation of endoscopic 3D reconstruction. Medical Image Analysis 76, 102302. \url{https://doi.org/10.1016/j.media.2021.102302}.
\bibitem[Cui et al.(2024)]{Cui2024} Cui, B., Islam, M., Bai, L., Wang, A., Ren, H., 2024. EndoDAC: Efficient Adapting Foundation Model for Self-Supervised Depth Estimation from Any Endoscopic Camera. arXiv:2405.08672. \url{https://arxiv.org/abs/2405.08672}.
\bibitem[Song et al.(2025)]{Song2025} Song, Y., Du, S., Wang, R., et al., 2025. Polyp-Size: A Precise Endoscopic Dataset for AI-Driven Polyp Sizing. Scientific Data. \url{https://doi.org/10.1038/s41597-025-05251-x}.
\end{thebibliography}
\end{document}
'''
direction=f'The aggregate MAE changes by {percent:+.2f} percent relative to affine LS (positive denotes reduction).'
conclusion='Radial features improved development MAE, but the selected ridge regularization did not preserve that advantage on new sequences. Independent ex vivo device/domain transfer produced large errors for all compared heads. Post-hoc object-disjoint recalibration reduced these errors, but two previously observed specimens do not establish generalization to further objects. Spatial failures and the matched EndoDAC comparison limit a claim of general superiority.'
replace={'AUTHORS':authors,'TESTVALUES':', '.join(f"{test[m]['mae_mm']:.3f}" for m in methods),'WINS':str(comparison['improved_videos']),'DELTA':f'{delta:+.3f}','LOW':f'{ci[0]:+.3f}','HIGH':f'{ci[1]:+.3f}','TABLE':'\n'.join(row),'SEQ':'\n'.join(seq),'REGIONS':'\n'.join(rr),'DIRECTION':direction,'HISTAFF':f"{test['historical_affine']['mae_mm']:.3f}",'HISTRAD':f"{test['historical_radial']['mae_mm']:.3f}",'HISTWINS':str(hist['improved_videos']),'CONCLUSION':conclusion}
for k,v in replace.items():text=text.replace('@@'+k+'@@',v)
if '@@' in text:raise ValueError('Unfilled manuscript placeholder')
(MAN/'main_cmig.tex').write_text(text)
(MAN/'HIGHLIGHTS_CMIG.md').write_text('''# Highlights

- Radial and global affine depth heads are compared under equal pixel budgets.
- Scene-grouped validation separates spatial terms from ridge regularization.
- Frozen transfer includes four phantom videos and two independent ex vivo specimens.
- Matched EndoDAC and generic-backbone tests expose cross-device/domain failures.
- Clinical lesion sizing remains outside the validated endpoint.
''')
(MAN/'COVER_LETTER_CMIG.md').write_text(f'''# Draft cover letter — requires author review before submission

Dear Editor,

Please consider “Radial Affine Calibration of Monocular Endoscopic Depth: Controlled Ablation and Cross-Device Transfer Evaluation” for Computerized Medical Imaging and Graphics.

The manuscript examines a small spatial calibration head for frozen monocular depth. A four-way, equal-budget comparison separates radial features from ridge regularization. Six development scenes support nested scene-grouped selection; four newly acquired C3VD sequences are evaluated after coefficients and code are frozen. Global affine LS and radial ridge yield {test['affine_ls']['mae_mm']:.3f} and {test['radial_ridge']['mae_mm']:.3f} mm test MAE, with improvement in {comparison['improved_videos']}/4 videos. We report paired video differences and radius-by-depth residuals. A further frozen evaluation on two independent SERV-CT specimens uses eight heads, including EndoDAC and Depth Anything V2 Large under the same calibration budget. All eight yield MAE above 53 mm, exposing failure of direct device/domain transfer; this negative result is reported explicitly.

The study addresses metric calibration and medical-image geometry. It does not claim clinical lesion-sizing validation, correction of a proven optical mechanism, or state-of-the-art performance. Synthetic point-cloud and clinical-image demonstrations are not presented as physical measurement ground truth.

The manuscript preserves the supplied author and funding information. Before sending, the authors must confirm authorship and affiliation details, the corresponding email, funding and conflict statements, originality, exclusive submission, and approval by all authors. This draft does not assert those approvals on their behalf.

Sincerely,
Shuyu Xiong, on behalf of the authors
''')
print('Updated CMIG source, highlights and draft cover letter')
