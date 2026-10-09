"""Create a journal-targeted working draft; not a submission-format certification."""
from pathlib import Path
import re,shutil
R=Path(__file__).resolve().parents[1];src=R/'manuscript_cmig';out=R/'manuscript_mbec';out.mkdir(exist_ok=True)
for file in [src/'independent_validation.tex',R/'scripts/write_independent_validation_report.py']:
 s=file.read_text();s=s.replace('Target-domain calibration and broader independent-object evaluation remain future work; no target-dependent correction was applied after scoring.','Broader independent-object evaluation remains necessary; the original frozen transfer results have not been retroactively corrected.');file.write_text(s)
s=(src/'main_cmig.tex').read_text();body=s.split(r'\end{frontmatter}',1)[1]
preamble=r'''\documentclass[12pt]{article}
\usepackage{amsmath,amssymb,graphicx,booktabs,hyperref,lineno,geometry,natbib}
\geometry{a4paper,margin=1in}
\setlength{\emergencystretch}{2em}
\hypersetup{hidelinks}
\graphicspath{{../outputs/radial_confirmatory/figures/}}
\title{Calibration Transfer and Object-Disjoint Recalibration of Monocular Endoscopic Depth}
\author{Shuyu Xiong$^{1,2}$\thanks{Correspondence: shuyux@cqou.edu.com}\quad Yongtao Deng$^1$}
\date{}
\begin{document}
\maketitle
\begin{center}\small
$^1$School of Computer Science and Technology, Chongqing University of Posts and Telecommunications, Chongqing 400065, China\\
$^2$School of Electronic Information Engineering, Chongqing Open University, Chongqing 400052, China
\end{center}
\begin{abstract}
Metric interpretation of monocular endoscopic depth depends on calibration that may not transfer across acquisition domains. We assess a four-coefficient radial inverse-depth mapping against global affine calibration, separating spatial terms from ridge regularization under matched reference-pixel budgets. Six development scenes support grouped selection. Frozen models are evaluated on four previously unscored colon-phantom sequences and, subsequently, two independent ex vivo SERV-CT specimens acquired with a different endoscope system. EndoDAC and Depth Anything V2 Large provide additional frozen-feature comparisons under the same calibration protocol. Development gains do not translate into consistent external improvements: Small affine and radial ridge yield 8.794 and 8.899 mm MAE on the four phantom videos, while all eight evaluated heads exceed 53 mm MAE on SERV-CT. A separate post-hoc experiment fits calibration on one SERV-CT specimen and evaluates the other, reversing the roles without sharing object pixels. At 16,384 calibration pixels, Small affine MAE decreases from 56.245 to 19.408 mm and EndoDAC affine from 53.295 to 17.691 mm. Radial terms do not provide consistent additional benefit across backbones. The findings identify calibration transfer as an important limitation and show that reference-assisted recalibration can reduce error within this small dataset. Two specimens and exploratory reuse do not establish population generalization or clinical lesion-sizing accuracy. Calibration and evaluation records are retained for reproducibility.
\end{abstract}
\noindent\textbf{Keywords:} Endoscopic depth; Metric calibration; Domain shift; Radial regression; Independent-object evaluation
'''
(out/'main_mbec.tex').write_text(preamble+body)
for n in ['implementation_appendix.tex','sampling_results.tex','independent_validation.tex','object_recalibration.tex']:shutil.copy2(src/n,out/n)
shutil.copytree(src/'figures',out/'figures',dirs_exist_ok=True)
(out/'SUBMISSION_ROUTE.md').write_text('''# Target: Medical & Biological Engineering & Computing

Chosen route: Subscription publishing model (non-OA article). MBEC is a hybrid journal; its official page states no APC applies to the subscription route.

Official policy: https://link.springer.com/journal/11517/how-to-publish-with-us
Scope: https://link.springer.com/journal/11517/aims-and-scope

This is a working draft, not a certification of complete submission-format compliance. Confirm the current author guidelines, author/affiliation/contact details, declarations, data/code release and applicable funder requirements before submission. No manuscript has been submitted or sent to anyone.

Primary claim: calibration-transfer limitations and exploratory benefits of object-disjoint recalibration. Do not claim clinical lesion-size accuracy, uniform radial superiority or a new blind study for the post-hoc recalibration.
''')
print('Built MBEC working source')
