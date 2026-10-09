"""Curated source-checked bibliography and claim-level citation updates."""
from pathlib import Path
import json,re,zipfile,hashlib
R=Path(__file__).resolve().parents[1];O=R/'outputs/reference_audit';M=json.loads((O/'crossref_metadata.json').read_text())
DOIS={
'Bobrow2023':'10.1016/j.media.2023.102956','Golhar2026':'10.1038/s41597-026-07471-1','Yang2024':'10.52202/079017-0688','Crane2013':'10.1145/2516971.2516977','Edwards2022':'10.1016/j.media.2021.102302','Cui2024':'10.1007/978-3-031-72089-5_20','Song2025':'10.1038/s41597-025-05251-x','Lipson2021':'10.1109/3DV53792.2021.00032','Ranftl2022':'10.1109/TPAMI.2020.3019967','Ozyoruk2021':'10.1016/j.media.2021.102058','Shao2022':'10.1016/j.media.2021.102338','Ou2026':'10.1609/aaai.v40i10.37766','Mayer2016':'10.1109/CVPR.2016.438','Hirschmuller2008':'10.1109/TPAMI.2007.1166','Zhang2000':'10.1109/34.888718','Hoerl1970':'10.1080/00401706.1970.10488634','Varma2006':'10.1186/1471-2105-7-91','Koo2016':'10.1016/j.jcm.2016.02.012','Bland1986':'10.1016/S0140-6736(86)90837-8'}
DOIS.update({"Wang2025":"10.1016/j.media.2025.103534","Liu2026":"10.1016/j.compmedimag.2026.102734","LozaGalindo2026":"10.1038/s41597-026-07775-2"})
records={}
for key,doi in DOIS.items():
 m=M[doi];assert m.get('title') and m.get('author'),key
 title=m['title'][0].replace('\xa0',' ')
 if key=='Crane2013':title+=': '+m['subtitle'][0]
 records[key]={'key':key,'title':title,'authors':[{'given':a.get('given',''),'family':a['family']} for a in m['author']],'year':m['published']['date-parts'][0][0],'venue':m['container-title'][-1],'volume':m.get('volume',''),'issue':m.get('issue',''),'pages':m.get('page',m.get('article-number','')),'doi':doi,'url':'https://doi.org/'+doi,'verification':'Crossref DOI/title/authors/year plus publisher or author primary source for cited claim','type':'article'}
records['Yang2024'].update(venue='Advances in Neural Information Processing Systems',volume='37',type='inproceedings')
records['Cui2024'].update(venue='Medical Image Computing and Computer Assisted Intervention -- MICCAI 2024, Lecture Notes in Computer Science',volume='15006',type='inproceedings')
for key in ['Lipson2021','Mayer2016']:records[key]['type']='inproceedings'
records['Crane2013']['pages']='152:1--152:11'
records['Bland1986'].update(title='Statistical methods for assessing agreement between two methods of clinical measurement',authors=[{'given':'J. Martin','family':'Bland'},{'given':'Douglas G.','family':'Altman'}])
records['Hirschmuller2008']['authors'][0]['family']='Hirschmüller'
records['Mayer2016']['authors'][2]['family']='Häusser'
records['Ranftl2022']['authors'][0]['given']='René'
def manual(key,title,authors,year,venue,volume,pages,url,kind='article'):
 records[key]={'key':key,'title':title,'authors':[{'family':a,'given':b} for a,b in authors],'year':year,'venue':venue,'volume':volume,'issue':'','pages':pages,'doi':'','url':url,'verification':'Publisher/author primary page checked; no unverified DOI added','type':kind}
manual('Allan2021','Stereo Correspondence and Reconstruction of Endoscopic Data Challenge',[('Allan','Max'),('McLeod','Jonathan'),('Wang','Congcong'),('others','')],2021,'arXiv preprint arXiv:2101.01133','','','https://arxiv.org/abs/2101.01133')
manual('Cawley2010','On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation',[('Cawley','Gavin C.'),('Talbot','Nicola L. C.')],2010,'Journal of Machine Learning Research','11','2079--2107','https://jmlr.org/papers/v11/cawley10a.html')
manual('Bhat2023','ZoeDepth: Zero-shot Transfer by Combining Relative and Metric Depth',[('Bhat','Shariq Farooq'),('Birkl','Reiner'),('Wofk','Diana'),('Wonka','Peter'),('Müller','Matthias')],2023,'arXiv preprint arXiv:2302.12288','','','https://arxiv.org/abs/2302.12288')
manual('Lawrence2026','Geometry-Aware Depth-Guided Explainable Multimodal Polyp Size Estimation: A Fusion Model Beyond RGB',[('Lawrence','Krispian'),('Goparaju','Usha'),('Lamb','Luis')],2026,'Proceedings of The 9th International Conference on Medical Imaging with Deep Learning, Proceedings of Machine Learning Research','315','3230--3244','https://proceedings.mlr.press/v315/lawrence26a.html','inproceedings')
assert len(records)==26
support={
'Wang2025':'Photometric-invariant cycle training; not locally reproduced',
'Liu2026':'Intrinsics-guided scale-consistent colonoscopy reconstruction; not equivalent to clinical size validation',
'LozaGalindo2026':'Dynamic endoscopy dataset and alignment-aware evaluation; not evaluated locally',
'Lawrence2026':'Direct multimodal polyp-size regression; not a comparable local depth-MAE result',
'Ranftl2022':'Relative-depth generalization versus metric scale; no local MiDaS benchmark claimed',
'Bhat2023':'Relative pretraining and learned metric heads; context only, not evaluated here',
'Ozyoruk2021':'Endoscopic data and self-supervised depth/odometry context',
'Shao2022':'Appearance flow addresses brightness inconsistency; not reproduced here',
'Ou2026':'Recent illumination-decoupling work; no claim that radial head implements this mechanism',
'Mayer2016':'SceneFlow synthetic dataset provenance, not proof of this exact checkpoint training history',
'Hirschmuller2008':'Semiglobal matching algorithm lineage; OpenCV SGBM is an implementation variant',
'Zhang2000':'Camera parameter calibration distinguished from empirical depth remapping',
'Hoerl1970':'Ridge regression origin; no invention claim',
'Varma2006':'Nested model selection and error-estimation bias',
'Cawley2010':'Selection overfitting and adaptive evaluation caveat',
'Koo2016':'Choosing/reporting ICC model for future observer study, no ICC newly measured',
'Bland1986':'Method agreement needs direct differences; no clinical agreement study claimed'}
for k,v in records.items():v['supports']=support.get(k,'Original dataset or implemented method; existing citation scope retained')
(O/'verified_references.json').write_text(json.dumps(records,indent=2,ensure_ascii=False)+'\n')
def tex(s):
 s=str(s).replace('&',r'\&').replace('%',r'\%').replace('_',r'\_').replace('–','--').replace('—','---')
 for c,t in {'ü':r'\"{u}','ä':r'\"{a}','ö':r'\"{o}','é':r"\'{e}",'Ö':r'\"{O}'}.items():s=s.replace(c,t)
 return s

def initials(given):return ''.join(x[0]+'.' for x in re.findall(r'[^\W\d_]+',given,re.UNICODE))
def entry(rec):
 authors=rec['authors'];shown=authors if len(authors)<=7 else authors[:3]+[{'family':'others','given':''}]
 author_text=', '.join('et al.' if a['family']=='others' else tex(a['family'])+', '+tex(initials(a['given'])) for a in shown)
 if len(authors)==1:label=tex(authors[0]['family'])
 elif len(authors)==2:label=tex(authors[0]['family'])+' and '+tex(authors[1]['family'])
 else:label=tex(authors[0]['family'])+' et al.'
 where=tex(rec['venue'])
 if rec['volume']:where+=' '+tex(rec['volume'])
 if rec['issue']:where+='('+tex(rec['issue'])+')'
 if rec['pages']:where+=', '+tex(rec['pages'].replace('--','-').replace('-','--'))
 return r'\bibitem['+label+'('+str(rec['year'])+')]'+ '{'+rec['key']+'} '+author_text+', '+str(rec['year'])+'. '+tex(rec['title'])+'. '+where+r'. \url{'+rec['url']+'}.'
# Alphabetical bibliography for the existing author-year style.
ordered=sorted(records.values(),key=lambda r:(r['authors'][0]['family'].lower(),r['year']))
bibliography='\n'.join(entry(r) for r in ordered)
bib=[]
for r in ordered:
 fields={'author':' and '.join('others' if a['family']=='others' else a['family']+', '+a['given'] for a in r['authors']),'title':'{'+r['title']+'}','year':str(r['year']),'booktitle' if r['type']=='inproceedings' else 'journal':r['venue'],'volume':str(r['volume']),'number':str(r['issue']),'pages':str(r['pages']).replace('--','-').replace('-','--'),'doi':r['doi'],'url':r['url']}
 bib.append('@'+r['type']+'{'+r['key']+',\n'+',\n'.join('  '+k+' = {'+(v if k=='url' else tex(v))+'}' for k,v in fields.items() if v)+'\n}')
related=r'''% BEGIN VERIFIED RELATED WORK
\subsection{Related work and the scope of this study}
Relative-depth generalization and metric-depth recovery are related but distinct problems. MiDaS studies cross-dataset transfer with a training objective invariant to depth scale and range \citep{Ranftl2022}, whereas ZoeDepth combines relative-depth pretraining with domain-specific metric heads \citep{Bhat2023}. Depth Anything V2 further develops generic relative-depth backbones using synthetic supervision and pseudo-labelled real images \citep{Yang2024}. These works motivate separating representation quality from downstream metric calibration; none establishes that a source-fitted calibration will retain its scale in a new endoscopic domain.

Endoscopic depth methods also address image-formation effects within the learned model. EndoSLAM combines a dataset contribution with self-supervised depth and visual odometry \citep{Ozyoruk2021}. Appearance-flow learning explicitly addresses brightness inconsistency \citep{Shao2022}; EndoDAC adapts foundation features to surgical images \citep{Cui2024}; and DeLightMono models uneven illumination through a learned decomposition \citep{Ou2026}. These are relevant methodological alternatives, not all reproduced baselines in this study. Our frozen-feature regression does not estimate illumination or reflectance, and its radial coefficients cannot be interpreted as equivalent physical corrections.

Camera calibration estimates image-projection and distortion parameters, as exemplified by planar-target calibration \citep{Zhang2000}. The present inverse-depth remapping operates downstream of image formation and does not replace camera calibration. Stereo methods instead recover geometric information from correspondence and known camera geometry: semiglobal matching \citep{Hirschmuller2008} and RAFT-Stereo \citep{Lipson2021} are therefore useful references, but their additional inputs must be distinguished from monocular inference. SceneFlow provides synthetic stereo training data \citep{Mayer2016}; it is separate from the physical endoscopic references used for evaluation.

Recent work makes the distinction between calibration transfer and end-to-end reconstruction particularly important. MonoPCC redesigns the self-supervised photometric constraint as a cycle to address brightness variation \citep{Wang2025}. Liu et al. combine synthetic pretraining, camera-intrinsics-guided pseudo-depth and joint depth--pose learning to improve scale consistency in colonoscopy \citep{Liu2026}. Our four-coefficient mapping does not replace these learned geometric constraints, and we have not established superiority over either method under a matched protocol.

Recent validation resources also broaden the required scope. DRENDS provides calibrated stereo observations and reference geometry under tissue deformation and different illumination conditions \citep{LozaGalindo2026}; these dynamic conditions are not covered by the present selected-keyframe evaluation. For the distinct sizing endpoint, Lawrence et al. fuse RGB, depth and geometric cues for direct polyp-size regression \citep{Lawrence2026}. Such size-estimation results cannot be compared numerically with our axial-depth MAE, and the present study does not establish clinical sizing accuracy. These 2025--2026 developments motivate stronger domain baselines, dynamic validation and direct physical-size evaluation rather than a broad state-of-the-art claim.

The contribution here is a controlled study of a small spatial calibration model and its transfer limits, not a new foundation architecture or a comprehensive comparison with current endoscopic depth systems. This positioning also separates axial-depth validation from physical lesion-size and observer-agreement validation.
% END VERIFIED RELATED WORK
'''
backup=O/'manuscripts_before_reference_revision.zip'
if not backup.exists():
 with zipfile.ZipFile(backup,'w',zipfile.ZIP_DEFLATED) as z:
  for folder in ['manuscript_cmig','manuscript_mbec']:
   for p in (R/folder).glob('*.tex'):z.write(p,str(p.relative_to(R)))
for folder,file in [('manuscript_mbec','main_mbec.tex'),('manuscript_cmig','main_cmig.tex')]:
 d=R/folder;p=d/file;s=p.read_text()
 # Directly scope dataset citations to depth data, rather than to unvalidated clinical sizing.
 old=r'Quantitative interpretation of monocular endoscopy depends on the relationship between relative image-based depth and physical geometry. Dense phantom reference datasets permit depth evaluation under real acquisition optics, while clinical lesion dimensions additionally require calibrated camera rays, lesion boundaries and a suitable physical reference \citep{Bobrow2023,Golhar2026}. These endpoints must not be conflated: lower axial-depth error does not necessarily imply lower error in a three-dimensional diameter.'
 new=r'Quantitative interpretation of monocular endoscopy depends on the relationship between relative image-based depth and physical geometry. Dense phantom reference datasets permit depth evaluation under real acquisition optics \citep{Bobrow2023,Golhar2026}. For the geometric sizing interface considered here, physical interpretation additionally requires calibrated camera rays, lesion boundaries and a compatible physical reference. Polyp-Size illustrates a distinct endpoint based on post-resection caliper measurements \citep{Song2025}. These endpoints must not be conflated: lower axial-depth error does not necessarily imply lower error in a three-dimensional diameter.'
 s=s.replace(old,new)
 if '% BEGIN VERIFIED RELATED WORK' in s:s=re.sub(r'% BEGIN VERIFIED RELATED WORK.*?% END VERIFIED RELATED WORK\n',lambda m:related,s,flags=re.S)
 else:s=s.replace(r'\subsection{Workflow overview}',related+'\n'+r'\subsection{Workflow overview}')
 if 'classical ridge-regression' not in s:s=s.replace(r'\subsection{Explicit fitting objective}',r'\subsection{Explicit fitting objective}'+'\n'+r'The penalty belongs to the classical ridge-regression family \citep{Hoerl1970}; the study tests its incremental effect rather than claiming a new regularization principle.'+'\n')
 if 'Model selection must be contained' not in s:
  target=re.search(r'\\subsection\{[^}]*[Ss]election[^}]*\}',s)
  assert target,folder
  text=r'Model selection must be contained within the training portion of performance estimation to reduce selection-induced optimism \citep{Varma2006,Cawley2010}. This principle motivates the nested design; it does not erase prior exploratory use of the development data.'
  s=s[:target.end()]+'\n'+text+'\n'+s[target.end():]
 s=s.replace('Agreement statistics must specify their model and endpoint.',r'Agreement statistics must specify their model and endpoint, including absolute agreement versus consistency and single versus averaged ratings \citep{Koo2016}. Comparison against a physical reference additionally requires direct assessment of measurement differences and agreement, rather than correlation alone \citep{Bland1986}.')
 start=s.index(r'\begin{thebibliography}');end=s.index(r'\end{thebibliography}',start)+len(r'\end{thebibliography}')
 s=s[:start]+r'\begin{thebibliography}{99}'+'\n'+bibliography+'\n'+r'\end{thebibliography}'+s[end:];p.write_text(s)
 (d/'references.bib').write_text('\n\n'.join(bib)+'\n')
 ext=d/'scared_expansion.tex';t=ext.read_text();t=t.replace(r'checkpoint of RAFT-Stereo \citep{Lipson2021} and a fixed OpenCV SGBM configuration.',r'checkpoint of RAFT-Stereo \citep{Lipson2021}, trained on the synthetic SceneFlow resource \citep{Mayer2016}, and a fixed OpenCV SGBM configuration derived from semiglobal matching \citep{Hirschmuller2008}.');ext.write_text(t)
checks={}
for folder,file in [('manuscript_mbec','main_mbec.tex'),('manuscript_cmig','main_cmig.tex')]:
 text='\n'.join(p.read_text() for p in (R/folder).glob('*.tex'));used=set()
 for group in re.findall(r'\\cite\w*\{([^}]+)\}',text):used.update(group.split(','))
 defined=re.findall(r'\\bibitem(?:\[[^\]]+\])?\{([^}]+)\}',text)
 checks[folder]={'references':len(defined),'unique_references':len(set(defined)),'undefined_citations':sorted(used-set(defined)),'uncited_references':sorted(set(defined)-used)}
 assert len(defined)==len(records) and len(set(defined))==len(records) and not checks[folder]['undefined_citations'] and not checks[folder]['uncited_references']
(O/'citation_checks.json').write_text(json.dumps(checks,indent=2)+'\n');print(json.dumps(checks,indent=2))
