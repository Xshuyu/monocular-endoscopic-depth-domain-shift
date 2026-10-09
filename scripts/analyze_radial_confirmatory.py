"""Prespecified paired and radius-by-depth diagnostics; no refitting."""
import json,csv
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/radial_confirmatory';FIG=OUT/'figures';FIG.mkdir(exist_ok=True)
METHODS=['affine_ls','affine_ridge','radial_ls','radial_ridge'];LABELS=['Affine LS','Affine ridge','Radial LS','Radial ridge']
def read_csv(n):
 with (OUT/n).open() as f:return list(csv.DictReader(f))
def write_csv(n,rows):
 with (OUT/n).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
paired=[]
for phase in ['development','test']:
 if not (OUT/f'{phase}_summary.json').exists():continue
 rows=read_csv(f'{phase}_per_video.csv');scenes=sorted({r['scene'] for r in rows});lookup={(r['scene'],r['method']):float(r['mae_mm']) for r in rows}
 for a,b in [('radial_ridge','affine_ls'),('radial_ls','affine_ls'),('radial_ridge','radial_ls'),('affine_ridge','affine_ls')]+([('historical_radial','historical_affine')] if phase=='test' else []):
  delta=np.array([lookup[s,a]-lookup[s,b] for s in scenes]);rng=np.random.default_rng(20260927);ci=np.quantile(rng.choice(delta,(20000,len(delta)),replace=True).mean(1),[.025,.975])
  paired.append({'phase':phase,'a':a,'b':b,'mean_delta_mm':float(delta.mean()),'video_bootstrap_descriptive_95ci':ci.tolist(),'improved_videos':int((delta<0).sum()),'n_videos':len(delta),'video_deltas':dict(zip(scenes,delta.tolist()))})
 # Anatomical segment grouped sensitivity: repeated cecum textures kept together.
 if phase=='test':
  groups=sorted({s.split('_')[0] for s in scenes});d=np.array([np.mean([lookup[s,'radial_ridge']-lookup[s,'affine_ls'] for s in scenes if s.startswith(g+'_')]) for g in groups]);rng=np.random.default_rng(20260927)
  (OUT/'anatomical_group_sensitivity.json').write_text(json.dumps({'groups':groups,'mean_delta_mm':float(d.mean()),'descriptive_95ci':np.quantile(rng.choice(d,(20000,len(d)),replace=True).mean(1),[.025,.975]).tolist(),'note':'Only three segment groups, not independent patients; descriptive sensitivity only'},indent=2))
 data=read_csv(f'{phase}_strata.csv');groups=defaultdict(list)
 for r in data:groups[(r['method'],int(r['radius_bin']),int(r['depth_bin']),r['scene'])].append(r)
 video=[]
 for (m,rb,db,s),a in groups.items():video.append({'method':m,'radius_bin':rb,'depth_bin':db,'scene':s,'frames':len(a),'pixels':sum(int(r['pixels']) for r in a),'mae_mm':float(np.mean([float(r['mae_mm']) for r in a])),'bias_mm':float(np.mean([float(r['bias_mm']) for r in a]))})
 write_csv(f'{phase}_strata_per_video.csv',video);aggregated=[]
 for m in sorted({r['method'] for r in video}):
  for rb in range(3):
   for db in [-1,0,1,2]:
    a=[r for r in video if (r['method'],r['radius_bin'],r['depth_bin'])==(m,rb,db)]
    if a:aggregated.append({'method':m,'radius_bin':rb,'depth_bin':db,'videos':len(a),'frames':sum(r['frames'] for r in a),'pixels':sum(r['pixels'] for r in a),'mae_mm':float(np.mean([r['mae_mm'] for r in a])),'bias_mm':float(np.mean([r['bias_mm'] for r in a]))})
 write_csv(f'{phase}_strata_summary.csv',aggregated)
 fig,axes=plt.subplots(2,2,figsize=(9,6.8),sharey=True);axes=axes.ravel()
 for ax,db,title in zip(axes,[-1,0,1,2],['All depths','0–10 mm','10–30 mm','30–100 mm']):
  styles=[dict(color='#1f77b4',linestyle='-',linewidth=3,marker='o',markersize=8,markerfacecolor='white',markeredgewidth=1.6,zorder=2),dict(color='#ff7f0e',linestyle='--',linewidth=1.8,marker='x',markersize=6,markeredgewidth=1.5,zorder=4),dict(color='#2ca02c',linestyle='-',linewidth=1.8,marker='^',markersize=6,zorder=3),dict(color='#d62728',linestyle='-.',linewidth=1.8,marker='D',markersize=5,zorder=3)]
  for m,label,style in zip(METHODS,LABELS,styles):
   vals=[next((r['mae_mm'] for r in aggregated if (r['method'],r['radius_bin'],r['depth_bin'])==(m,rb,db)),np.nan) for rb in range(3)];ax.plot(range(3),vals,label=label,**style)
  ax.set_xticks(range(3),['Center','Middle','Edge']);ax.set_title(title);ax.grid(alpha=.2)
 axes[0].set_ylabel('Equal-video region MAE (mm)');fig.suptitle(phase.title()+': radius-stratified error, conditional on reference depth')
 handles,legend_labels=axes[0].get_legend_handles_labels();fig.legend(handles,legend_labels,loc='lower center',bbox_to_anchor=(.5,.035),ncol=4,fontsize=9,frameon=False)
 if phase=='test':fig.text(.5,.015,'Affine LS and affine ridge overlap (maximum difference < 0.000001 mm).',ha='center',fontsize=9)
 fig.tight_layout(rect=(0,.10,1,.96))
 for ext in ['pdf','png']:fig.savefig(FIG/f'{phase}_radius_depth.{ext}',dpi=180)
 plt.close(fig)
 if phase=='test':
  fig,ax=plt.subplots(figsize=(9,4));x=np.arange(len(scenes));width=.19
  for i,(m,label) in enumerate(zip(METHODS,LABELS)):ax.bar(x+(i-1.5)*width,[lookup[s,m] for s in scenes],width,label=label)
  ax.set_xticks(x,scenes);ax.set_ylabel('MAE (mm)');ax.legend(ncol=2);ax.grid(axis='y',alpha=.2);fig.tight_layout()
  for ext in ['pdf','png']:fig.savefig(FIG/f'test_per_video.{ext}',dpi=180)
  plt.close(fig)
(OUT/'paired_diagnostics.json').write_text(json.dumps({'scope':'Descriptive cluster bootstrap, conditional on small phantom sample; no population significance or multiplicity-adjusted inference','comparisons':paired},indent=2)+'\n')
print(json.dumps(paired,indent=2))
