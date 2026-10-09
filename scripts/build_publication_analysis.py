"""Create figures and descriptive, video-level paired summaries without retuning."""
from pathlib import Path
import csv,json
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/publication_study';FIG=OUT/'figures';FIG.mkdir(exist_ok=True)
def read(name):
 with (OUT/name).open() as f:return list(csv.DictReader(f))
labels={'constant_median':'Calibration median','photometric_raw':'Photometric','small_single_scene_affine':'Small: one calibration scene','small_affine':'Small: multi-scene affine','small_monotone':'Small: monotone bins','nested_selected':'Small: nested selection','selected_calibration':'Small: frozen spline','large_affine':'Large: multi-scene affine','endodac_affine':'EndoDAC: multi-scene affine'}
dev=json.loads((OUT/'development_summary.json').read_text());final=json.loads((OUT/'final_summary.json').read_text())
dm={r['method']:r for r in dev};fm={r['method']:r for r in final}
order=['constant_median','photometric_raw','small_single_scene_affine','small_affine','small_monotone','selected_calibration','large_affine','endodac_affine']
colors=['#aaaaaa','#888888','#84b6d6','#286493','#39896e','#d68127','#663e92','#c25264']
fig,axes=plt.subplots(1,2,figsize=(11,5),sharey=True)
for ax,table,title in zip(axes,[dm,fm],['Development: nested scene CV\n6 videos, one c1 anatomy','Locked final transfer\n3 previously unscored C3VD videos']):
 vals=[table['nested_selected' if m=='selected_calibration' and table is dm else m]['mae_mm'] for m in order]
 bars=ax.barh(range(len(order)),vals,color=colors)
 for b,v in zip(bars,vals):ax.text(v+.1,b.get_y()+b.get_height()/2,f'{v:.2f}',va='center',fontsize=8)
 ax.set_title(title,fontsize=11);ax.set_xlabel('Mean axial depth MAE (mm)');ax.set_xlim(0,17);ax.grid(axis='x',alpha=.2)
axes[0].set_yticks(range(len(order)),[labels[x].replace('frozen spline','selected calibration*') for x in order]);axes[0].invert_yaxis();fig.tight_layout()
for ext in ['pdf','png']:fig.savefig(FIG/f'comparison.{ext}',dpi=200)
plt.close(fig)
rows=read('final_per_video.csv');scenes=sorted({r['scene'] for r in rows});methods=['small_affine','small_monotone','selected_calibration','large_affine','endodac_affine']
fig,ax=plt.subplots(figsize=(9,4.5));xx=np.arange(len(scenes));width=.15
for j,m in enumerate(methods):
 vals=[float(next(r['mae_mm'] for r in rows if r['scene']==s and r['method']==m)) for s in scenes]
 ax.bar(xx+(j-2)*width,vals,width,label=labels[m],color=colors[order.index(m)])
ax.set_xticks(xx,scenes);ax.set_ylabel('Mean axial depth MAE (mm)');ax.set_ylim(0,21);ax.legend(fontsize=8,ncol=2);ax.grid(axis='y',alpha=.2);fig.tight_layout()
for ext in ['pdf','png']:fig.savefig(FIG/f'final_sequence_results.{ext}',dpi=200)
plt.close(fig)
# Descriptive bootstrap unit is video, never pixels. Conditional on these phantoms.
paired=[]
for phase,file,selected in [('development','development_per_video.csv','nested_selected'),('final','final_per_video.csv','selected_calibration')]:
 rows=read(file);scenes=sorted({r['scene'] for r in rows});lookup={(r['scene'],r['method']):float(r['mae_mm']) for r in rows}
 for a,b in [(selected,'small_affine'),('small_monotone','small_affine'),('small_affine','small_single_scene_affine'),('large_affine','small_affine'),('endodac_affine','small_affine')]:
  delta=np.array([lookup[s,a]-lookup[s,b] for s in scenes]);rng=np.random.default_rng(20260927)
  ci=np.quantile(rng.choice(delta,(20000,len(delta)),replace=True).mean(axis=1),[.025,.975])
  paired.append({'phase':phase,'a':a,'b':b,'mean_delta_mae_mm':float(delta.mean()),'descriptive_95ci':ci.tolist(),'n_videos':len(delta),'a_better_videos':int((delta<0).sum()),'scene_differences':dict(zip(scenes,delta.tolist()))})
(OUT/'paired_video_diagnostics.json').write_text(json.dumps({'interpretation':'Descriptive video bootstrap only; correlated scenes, tiny phantom sample, not patient-population uncertainty; no multiplicity-adjusted inference','comparisons':paired},indent=2)+'\n')
strata=read('development_strata.csv');summaries=[]
for region in sorted({r['stratum'] for r in strata}):
 for method in sorted({r['method'] for r in strata}):
  group=[r for r in strata if r['stratum']==region and r['method']==method]
  video_means=[np.mean([float(r['mae_mm']) for r in group if r['scene']==scene]) for scene in sorted({r['scene'] for r in group})]
  summaries.append({'stratum':region,'method':method,'mae_mm':float(np.mean(video_means)),'n_videos':len(video_means),'n_frames_with_support':len(group),'pixels':sum(int(r['pixels']) for r in group)})
with (OUT/'development_strata_summary.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(summaries[0]));w.writeheader();w.writerows(summaries)
regions=['near_0_10mm','mid_10_30mm','far_30_100mm'];methods=['small_affine','small_monotone','nested_selected','large_affine','endodac_affine']
fig,ax=plt.subplots(figsize=(9,4));x=np.arange(3)
for j,m in enumerate(methods):
 values=[next(r['mae_mm'] for r in summaries if r['method']==m and r['stratum']==region) for region in regions]
 ax.plot(x,values,marker='o',label=labels[m])
ax.set_xticks(x,['0–10 mm','10–30 mm','30–100 mm']);ax.set_ylabel('Region MAE (mm)');ax.set_xlabel('Reference depth interval (development OOF only)');ax.legend(fontsize=8);ax.grid(alpha=.2);fig.tight_layout()
for ext in ['pdf','png']:fig.savefig(FIG/f'development_depth_strata.{ext}',dpi=200)
plt.close(fig)
print(json.dumps(paired,indent=2))
