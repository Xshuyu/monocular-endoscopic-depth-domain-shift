"""Build same-protocol baseline comparison; clinical data audit remains separate."""
import csv,json,hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];out=ROOT/'outputs/next_validation'
small=json.loads((ROOT/'outputs/c3vdv2_external_validation/summary.json').read_text())
large=json.loads((ROOT/'outputs/c3vdv2_large_validation/summary.json').read_text())
candidate=json.loads((ROOT/'outputs/calibrated_depth_candidate/summary.json').read_text())
new=json.loads((ROOT/'outputs/c3vd_frozen_transfer/summary.json').read_text())
method='depth_anything_v2_calibration_inverse_affine'
def get(rows,name):return next(r for r in rows if r['method']==name and r.get('split','test')=='test')
old=[get(small['summary'],'photometric_raw'),get(small['summary'],method),get(large['summary'],method),get(candidate,'selected_fusion')]
labels=['Photometric','DA V2 Small','DA V2 Large','Frozen candidate']
new=[get(new,m) for m in ['photometric','small_affine','large_affine','frozen_candidate']]
with (out/'same_protocol_comparison.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=['dataset','method','mae_mm','rmse_mm','abs_rel','delta1_fraction']);w.writeheader()
 for dataset,rows in [('C3VDv2 old observed test',old),('C3VD new trans_t1_a',new)]:
  for label,r in zip(labels,rows):w.writerow({'dataset':dataset,'method':label,**{k:r[k] for k in ['mae_mm','rmse_mm','abs_rel','delta1_fraction']}})
fig,axes=plt.subplots(1,2,figsize=(10,4),sharey=True)
for ax,title,rows in zip(axes,['C3VDv2: 6 videos, 1 anatomy\nPreviously observed test','C3VD: 1 new video, 12 frames\nFrozen cross-dataset transfer'],[old,new]):
 bars=ax.bar(np.arange(4),[r['mae_mm'] for r in rows],color=['#999999','#3d7ca6','#235535','#c48430'])
 ax.set_xticks(np.arange(4),labels,rotation=25,ha='right');ax.set_title(title);ax.set_ylim(0,16)
 for b,r in zip(bars,rows):ax.text(b.get_x()+b.get_width()/2,b.get_height()+.15,f"{r['mae_mm']:.2f}",ha='center')
axes[0].set_ylabel('Axial depth MAE (mm), lower is better');fig.tight_layout();fig.savefig(out/'depth_comparison.png',dpi=180);fig.savefig(out/'depth_comparison.pdf');plt.close(fig)
def load_csv(path):
 with path.open() as f:return list(csv.DictReader(f))
a=load_csv(ROOT/'outputs/c3vdv2_external_validation/per_video.csv');b=load_csv(ROOT/'outputs/c3vdv2_large_validation/per_video.csv')
aa={r['scene']:float(r['mae_mm']) for r in a if r['method']==method and r['split']=='test'}
bb={r['scene']:float(r['mae_mm']) for r in b if r['method']==method and r['split']=='test'}
diff=np.array([bb[k]-aa[k] for k in sorted(aa)]);rng=np.random.default_rng(20260927)
ci=np.quantile(rng.choice(diff,(10000,len(diff))).mean(axis=1),[.025,.975])
(out/'large_small_paired.json').write_text(json.dumps({'large_minus_small_mae_mm':float(diff.mean()),'sequence_bootstrap_descriptive_95ci':ci.tolist(),'limits':'Old observed test, only one anatomy; not patient-level inference'},indent=2)+'\n')
print('Comparison and figure written; Large-minus-Small old test',diff.mean(),ci)
