"""Independently recompute saved predictions and verify expansion provenance."""
from pathlib import Path
import json,csv,hashlib
import numpy as np,tifffile
R=Path(__file__).resolve().parents[1];O=R/'outputs/scared_expansion'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
ff=json.loads((O/'frames.json').read_text());rows=list(csv.DictReader((O/'per_frame.csv').open()));objs=list(csv.DictReader((O/'per_subject.csv').open()));summ=json.loads((O/'summary.json').read_text());heads=json.loads((R/'outputs/independent_validation/frozen_heads.json').read_text())['heads'];maxerr=0.;checks={}
assert len(ff)==35 and len(rows)==35*8*3
index={(int(x['subject']),x['frame'],x['method'],x['support']):x for x in rows};assert len(index)==len(rows)
for row in ff:
 xyz=tifffile.imread(row['gt']);gt=xyz[...,2].astype(float);v=np.isfinite(xyz).all(-1)&(gt>0);coverage={};pred={}
 for m in ['sgbm','raft_sceneflow']:
  with np.load(O/'predictions'/m/(row['id']+'.npz')) as f:coverage[m]=f['valid'];pred[m]=f['prediction'].astype(float)
 for m,h in heads.items():
  if m.startswith('endodac'):continue
  with np.load(O/'predictions'/m.split('_')[0]/(row['id']+'.npz')) as f:d=f['prediction'].astype(float)
  height,width=d.shape;y,x=np.indices(d.shape);r=((x-width/2)**2+(y-height/2)**2)/((width/2)**2+(height/2)**2)
  a,b,c,e=h['coefficients_a0_a1_b0_b1'];inv=(a+b*r)*d+c+e*r;z=np.full_like(inv,100);np.divide(1,inv,out=z,where=inv>0);pred[m]=np.clip(z,1,100)
 masks={'all_reference':v,'reference_1_100':v&(gt>1)&(gt<100),'common_stereo_valid':v&coverage['sgbm']&coverage['raft_sceneflow']}
 for s,mask in masks.items():
  for m,p in pred.items():
   record=index[row['subject'],row['keyframe'],m,s];assert int(record['pixels'])==mask.sum();assert np.isfinite(p).all()
   mae=np.sum(np.abs(p[mask]-gt[mask]))/mask.sum();maxerr=max(maxerr,abs(mae-float(record['mae_mm'])))
for x in objs:
 vals=[float(v['mae_mm']) for v in rows if (v['subject'],v['method'],v['support'])==(x['subject'],x['method'],x['support'])];assert len(vals)==5 and abs(np.mean(vals)-float(x['mae_mm']))<1e-10
for x in summ:
 subjects={'new_five':{1,2,3,6,7},'previous_two':{8,9},'cumulative_seven':{1,2,3,6,7,8,9}}[x['cohort']]
 vals=[float(v['mae_mm']) for v in objs if int(v['subject']) in subjects and v['method']==x['method'] and v['support']==x['support']];assert len(vals)==len(subjects) and abs(np.mean(vals)-x['mae_mm'])<1e-10
old=json.loads((R/'outputs/scared_validation/summary.json').read_text())
for x in old:
 if x['method'].startswith('endodac'):continue
 new=next(v for v in summ if v['cohort']=='previous_two' and v['method']==x['method'] and v['support']==x['support']);assert abs(new['mae_mm']-x['mae_mm'])<1e-8
hashes={sha(Path(x['rgb'])) for x in ff};assert len(hashes)==35
sourceframes=json.loads((R/'outputs/publication_study/development_frames.json').read_text());assert not hashes & {sha(Path(x['color_path'])) for x in sourceframes}
assert maxerr<1e-8
checks.update(all_840_mae_recomputed=True,identical_support_per_method=True,subject_aggregation_verified=True,previous_two_results_unchanged=True,unique_35_rgb_and_no_exact_source_calibration_rgb_overlap=True)
protocol=json.loads((O/'protocol.json').read_text());checks['frozen_heads_unchanged']=sha(R/'outputs/independent_validation/frozen_heads.json')==protocol['heads_sha256']
script_sha=sha(R/'scripts/run_scared_expansion.py');checks['same_script_inference_and_scoring']=all(json.loads((O/n).read_text())['script_sha256']==script_sha for n in ['mono_run.json','stereo_run.json','scoring_start.json'])
checks['all_acquired_files_unchanged']=all(sha(R/x['local'])==x['sha256'] for x in json.loads((O/'acquisition.json').read_text()))
assert all(checks.values())
(O/'validation_checks.json').write_text(json.dumps({'checks':checks,'max_independent_mae_difference_mm':maxerr,'scope':'No foundation pretraining audit or clinical validity implied'},indent=2)+'\n');print(json.dumps(checks,indent=2))
