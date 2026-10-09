"""Independent arithmetic, provenance and reference-sensitivity checks."""
from pathlib import Path
import sys,json,hashlib,csv
import numpy as np,cv2,tifffile
R=Path(__file__).resolve().parents[1];O=R/'outputs/major_revision'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4<<20),b''):h.update(b)
 return h.hexdigest()
p=json.loads((O/'training_protocol.json').read_text());assert sha(R/'scripts/train_major_revision_baseline.py')==p['script_sha256']
for row in p['rows']:assert sha(row['rgb'])==p['training_rgb_sha256'][row['id']];assert sha(row['gt'])==p['training_gt_sha256'][row['id']]
trained=json.loads((O/'training_complete.json').read_text());assert len(trained['checkpoints'])==6
for n,h in trained['checkpoints'].items():assert sha(O/'trained_models'/n)==h
old=json.loads((R/'outputs/independent_validation/frozen_heads.json').read_text())
for n,h in old['source_sha256'].items():assert sha(R/n)==h,n
sys.path.insert(0,str(R/'scripts'));from evaluate_major_revision import frames
ff=frames();assert len(ff)==101;train=set(p['training_rgb_sha256'].values());testhashes=[sha(R/r['rgb']) for r in ff];assert not train&set(testhashes);assert len(testhashes)==len(set(testhashes))
rr=list(csv.DictReader((O/'metric_per_frame.csv').open()));ss=json.loads((O/'metric_summary.json').read_text())
for s in ss:
 a=[r for r in rr if all(str(s[k])==r[k] for k in ['cohort','regime','seed','factor','bound','support'])];objs=sorted({r['object'] for r in a})
 val=np.mean([np.mean([float(r['mae_mm']) for r in a if r['object']==o]) for o in objs]);assert np.isclose(val,s['mae_mm'],rtol=1e-12)
refs=[]
for r in ff:
 if r['cohort']!='drends':continue
 d=R/r['rgb'];d=d.parent;gt=tifffile.imread(R/r['gt']);alt=tifffile.imread(d/'depth_mm_sigma5mm.tiff');v=(cv2.imread(str(R/r['mask']),0)>0)&(cv2.imread(str(d/'mask_sigma5mm.png'),0)>0);w,h=gt.shape[1],gt.shape[0]
 for regime in ['source_only','adapted']:
  for seed in [11,23,47]:
   with np.load(O/'metric_predictions'/f'{regime}_{seed}'/(r['id']+'_identity.npz')) as z:pred=np.clip(cv2.resize(z['prediction'],(w,h)),1,100)
   m=float(np.abs(pred[v]-gt[v]).mean());n=float(np.abs(pred[v]-alt[v]).mean());difference=float(np.abs(alt[v]-gt[v]).mean());assert abs(m-n)<=difference+1e-4
   refs.append({'id':r['id'],'object':r['object'],'regime':regime,'seed':seed,'primary_ref_mae_mm':m,'sigma5mm_ref_mae_mm':n,'difference_mm':n-m})
(O/'reference_model_sensitivity.json').write_text(json.dumps(refs,indent=2))
checks={'training_rgb_gt_hashes_unchanged':True,'training_and_test_rgb_disjoint':True,'six_checkpoints_verified':True,'historical_frozen_sources_unchanged':True,'summary_independently_reaggregated':True,'test_frames':len(ff),'new_drends_frames':27,'test_objects_caution':'C3VD groups are videos; DRENDS groups are organ/setup, not independently verified donors or device serial numbers','max_same_support_reference_mae_change_mm':max(abs(r['difference_mm']) for r in refs),'reference_triangle_inequality_check':True,'seeds_are_not_independent_patients':True}
(O/'validation_checks.json').write_text(json.dumps(checks,indent=2));print(json.dumps(checks,indent=2))
