"""Range and oracle diagnostics of original frozen heads; oracle never a deployable result."""
from pathlib import Path
import os,sys,json,csv
os.environ['HF_HUB_OFFLINE']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
R=Path(__file__).resolve().parents[1];sys.path[:0]=[str(R),str(R/'scripts'),str(R/'tests')]
import numpy as np,cv2,torch
from evaluate_major_revision import frames,reference
from src.radial_affine_calibration import radius_squared,predict
from transformers import AutoImageProcessor,AutoModelForDepthEstimation
O=R/'outputs/major_revision';model=None;proc=None
heads=json.loads((R/'outputs/independent_validation/frozen_heads.json').read_text())['heads'];rows=[]
for row in frames():
 cohort=row['cohort']
 if cohort=='c3vd':path=R/'outputs/radial_confirmatory/predictions'/f"{row['scene']}_{row['frame_id']}.npz"
 elif cohort=='servct':path=R/'outputs/independent_validation/predictions/small'/f"{row['object']}_{row['frame_id']}.npz"
 elif cohort=='scared':path=R/'outputs/scared_expansion/predictions/small'/(row['id'].removeprefix('scared_')+'.npz')
 else:
  path=O/'original_drends_raw'/(row['id']+'.npz');path.parent.mkdir(exist_ok=True)
  if not path.exists():
   if model is None:
    torch.set_num_threads(4);md=R/'datasets/external_models/Depth-Anything-V2-Small-hf';proc=AutoImageProcessor.from_pretrained(md,local_files_only=True,use_fast=False);model=AutoModelForDepthEstimation.from_pretrained(md,local_files_only=True).cuda().eval()
   rgb=cv2.cvtColor(cv2.imread(str(R/row['rgb'])),cv2.COLOR_BGR2RGB)
   with torch.no_grad():raw=model(**proc(images=rgb,return_tensors='pt').to('cuda')).predicted_depth
   d=torch.nn.functional.interpolate(raw.unsqueeze(1),size=rgb.shape[:2],mode='bilinear',align_corners=False).squeeze().cpu().numpy();np.savez_compressed(path,prediction=d)
 with np.load(path) as z:d=z['prediction'].astype(float)
 gt,v=reference(row);r2=radius_squared(d.shape)
 for method in ['small_affine_ls','small_radial_ls','small_radial_ridge']:
  h=heads[method];a0,a1,b0,b1=h['coefficients_a0_a1_b0_b1'];inv=(a0+a1*r2)*d+b0+b1*r2
  for bound in [100,300]:
   p=np.clip(np.divide(1.,inv,out=np.full_like(inv,float(bound)),where=inv>0),1,bound)
   if bound==100:assert np.allclose(p,predict(h,d))
   for support,mask in [('all_reference',v),('reference_1_100',v&(gt>1)&(gt<100))]:
    if not mask.any():continue
    x=p[mask];y=gt[mask];off=np.median(y-x);scale=np.median(y/np.maximum(x,1e-6))
    rows.append({'cohort':cohort,'object':row['object'],'frame':row['id'],'method':method,'bound':bound,'support':support,'mae_mm':float(np.abs(x-y).mean()),'bias_mm':float((x-y).mean()),'reference_above100':float((y>100).mean()),'bound_fraction':float(((x<=1)|(x>=bound)).mean()),'nonpositive_inverse_fraction':float((inv[mask]<=0).mean()),'oracle_offset_mae_mm':float(np.abs(x+off-y).mean()),'oracle_scale_mae_mm':float(np.abs(x*scale-y).mean())})
with (O/'original_transfer_diagnostics.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
keys=['mae_mm','bias_mm','reference_above100','bound_fraction','nonpositive_inverse_fraction','oracle_offset_mae_mm','oracle_scale_mae_mm'];gk=['cohort','method','bound','support'];summ=[]
for group in sorted({tuple(r[k] for k in gk) for r in rows}):
 a=[r for r in rows if tuple(r[k] for k in gk)==group];oo=[]
 for obj in sorted({r['object'] for r in a}):
  b=[r for r in a if r['object']==obj];oo.append({k:float(np.mean([r[k] for r in b])) for k in keys})
 summ.append({**dict(zip(gk,group)),'groups':len(oo),**{k:float(np.mean([r[k] for r in oo])) for k in keys}})
(O/'original_transfer_summary.json').write_text(json.dumps(summ,indent=2));print('Completed',len(rows),'diagnostic rows')
