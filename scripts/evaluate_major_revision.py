"""Frozen adaptation evaluation and paired image-factor interventions."""
import os
os.environ['HF_HUB_OFFLINE']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import sys,json,csv,hashlib,argparse
import numpy as np,cv2,tifffile,torch
R=Path(__file__).resolve().parents[1];sys.path[:0]=[str(R),str(R/'scripts'),str(R/'tests')]
from train_major_revision_baseline import model,M,sha,dump
from c3vdv2_external_depth_validation import read_reference,metrics
from transformers import AutoImageProcessor
O=R/'outputs/major_revision'
FACTORS={'identity':(1.,0.),'dark':(1.5,0.),'blur':(1.,1.5),'dark_blur':(1.5,1.5)}
def frames():
 out=[]
 for x in json.loads((R/'outputs/radial_confirmatory/test_frames.json').read_text()):out.append({**x,'cohort':'c3vd','object':x['scene'],'id':'c3vd_'+x['scene']+'_'+x['frame_id'],'rgb':x['color_path']})
 for x in json.loads((R/'outputs/independent_validation/frames.json').read_text()):out.append({**x,'cohort':'servct','id':'servct_'+x['object']+'_'+x['frame_id']})
 for x in json.loads((R/'outputs/scared_expansion/frames.json').read_text()):
  if x['subject'] in [8,9]:out.append({**x,'cohort':'scared','object':str(x['subject']),'id':'scared_'+x['id']})
 if (O/'drends_frames.json').exists():out+=json.loads((O/'drends_frames.json').read_text())
 return out
def reference(r):
 if r['cohort']=='c3vd':return read_reference(Path(r['depth_path']))
 if r['cohort']=='servct':
  gt=cv2.imread(str(R/r['depth']),cv2.IMREAD_UNCHANGED).astype(float)/256.;occ=cv2.imread(str(R/r['occlusion']));return gt,np.isfinite(gt)&(gt>0)&~np.all(occ==(255,0,0),-1)
 if r['cohort']=='scared':
  xyz=tifffile.imread(r['gt']);gt=xyz[...,2].astype(float);return gt,np.isfinite(xyz).all(-1)&(gt>0)
 gt=tifffile.imread(R/r['gt']).astype(float);v=cv2.imread(str(R/r['mask']),0)>0;return gt,v&np.isfinite(gt)&(gt>0)
def features():
 torch.set_num_threads(4);m=model().cuda().eval();proc=AutoImageProcessor.from_pretrained(M,local_files_only=True,use_fast=False);out=O/'test_features';out.mkdir(exist_ok=True)
 for r in frames():
  rgb=cv2.cvtColor(cv2.imread(str(R/r['rgb'])),cv2.COLOR_BGR2RGB)
  for factor,(gamma,sigma) in FACTORS.items():
   # DRENDS includes naturally varying illumination; artificial perturbations restricted to earlier cohorts.
   if r['cohort']=='drends' and factor!='identity':continue
   p=out/(r['id']+'_'+factor+'.pt')
   if p.exists():continue
   im=np.uint8(np.clip((rgb.astype(np.float32)/255)**gamma*255,0,255))
   if sigma:im=cv2.GaussianBlur(im,(0,0),sigma)
   inputs=proc(images=im,return_tensors='pt',size={'height':224,'width':280},keep_aspect_ratio=False).to('cuda')
   with torch.no_grad():ff=m.backbone.forward_with_filtered_kwargs(inputs.pixel_values).feature_maps
   torch.save({'features':[x.cpu() for x in ff],'native_shape':list(rgb.shape[:2])},p)
  print('FEATURES',r['id'],flush=True)
def inference():
 torch.set_num_threads(4);m=model().cuda().eval();protocol_sha=sha(O/'training_protocol.json')
 for seed in [11,23,47]:
  for regime in ['source_only','adapted']:
   state=torch.load(O/'trained_models'/f'{regime}_{seed}.pt',map_location='cpu',weights_only=True);assert state['protocol_sha256']==protocol_sha;m.neck.load_state_dict(state['neck'],strict=True);m.head.load_state_dict(state['head'],strict=True)
   d=O/'metric_predictions'/f'{regime}_{seed}';d.mkdir(parents=True,exist_ok=True)
   for r in frames():
    for factor in FACTORS:
     if r['cohort']=='drends' and factor!='identity':continue
     p=d/(r['id']+'_'+factor+'.npz')
     if p.exists():continue
     v=torch.load(O/'test_features'/(r['id']+'_'+factor+'.pt'),weights_only=True)
     with torch.no_grad():log=m.head(m.neck([x.cuda() for x in v['features']],16,20),16,20)
     pred=torch.exp(log.clamp(-10,10)).squeeze().cpu().numpy();assert np.isfinite(pred).all();np.savez_compressed(p,prediction=pred)
   print('PREDICTED',regime,seed,flush=True)
def score():
 rows=[]
 for r in frames():
  gt,v=reference(r);assert v.any();hh,ww=gt.shape
  for regime in ['source_only','adapted']:
   for seed in [11,23,47]:
    for factor in FACTORS:
     p=O/'metric_predictions'/f'{regime}_{seed}'/(r['id']+'_'+factor+'.npz')
     if not p.exists():continue
     with np.load(p) as z:pred=cv2.resize(z['prediction'],(ww,hh),interpolation=cv2.INTER_LINEAR).astype(float)
     for bound in [100,300]:
      pp=np.clip(pred,1,bound)
      for support,mask in [('all_reference',v),('reference_1_100',v&(gt>1)&(gt<100))]:
       if not mask.any():continue
       met=metrics(pp,gt,mask);delta=gt[mask]-pp[mask];offset=np.median(delta);ratio=np.median(gt[mask]/np.maximum(pp[mask],1e-6))
       rows.append({'cohort':r['cohort'],'object':r['object'],'frame':r['id'],'regime':regime,'seed':seed,'factor':factor,'bound':bound,'support':support,'pixels':int(mask.sum()),**met,'boundary_fraction':float(((pred<=1)|(pred>=bound))[mask].mean()),'reference_above100':float((gt[mask]>100).mean()),'oracle_offset_mae_mm':float(np.abs(pp[mask]+offset-gt[mask]).mean()),'oracle_scale_mae_mm':float(np.abs(pp[mask]*ratio-gt[mask]).mean()),'oracle_offset_mm':float(offset),'oracle_scale':float(ratio)})
 with (O/'metric_per_frame.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 keys=['mae_mm','rmse_mm','abs_rel','bias_mm','delta1_fraction','boundary_fraction','reference_above100','oracle_offset_mae_mm','oracle_scale_mae_mm'];summ=[];objs=[]
 groupkeys=['cohort','regime','seed','factor','bound','support']
 for group in sorted({tuple(r[k] for k in groupkeys) for r in rows}):
  rr=[r for r in rows if tuple(r[k] for k in groupkeys)==group];oo=[]
  for obj in sorted({r['object'] for r in rr}):
   a=[r for r in rr if r['object']==obj];z={**dict(zip(groupkeys,group)),'object':obj,'frames':len(a),**{k:float(np.mean([v[k] for v in a])) for k in keys}};objs.append(z);oo.append(z)
  summ.append({**dict(zip(groupkeys,group)),'groups':len(oo),**{k:float(np.mean([v[k] for v in oo])) for k in keys}})
 dump(O/'metric_per_object.json',objs);dump(O/'metric_summary.json',summ)
 print(json.dumps([x for x in summ if x['factor']=='identity' and x['bound']==100 and x['support']=='all_reference'],indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('phase',choices=['features','inference','score']);a=p.parse_args();globals()[a.phase]()
