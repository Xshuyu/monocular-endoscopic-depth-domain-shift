"""Frozen C3VD calibration transfer to original SCARED measured keyframes."""
import os
os.environ['HF_HUB_OFFLINE']='1';os.environ['XFORMERS_DISABLED']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import sys,json,csv,hashlib,datetime,argparse
import numpy as np
import cv2,tifffile
R=Path(__file__).resolve().parents[1];sys.path[:0]=[str(R),str(R/'scripts'),str(R/'tests')]
from src.radial_affine_calibration import predict
from c3vdv2_external_depth_validation import metrics
O=R/'outputs/scared_validation';D=R/'datasets/external/SCARED_official/measured_keyframes'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4<<20),b''):h.update(b)
 return h.hexdigest()
def dump(p,obj):p.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
def csvout(p,rows):
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def check():
 protocol=json.loads((O/'protocol.json').read_text());h=R/protocol['heads_source'];assert sha(h)==protocol['heads_sha256']
 heads=json.loads(h.read_text())
 for p,digest in heads['source_sha256'].items():assert sha(R/p)==digest,p
 return heads['heads']
def frames():
 rows=[]
 for subject in [8,9]:
  for k in range(5):
   p=D/f'dataset_{subject}/keyframe_{k}'
   row={'object':f'dataset_{subject}','frame':f'keyframe_{k}','rgb':str((p/'Left_Image.png').relative_to(R)),'depth':str((p/'left_depth_map.tiff').relative_to(R)),'calibration':str((p/'endoscope_calibration.yaml').relative_to(R))}
   for key in ['rgb','depth','calibration']:assert (R/row[key]).is_file(),row[key]
   rows.append(row)
 return rows

def inference():
 check();ff=frames()
 acquired={r['local']:r['sha256'] for r in json.loads((O/'keyframe_acquisition.json').read_text())}
 for row in ff:
  for key in ['rgb','depth','calibration']:assert sha(R/row[key])==acquired[row[key]]
  row['rgb_sha256']=sha(R/row['rgb'])
 assert len(set(r['rgb_sha256'] for r in ff))==10
 old=json.loads((R/'outputs/independent_validation/frames.json').read_text());oldhash={sha(R/x['rgb']) for x in old}
 olddev=json.loads((R/'outputs/publication_study/development_frames.json').read_text());oldhash|={sha(x['color_path']) for x in olddev}
 assert not oldhash & {r['rgb_sha256'] for r in ff}
 dump(O/'frames.json',ff)
 from publication_inference import EndoDACPredictor
 from transformers import AutoImageProcessor,AutoModelForDepthEstimation
 import torch
 torch.manual_seed(20261001);torch.set_num_threads(4);torch.backends.cudnn.benchmark=False
 runs=[];weights={}
 for name in ['small','large','endodac']:
  dest=O/'predictions'/name;dest.mkdir(parents=True,exist_ok=True)
  if name=='endodac':
   model=EndoDACPredictor();count=model.parameters;weights[name]=sha(R/'datasets/external_models/EndoDAC/depth_model.pth')
  else:
   folder=R/f'datasets/external_models/Depth-Anything-V2-{name.title()}-hf'
   for rec in json.loads((folder/'download_manifest.json').read_text())['files']:
    assert sha(folder/rec['path'])==rec['sha256'];weights[str((folder/rec['path']).relative_to(R))]=rec['sha256']
   processor=AutoImageProcessor.from_pretrained(folder,local_files_only=True,use_fast=False)
   model=AutoModelForDepthEstimation.from_pretrained(folder,local_files_only=True).cuda().eval();count=sum(p.numel() for p in model.parameters())
  for row in ff:
   p=dest/f"{row['object']}_{row['frame']}.npz"
   if p.exists():continue
   rgb=cv2.cvtColor(cv2.imread(str(R/row['rgb'])),cv2.COLOR_BGR2RGB)
   if name=='endodac':d=model.predict(rgb)
   else:
    with torch.inference_mode():
     raw=model(**processor(images=rgb,return_tensors='pt').to('cuda:0')).predicted_depth
     d=torch.nn.functional.interpolate(raw.unsqueeze(1),size=rgb.shape[:2],mode='bilinear',align_corners=False).squeeze().float().cpu().numpy()
   assert d.shape==rgb.shape[:2] and np.isfinite(d).all();np.savez_compressed(p,prediction=d)
   print('INFERRED',name,row['object'],row['frame'],flush=True)
  runs.append({'model':name,'parameters':count,'frames':len(ff),'independence':'published training split contains target subjects' if name=='endodac' else 'no task-specific SCARED fit; foundation pretraining not fully auditable'})
  del model;torch.cuda.empty_cache()
 dump(O/'inference.json',{'runs':runs,'weights_sha256':weights,'reference_values_used_for_inference':False,'protocol_sha256':sha(O/'protocol.json'),'script_sha256':sha(__file__),'torch':torch.__version__,'cuda':torch.version.cuda})

def evaluate():
 heads=check();assert not (O/'summary.json').exists(),'Refusing repeat scoring'
 dump(O/'evaluation_start.json',{'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'protocol_sha256':sha(O/'protocol.json'),'overlap_amendment_sha256':sha(O/'preinference_overlap_amendment.json'),'script_sha256':sha(__file__)})
 rows=[];audits=[]
 for row in frames():
  xyz=tifffile.imread(R/row['depth']);rgb=cv2.imread(str(R/row['rgb']));assert xyz.shape==rgb.shape and xyz.shape[-1]==3
  gt=xyz[...,2].astype(float);valid=np.isfinite(xyz).all(-1)&(gt>0);assert valid.any()
  c=cv2.FileStorage(str(R/row['calibration']),cv2.FILE_STORAGE_READ);K=c.getNode('M1').mat();dist=c.getNode('D1').mat();T=c.getNode('T').mat();c.release()
  assert np.isfinite(K).all() and K[0,0]>0 and K[1,1]>0
  ix=np.flatnonzero(valid)[::100];pts=xyz.reshape(-1,3)[ix].astype(float)
  projected,_=cv2.projectPoints(pts,np.zeros(3),np.zeros(3),K,dist)
  target=np.stack([ix%gt.shape[1],ix//gt.shape[1]],axis=-1);residual=np.linalg.norm(projected.reshape(-1,2)-target,axis=-1)
  audits.append({'object':row['object'],'frame':row['frame'],'shape':list(gt.shape),'valid_pixels':int(valid.sum()),'valid_fraction':float(valid.mean()),'z_min_mm':float(gt[valid].min()),'z_max_mm':float(gt[valid].max()),'z_gt100_fraction':float((gt[valid]>100).mean()),'fx':float(K[0,0]),'fy':float(K[1,1]),'baseline_mm':float(np.linalg.norm(T)),'xyz_projection_residual_p95_px':float(np.percentile(residual,95)),'projection_check_scope':'internal consistency only, not independent calibration accuracy'})
  for method,head in heads.items():
   model=method.split('_')[0]
   with np.load(O/'predictions'/model/f"{row['object']}_{row['frame']}.npz") as data:d=data['prediction'].astype(float)
   assert d.shape==gt.shape;p=predict(head,d)
   for support,mask in [('all_reference',valid),('reference_1_100',valid&(gt>1)&(gt<100))]:
    assert mask.any()
    rows.append({'object':row['object'],'frame':row['frame'],'method':method,'support':support,'pixels':int(mask.sum()),**metrics(p,gt,mask),'bound_fraction':float(((p<=1)|(p>=100))[mask].mean())})
 keys=['mae_mm','rmse_mm','abs_rel','bias_mm','delta1_fraction','bound_fraction'];objects=[];summary=[]
 for support in ['all_reference','reference_1_100']:
  for method in heads:
   for obj in ['dataset_8','dataset_9']:
    vals=[v for v in rows if (v['support'],v['method'],v['object'])==(support,method,obj)]
    objects.append({'object':obj,'method':method,'support':support,'frames':len(vals),**{k:float(np.mean([v[k] for v in vals])) for k in keys}})
   vals=[v for v in objects if (v['support'],v['method'])==(support,method)]
   summary.append({'method':method,'support':support,'objects':len(vals),'training_overlap_caveat':method.startswith('endodac'),**{k:float(np.mean([v[k] for v in vals])) for k in keys}})
 csvout(O/'per_frame.csv',rows);csvout(O/'per_object.csv',objects);dump(O/'reference_audit.json',audits);dump(O/'summary.json',summary)
 print(json.dumps([x for x in summary if x['support']=='all_reference'],indent=2),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=['inference','evaluate']);a=p.parse_args();globals()[a.phase]()
