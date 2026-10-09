"""Frozen C3VD calibration transfer to two independent SERV-CT specimens.
No SERV-CT values used for calibration, selection or test-reference alignment.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['HF_HUB_OFFLINE']='1'
os.environ['XFORMERS_DISABLED']='1'
from pathlib import Path,PurePosixPath
import sys,json,csv,hashlib,datetime,argparse,zipfile
import numpy as np
import cv2
R=Path(__file__).resolve().parents[1];sys.path[:0]=[str(R),str(R/'scripts'),str(R/'tests')]
from src.radial_affine_calibration import fit,predict,radius_squared
from run_radial_confirmatory import balanced
from c3vdv2_external_depth_validation import read_reference,metrics
O=R/'outputs/independent_validation';B=R/'outputs/radial_confirmatory';P=R/'outputs/publication_study';D=R/'datasets/external/SERV_CT_official'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4<<20),b''):h.update(b)
 return h.hexdigest()
def dump(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def csvout(p,rows):
 with p.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def sources():
 pp=[Path(__file__),R/'scripts/publication_inference.py',R/'src/radial_affine_calibration.py',R/'scripts/run_radial_confirmatory.py',R/'tests/c3vdv2_external_depth_validation.py']+list((R/'third_party/EndoDAC/models').rglob('*.py'))
 return {str(p.relative_to(R)):sha(p) for p in pp}
def freeze():
 if (O/'frozen_heads.json').exists():raise ValueError('Refusing overwrite')
 if D.exists():raise ValueError('Acquire SERV-CT only after freezing')
 old=json.loads((B/'frozen_heads.json').read_text());assert all(sha(R/n)==h for n,h in old['source_sha256'].items())
 protocol={'utc':now(),'dataset_doi':'10.5522/04/26352199.v1','objects':'Two ex vivo porcine specimens; Experiment_1 and Experiment_2, eight left RGB views each. Different hardware/application from colon phantom development; device/object/domain shifts confounded.','selection':'All 16 left views; CT primary reference; no performance-based replacement or view removal.','fit':'Six original C3VDv2 development scenes, 52 frames, identical deterministic sampled positions; 16384 pixels per head, equal scenes; training-only standardization; global/radial LS for Large and EndoDAC. Small four heads unchanged.','calibration_target':'Inverse axial depth in mm; no target-domain fit or scale alignment','inference':'Frozen DA V2 Small/Large and official EndoDAC; published model-specific preprocessing; no GT values read. EndoDAC strict weights and author RGB preprocessing.','primary_support':'All finite positive CT left depth pixels except blue outside-reference mask; occluded/non-overlap pixels retained for monocular task. No >100mm GT exclusion.','secondary_support':'Official nonoccluded overlap mask (exclude blue,yellow,red,green) and separate 1<GT<100mm support; not substitutes for primary.','depth_decode':'SERV-CT uint16/256 mm, NOT C3VD encoding','prediction_range':'All heads use unchanged [1,100] mm bounds; nonpositive inverse maps to100. Report reference >100mm fraction and boundary activation.','metrics':'Native pixel MAE, per-image RMSE, AbsRel, delta1; average views within specimen then equal specimens. N=2 physical objects, not16 independent objects; descriptive no p-values.','methods':['small_affine_ls','small_affine_ridge','small_radial_ls','small_radial_ridge','large_affine_ls','large_radial_ls','endodac_affine_ls','endodac_radial_ls'],'scope':'Independent-object/device depth-transfer stress test; not in-vivo lesion-size validation or contemporary SOTA ranking','seed':20260928,'foundation_pretraining_overlap':'Not fully auditable; EndoDAC is SCARED-adapted, no SERV-CT training asserted by released training setup; this is not an independent audit of original training','source_sha256':sources()}
 dump(O/'protocol.json',protocol)
 frames=json.loads((P/'development_frames.json').read_text());scenes=sorted({r['scene'] for r in frames});heads={};inputs={}
 for name in ['small','large','endodac']:
  pool={}
  for row in frames:
   path=P/'predictions/development'/name/f"{row['scene']}_{row['frame_id']}.npz";dp=Path(row['depth_path']);inputs[str(path.relative_to(R))]=sha(path);inputs[str(dp.relative_to(R))]=sha(dp)
   gt,v=read_reference(dp)
   with np.load(path) as f:d=f['prediction'].astype(float)
   assert d.shape==gt.shape
   ix=np.flatnonzero(v);ix=ix[np.linspace(0,len(ix)-1,8192,dtype=int)]
   pool[row['scene'],row['frame_id']]={'d':d.ravel()[ix],'z':gt.ravel()[ix].astype(float),'r':radius_squared(d.shape).ravel()[ix]}
  a=balanced(pool,scenes)
  for radial in [False,True]:
   namehead=name+('_radial_ls' if radial else '_affine_ls');h=fit(a['d'],a['r'],a['z'],a['groups'],radial,0.)
   if name=='small':assert np.allclose(h['coefficients_a0_a1_b0_b1'],old['heads'][namehead[6:]]['coefficients_a0_a1_b0_b1'],atol=1e-12,rtol=0)
   heads[namehead]=h
 for m in ['affine_ls','affine_ridge','radial_ls','radial_ridge']:heads['small_'+m]=old['heads'][m]
 for name in ['Small','Large']:
  folder=R/f'datasets/external_models/Depth-Anything-V2-{name}-hf';manifest=json.loads((folder/'download_manifest.json').read_text())
  for x in manifest['files']:
   pp=folder/x['path'];hh=sha(pp);assert hh==x['sha256'];inputs[str(pp.relative_to(R))]=hh
 pp=R/'datasets/external_models/EndoDAC/depth_model.pth';hh=sha(pp);assert hh==json.loads((pp.parent/'manifest.json').read_text())['sha256'];inputs[str(pp.relative_to(R))]=hh
 dump(O/'frozen_heads.json',{'utc':now(),'heads':heads,'source_sha256':sources(),'protocol_sha256':sha(O/'protocol.json'),'input_sha256':inputs,'original_small_heads_sha256':sha(B/'frozen_heads.json'),'test_used_for_fitting':False})
 print('FROZEN',len(heads),'heads before acquisition',flush=True)
def check():
 f=json.loads((O/'frozen_heads.json').read_text());assert f['source_sha256']==sources();assert f['protocol_sha256']==sha(O/'protocol.json');return f

def acquire():
 check();import requests
 D.mkdir(parents=True,exist_ok=True);meta=json.loads((O/'servct_metadata.json').read_text());rec=next(x for x in meta['files'] if x['name']=='SERV-CT.zip');dest=D/'SERV-CT.zip'
 if not dest.exists():
  with requests.get(rec['download_url'],stream=True,timeout=(20,60)) as response:
   response.raise_for_status()
   with dest.with_suffix('.partial').open('wb') as f:
    for chunk in response.iter_content(1<<20):f.write(chunk)
  dest.with_suffix('.partial').rename(dest)
 assert dest.stat().st_size==rec['size'];assert hashlib.md5(dest.read_bytes()).hexdigest()==rec['computed_md5']
 with zipfile.ZipFile(dest) as z:
  for n in z.namelist():
   p=PurePosixPath(n.replace('\\','/'));assert not p.is_absolute() and '..' not in p.parts and ':' not in n
  assert z.testzip() is None;z.extractall(D)
 files=[p for p in D.rglob('*') if p.is_file() and p!=dest]
 dump(O/'acquisition.json',{'utc':now(),'source_doi':meta['doi'],'archive':rec,'archive_sha256':sha(dest),'official_md5_verified':True,'members':{str(p.relative_to(R)):sha(p) for p in files}})
 print('ACQUIRED',len(files),'members',flush=True)

def frames():
 result=[]
 for obj in ['Experiment_1','Experiment_2']:
  dirs=list(D.rglob(obj));assert len(dirs)==1;folder=dirs[0]
  rgb=sorted((folder/'Left_rectified').glob('*.png'));assert len(rgb)==8
  for p in rgb:
   cal=folder/'Rectified_calibration'/(p.stem+'.json')
   result.append({'object':obj,'frame_id':p.stem,'rgb':str(p.relative_to(R)),'depth':str((folder/'Ground_truth_CT/DepthL'/p.name).relative_to(R)),'occlusion':str((folder/'Ground_truth_CT/OcclusionL'/p.name).relative_to(R)),'calibration':str(cal.relative_to(R))})
 return result

def inference():
 check();from publication_inference import EndoDACPredictor
 import torch
 from transformers import AutoImageProcessor,AutoModelForDepthEstimation
 torch.manual_seed(20260928);torch.set_num_threads(4);torch.backends.cudnn.benchmark=False
 ff=frames();dump(O/'frames.json',ff);run=[]
 for name in ['small','large','endodac']:
  dest=O/'predictions'/name;dest.mkdir(parents=True,exist_ok=True)
  if name=='endodac':model=EndoDACPredictor();count=model.parameters
  else:
   folder=R/f'datasets/external_models/Depth-Anything-V2-{name.title()}-hf';processor=AutoImageProcessor.from_pretrained(folder,local_files_only=True,use_fast=False);model=AutoModelForDepthEstimation.from_pretrained(folder,local_files_only=True).cuda().eval();count=sum(p.numel() for p in model.parameters())
  for row in ff:
   pp=dest/(row['object']+'_'+row['frame_id']+'.npz')
   if pp.exists():continue
   rgb=cv2.cvtColor(cv2.imread(str(R/row['rgb'])),cv2.COLOR_BGR2RGB)
   if name=='endodac':d=model.predict(rgb)
   else:
    with torch.inference_mode():
     raw=model(**processor(images=rgb,return_tensors='pt').to('cuda:0')).predicted_depth;d=torch.nn.functional.interpolate(raw.unsqueeze(1),size=rgb.shape[:2],mode='bilinear',align_corners=False).squeeze().float().cpu().numpy()
   assert np.isfinite(d).all();np.savez_compressed(pp,prediction=d)
  run.append({'model':name,'parameters':count,'frames':len(ff)});del model;torch.cuda.empty_cache();print('INFERRED',name,flush=True)
 dump(O/'inference.json',{'utc':now(),'runs':run,'test_reference_values_read':False})

def evaluate():
 f=check();assert not (O/'summary.json').exists(),'Refusing repeat scoring'
 dump(O/'evaluation_start.json',{'utc':now(),'frozen_heads_sha256':sha(O/'frozen_heads.json'),'protocol_sha256':sha(O/'protocol.json')})
 rows=[];audit=[]
 for row in frames():
  raw=cv2.imread(str(R/row['depth']),cv2.IMREAD_UNCHANGED);assert raw.dtype==np.uint16;gt=raw.astype(float)/256.
  occ=cv2.imread(str(R/row['occlusion']));blue=np.all(occ==(255,0,0),axis=-1);valid=(gt>0)&np.isfinite(gt)&~blue
  noc=valid.copy()
  for c in [(0,255,255),(0,0,255),(0,255,0)]:noc &= ~np.all(occ==c,axis=-1)
  cal=json.loads((R/row['calibration']).read_text())
  audit.append({'object':row['object'],'frame_id':row['frame_id'],'shape':list(gt.shape),'valid_pixels':int(valid.sum()),'reference_gt100_fraction':float((gt[valid]>100).mean()),'reference_min_mm':float(gt[valid].min()),'reference_max_mm':float(gt[valid].max()),'calibration_keys':list(cal)})
  for m,h in f['heads'].items():
   name=m.split('_')[0]
   with np.load(O/'predictions'/name/(row['object']+'_'+row['frame_id']+'.npz')) as z:d=z['prediction'].astype(float)
   assert d.shape==gt.shape;p=predict(h,d)
   for support,mask in [('all_reference',valid),('nonoccluded',noc),('reference_1_100',valid&(gt>1)&(gt<100))]:
    if not mask.any():continue
    rows.append({'object':row['object'],'frame':row['frame_id'],'method':m,'support':support,'pixels':int(mask.sum()),**metrics(p,gt,mask),'bound_fraction':float(((p<=1)|(p>=100))[mask].mean())})
 keys=['mae_mm','rmse_mm','abs_rel','bias_mm','delta1_fraction','bound_fraction'];objects=[];summary=[]
 for support in ['all_reference','nonoccluded','reference_1_100']:
  for m in f['heads']:
   for obj in ['Experiment_1','Experiment_2']:
    vals=[x for x in rows if (x['method'],x['support'],x['object'])==(m,support,obj)]
    if vals:objects.append({'method':m,'support':support,'object':obj,'frames':len(vals),**{k:float(np.mean([v[k] for v in vals])) for k in keys}})
   vals=[x for x in objects if (x['method'],x['support'])==(m,support)]
   summary.append({'method':m,'support':support,'objects':len(vals),**{k:float(np.mean([v[k] for v in vals])) for k in keys}})
 csvout(O/'per_frame.csv',rows);csvout(O/'per_object.csv',objects);dump(O/'summary.json',summary);dump(O/'reference_audit.json',audit)
 print(json.dumps([x for x in summary if x['support']=='all_reference'],indent=2),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=['freeze','acquire','inference','evaluate']);args=p.parse_args();globals()[args.phase]()
