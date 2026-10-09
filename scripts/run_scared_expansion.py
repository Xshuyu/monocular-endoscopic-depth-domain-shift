"""Frozen monocular transfer plus explicitly unmatched-input stereo references."""
import os
os.environ['HF_HUB_OFFLINE']='1';os.environ['XFORMERS_DISABLED']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import sys,json,csv,argparse,hashlib,datetime,types
import cv2,numpy as np,tifffile
R=Path(__file__).resolve().parents[1];O=R/'outputs/scared_expansion';D=R/'datasets/external/SCARED_official/measured_keyframes';sys.path[:0]=[str(R),str(R/'scripts'),str(R/'tests')]
from src.radial_affine_calibration import predict
from c3vdv2_external_depth_validation import metrics

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4<<20),b''):h.update(b)
 return h.hexdigest()
def dump(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def csvout(p,rows):
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def heads():
 protocol=json.loads((O/'protocol.json').read_text());p=R/'outputs/independent_validation/frozen_heads.json';assert sha(p)==protocol['heads_sha256'];data=json.loads(p.read_text())
 for k,v in data['source_sha256'].items():assert sha(R/k)==v,k
 return {k:v for k,v in data['heads'].items() if not k.startswith('endodac')}
def frames():
 acquired=json.loads((O/'acquisition.json').read_text());out=[]
 for n in [1,2,3,6,7,8,9]:
  pp=sorted((D/f'dataset_{n}').glob('keyframe_*/Left_Image.png'));assert len(pp)==5,(n,len(pp))
  for p in pp:
   out.append({'subject':n,'keyframe':p.parent.name,'id':f'dataset_{n}_{p.parent.name}','rgb':str(p),'right':str(p.parent/'Right_Image.png'),'gt':str(p.parent/'left_depth_map.tiff'),'cal':str(p.parent/'endoscope_calibration.yaml')})
 for a in acquired:assert sha(R/a['local'])==a['sha256']
 return out

def mono():
 heads();ff=frames();dump(O/'frames.json',ff)
 import torch
 from transformers import AutoImageProcessor,AutoModelForDepthEstimation
 torch.manual_seed(20261002);torch.set_num_threads(4);torch.backends.cudnn.benchmark=False
 runs=[]
 for name in ['small','large']:
  dest=O/'predictions'/name;dest.mkdir(parents=True,exist_ok=True)
  modeldir=R/f'datasets/external_models/Depth-Anything-V2-{name.title()}-hf'
  for rec in json.loads((modeldir/'download_manifest.json').read_text())['files']:assert sha(modeldir/rec['path'])==rec['sha256']
  processor=AutoImageProcessor.from_pretrained(modeldir,local_files_only=True,use_fast=False);model=AutoModelForDepthEstimation.from_pretrained(modeldir,local_files_only=True).cuda().eval()
  for row in ff:
   destfile=dest/(row['id']+'.npz')
   if destfile.exists():continue
   old=R/'outputs/scared_validation/predictions'/name/(row['id']+'.npz')
   if old.exists():
    import shutil;shutil.copy2(old,destfile);continue
   rgb=cv2.cvtColor(cv2.imread(row['rgb']),cv2.COLOR_BGR2RGB)
   with torch.inference_mode():
    raw=model(**processor(images=rgb,return_tensors='pt').to('cuda')).predicted_depth
    prediction=torch.nn.functional.interpolate(raw.unsqueeze(1),size=rgb.shape[:2],mode='bilinear',align_corners=False).squeeze().float().cpu().numpy()
   assert np.isfinite(prediction).all();np.savez_compressed(destfile,prediction=prediction);print('MONO',name,row['id'],flush=True)
  runs.append({'model':name,'parameters':sum(p.numel() for p in model.parameters()),'frames':len(ff),'new_forward_passes':25,'reused_previous':10});del model;torch.cuda.empty_cache()
 dump(O/'mono_run.json',{'runs':runs,'script_sha256':sha(Path(__file__)),'gt_values_read':False,'protocol_sha256':sha(O/'protocol.json')})

def native_depth(disparity,K,dist,R1,P1,Q):
 h,w=disparity.shape;yy,xx=np.indices((h,w));native=np.stack([xx,yy],axis=-1).astype(np.float32)
 rect=cv2.undistortPoints(native.reshape(-1,1,2),K,dist,R=R1,P=P1).reshape(h,w,2)
 d=cv2.remap(disparity.astype(np.float32),rect[...,0],rect[...,1],cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=-1).astype(float)
 homogeneous=np.stack([rect[...,0],rect[...,1],d,np.ones_like(d)],axis=-1)@Q.T
 denom=homogeneous[...,3];xyz=np.divide(homogeneous[...,:3],denom[...,None],out=np.full_like(homogeneous[...,:3],np.nan),where=np.abs(denom[...,None])>1e-12)
 z=xyz@R1[:,2]
 valid=(d>0)&np.isfinite(z)&(z>0)&(rect[...,0]>=0)&(rect[...,0]<w-1)&(rect[...,1]>=0)&(rect[...,1]<h-1)&(rect[...,0]-d>=0)&(rect[...,0]-d<w-1)
 return np.where(valid,np.clip(z,1,100),100).astype(np.float32),valid

def stereo():
 ff=frames();heads()
 import torch
 sys.path[:0]=[str(R/'third_party/RAFT-Stereo'),str(R/'third_party/raft_runtime_deps')]
 from core.raft_stereo import RAFTStereo
 from core.utils.utils import InputPadder
 args=types.SimpleNamespace(hidden_dims=[128]*3,corr_implementation='reg',shared_backbone=False,corr_levels=4,corr_radius=4,n_downsample=2,context_norm='batch',slow_fast_gru=False,n_gru_layers=3,mixed_precision=False)
 torch.manual_seed(20261002);torch.set_num_threads(4)
 model=RAFTStereo(args);mp=R/'datasets/external_models/RAFT-Stereo';meta=json.loads((mp/'manifest.json').read_text());assert sha(mp/meta['checkpoint'])==meta['sha256']
 state=torch.load(mp/meta['checkpoint'],map_location='cpu',weights_only=True);model.load_state_dict({k.removeprefix('module.'):v for k,v in state.items()},strict=True);model=model.cuda().eval()
 audit=[]
 for row in ff:
  left=cv2.imread(row['rgb']);right=cv2.imread(row['right']);assert left.shape==right.shape;h,w=left.shape[:2]
  fs=cv2.FileStorage(row['cal'],cv2.FILE_STORAGE_READ);K1,K2,D1,D2,rot,T=[fs.getNode(k).mat().astype(float) for k in ['M1','M2','D1','D2','R','T']];fs.release()
  R1,R2,P1,P2,Q,_,_=cv2.stereoRectify(K1,D1,K2,D2,(w,h),rot,T.reshape(3,1),flags=cv2.CALIB_ZERO_DISPARITY,alpha=0)
  map1=cv2.initUndistortRectifyMap(K1,D1,R1,P1,(w,h),cv2.CV_32FC1);map2=cv2.initUndistortRectifyMap(K2,D2,R2,P2,(w,h),cv2.CV_32FC1)
  l=cv2.remap(left,*map1,cv2.INTER_LINEAR);r=cv2.remap(right,*map2,cv2.INTER_LINEAR)
  for method in ['sgbm','raft_sceneflow']:
   dest=O/'predictions'/method;dest.mkdir(parents=True,exist_ok=True);file=dest/(row['id']+'.npz')
   if file.exists():continue
   if method=='sgbm':
    matcher=cv2.StereoSGBM_create(minDisparity=0,numDisparities=256,blockSize=5,P1=8*25,P2=32*25,disp12MaxDiff=1,uniquenessRatio=10,speckleWindowSize=100,speckleRange=2,preFilterCap=31,mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
    disp=matcher.compute(cv2.cvtColor(l,cv2.COLOR_BGR2GRAY),cv2.cvtColor(r,cv2.COLOR_BGR2GRAY)).astype(np.float32)/16
   else:
    tensors=[torch.from_numpy(cv2.cvtColor(cv2.resize(im,(w//2,h//2),interpolation=cv2.INTER_AREA),cv2.COLOR_BGR2RGB)).permute(2,0,1).float().unsqueeze(0).cuda() for im in [l,r]]
    padder=InputPadder(tensors[0].shape,divis_by=32);tl,tr=padder.pad(*tensors)
    with torch.inference_mode():_,flow=model(tl,tr,iters=32,test_mode=True)
    small=-padder.unpad(flow).squeeze().float().cpu().numpy();disp=cv2.resize(small,(w,h),interpolation=cv2.INTER_LINEAR)*(w/(w//2))
   depth,valid=native_depth(disp,K1,D1,R1,P1,Q);np.savez_compressed(file,prediction=depth,valid=valid)
   print('STEREO',method,row['id'],float(valid.mean()),flush=True)
  audit.append({'id':row['id'],'baseline_mm':float(np.linalg.norm(T)),'rectified_fx':float(P1[0,0]),'depth_transform':'Q then inverse R1 to original-left axial Z; unrectified native pixel sampling','native_shape':[h,w]})
 dump(O/'stereo_run.json',{'frames':len(ff),'raft_parameters':sum(p.numel() for p in model.parameters()),'checkpoint':meta,'gt_values_read':False,'audit':audit,'script_sha256':sha(Path(__file__))})

def score():
 assert not (O/'summary.json').exists(),'Refusing overwrite';hh=heads();ff=frames();rows=[];audit=[]
 dump(O/'scoring_start.json',{'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'protocol_sha256':sha(O/'protocol.json'),'script_sha256':sha(Path(__file__))})
 for row in ff:
  xyz=tifffile.imread(row['gt']);gt=xyz[...,2].astype(float);valid=np.isfinite(xyz).all(-1)&(gt>0)
  predictions={};coverage={}
  for m,h in hh.items():
   with np.load(O/'predictions'/m.split('_')[0]/(row['id']+'.npz')) as z:d=z['prediction']
   predictions[m]=predict(h,d);coverage[m]=np.ones_like(valid)
  for m in ['sgbm','raft_sceneflow']:
   with np.load(O/'predictions'/m/(row['id']+'.npz')) as z:predictions[m]=z['prediction'].astype(float);coverage[m]=z['valid']
  masks={'all_reference':valid,'reference_1_100':valid&(gt>1)&(gt<100),'common_stereo_valid':valid&coverage['sgbm']&coverage['raft_sceneflow']}
  for support,mask in masks.items():
   assert mask.any()
   for method,p in predictions.items():
    assert np.isfinite(p).all() and p.shape==gt.shape
    rows.append({'subject':row['subject'],'frame':row['keyframe'],'method':method,'support':support,'pixels':int(mask.sum()),**metrics(p,gt,mask),'coverage':float(coverage[method][mask].mean())})
  audit.append({'id':row['id'],'z_min_mm':float(gt[valid].min()),'z_max_mm':float(gt[valid].max()),'gt100_fraction':float((gt[valid]>100).mean()),'valid_fraction':float(valid.mean())})
 keys=['mae_mm','rmse_mm','abs_rel','bias_mm','delta1_fraction','coverage'];objs=[];summ=[]
 for support in masks:
  for method in predictions:
   for n in [1,2,3,6,7,8,9]:
    r=[x for x in rows if (x['subject'],x['method'],x['support'])==(n,method,support)]
    objs.append({'subject':n,'method':method,'support':support,'frames':len(r),**{k:float(np.mean([x[k] for x in r])) for k in keys}})
   for cohort,subjects in [('new_five',[1,2,3,6,7]),('previous_two',[8,9]),('cumulative_seven',[1,2,3,6,7,8,9])]:
    r=[x for x in objs if x['subject'] in subjects and x['method']==method and x['support']==support]
    summ.append({'cohort':cohort,'method':method,'support':support,'subjects':len(r),**{k:float(np.mean([x[k] for x in r])) for k in keys}})
 csvout(O/'per_frame.csv',rows);csvout(O/'per_subject.csv',objs);dump(O/'summary.json',summ);dump(O/'reference_audit.json',audit)
 rng=np.random.default_rng(20261002);paired=[]
 for cohort,subjects in [('new_five',[1,2,3,6,7]),('cumulative_seven',[1,2,3,6,7,8,9])]:
  diffs=[]
  for n in subjects:
   vals={x['method']:x['mae_mm'] for x in objs if x['subject']==n and x['support']=='all_reference'};diffs.append(vals['small_radial_ridge']-vals['small_affine_ls'])
  sample=rng.choice(diffs,(10000,len(diffs)),replace=True).mean(1)
  paired.append({'cohort':cohort,'subject_differences_mm':diffs,'mean_difference_mm':float(np.mean(diffs)),'improved_subjects':int(np.sum(np.array(diffs)<0)),'exploratory_bootstrap_95ci':np.quantile(sample,[.025,.975]).tolist()})
 dump(O/'paired.json',paired);print(json.dumps([x for x in summ if x['support']=='all_reference'],indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('phase',choices=['mono','stereo','score']);a=p.parse_args();globals()[a.phase]()
