"""External physical-phantom depth pilot with calibration-only fitted scales.

This is not lesion-diameter validation. Test anatomy c2 is held out from fitted
calibration (c1); frames within a video are not independent experimental units.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,os,platform,shutil,sys,time
from pathlib import Path
from collections import defaultdict
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT.parent))
os.environ['HF_HUB_ENABLE_DOWNLOAD']='0';os.environ['HF_HUB_OFFLINE']='1';os.environ['TOKENIZERS_PARALLELISM']='false'
import cv2
import numpy as np
import torch
from transformers import AutoImageProcessor,AutoModelForDepthEstimation
from lesion_3d_measurement.src.depth_anything_backbone import DepthAnythingV2Backbone

METHODS=('constant_30mm','constant_calibration_median','photometric_raw','photometric_calibration_scale','depth_anything_v2_calibration_inverse_affine')

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
 return h.hexdigest()
def dump(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def csv_write(p,rows):
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def read_reference(p):
 d=cv2.imread(str(p),cv2.IMREAD_UNCHANGED)
 if d is None or d.dtype!=np.uint16 or d.ndim!=2:raise ValueError(f'Expected original uint16 axial depth, got {None if d is None else (d.dtype,d.shape)}: {p}')
 valid=(d>0)&(d<65535)
 return d.astype(np.float32)*(100.0/65535.0),valid

def metrics(pred,gt,valid):
 p=pred[valid].astype(float);g=gt[valid].astype(float);e=p-g
 return {'mae_mm':float(np.abs(e).mean()),'rmse_mm':float(np.sqrt(np.mean(e*e))),
         'abs_rel':float(np.mean(np.abs(e)/g)),'bias_mm':float(e.mean()),
         'delta1_fraction':float(np.mean(np.maximum(p/g,g/p)<1.25))}

def fit_calibration(photo,relative,reference):
 # All arrays contain only calibration-frame pixels; never test labels.
 q=np.asarray(photo,float);d=np.asarray(relative,float);g=np.asarray(reference,float)
 design=np.column_stack([d,np.ones(len(d))]);a,b=np.linalg.lstsq(design,1/g,rcond=None)[0]
 return {'constant_median_mm':float(np.median(g)),
         'photometric_scale':float(np.dot(q,g)/np.dot(q,q)),
         'depth_anything_inverse_affine_a':float(a),'depth_anything_inverse_affine_b':float(b)}

def predictors(photo,relative,fit):
 denominator=fit['depth_anything_inverse_affine_a']*relative+fit['depth_anything_inverse_affine_b']
 # The depth reference is censored at 100 mm. Apply the same [1,100] interval
 # to every method and report clipping; nonpositive inverse depth becomes 100.
 inv=np.divide(1.,denominator,out=np.full_like(relative,100.),where=denominator>0)
 return {'constant_30mm':np.full_like(photo,30.),
         'constant_calibration_median':np.full_like(photo,fit['constant_median_mm']),
         'photometric_raw':photo,
         'photometric_calibration_scale':photo*fit['photometric_scale'],
         'depth_anything_v2_calibration_inverse_affine':inv},denominator<=0

def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--data',type=Path,default=ROOT/'datasets/external/C3VDv2_native_subset')
 ap.add_argument('--model',type=Path,default=ROOT/'datasets/external_models/Depth-Anything-V2-Small-hf')
 ap.add_argument('--output',type=Path,default=ROOT/'outputs/c3vdv2_external_validation')
 ap.add_argument('--device',default='cuda:0' if torch.cuda.is_available() else 'cpu')
 args=ap.parse_args();data=args.data.resolve();model_dir=args.model.resolve();out=args.output.resolve()
 manifest=json.loads((data/'download_manifest.json').read_text());model_manifest=json.loads((model_dir/'download_manifest.json').read_text())
 frames=manifest['frames'];out.mkdir(parents=True,exist_ok=False);(out/'raw_predictions').mkdir()
 srcs=[Path(__file__).resolve(),ROOT/'src/depth_anything_backbone.py',ROOT/'src/depth_estimator.py']
 source_hash={str(p.relative_to(ROOT)):sha(p) for p in srcs}
 protocol={'source_sha256':source_hash,'data_manifest_sha256':sha(data/'download_manifest.json'),
           'model_manifest_sha256':sha(model_dir/'download_manifest.json'),'methods':METHODS,
           'primary_endpoint':'per-video mean axial depth MAE in mm, against registered phantom-reference depth',
           'reference_decode':'uint16 * 100/65535, axial Z in mm; exclude 0 and 65535',
           'prediction_clip_mm':[1,100],'reference_resampling':'none; predictions resized to native reference dimensions',
           'fit':'fixed scalar for photometric; affine inverse-depth for pretrained baseline; calibration only',
           'fit_pixels':'8192 uniform valid pixel indices per calibration frame, equal frame count',
           'test_fit':'prohibited; no per-test-frame median scaling or GT alignment',
           'selection':manifest['selection'],'frames':frames,'device':args.device,
           'inference':'eval mode; no training; raw predicted_depth tensor, not display-normalized pipeline image',
           'limits':['physical silicone phantom, not patients','one held-out colon anatomy; six test segments are not six patients',
                     'test sequences are debris conditions, so anatomy and acquisition differences are not isolated',
                     'mirror bytes not verified against inaccessible primary Dataverse',
                     'no lesion masks or physical lesion-size labels in this subset','pretrained-corpus overlap not independently established',
                     'classical methods and pretrained network have different training/resource histories; this is an external comparator, not a parameter-matched ablation'],
           'bootstrap':'10000 draws over test sequence means; descriptive conditional on this one held-out anatomy; no patient-population CI',
           'command':[sys.executable,*sys.argv]}
 dump(out/'protocol.json',protocol)
 for row in manifest['files']:
  if sha(data/row['path'])!=row['sha256']:raise ValueError('Input checksum mismatch '+row['path'])
 for row in model_manifest['files']:
  if sha(model_dir/row['path'])!=row['sha256']:raise ValueError('Model checksum mismatch')
 audit=[]
 for row in frames:
  gt,valid=read_reference(data/row['depth_path']);im=cv2.imread(str(data/row['color_path']))
  if im is None or im.shape[:2]!=gt.shape or gt.shape!=(1080,1350):raise ValueError('Unexpected native image/depth dimensions')
  if valid.sum()<8192:raise ValueError('Insufficient reference support')
  audit.append({**row,'height':gt.shape[0],'width':gt.shape[1],'valid_pixels':int(valid.sum()),'valid_fraction':float(valid.mean())})
 dump(out/'reference_audit.json',audit)
 # Verify disjoint images, not just names, across calibration and test.
 split_hashes=defaultdict(set)
 for row in frames:split_hashes[row['split']].add(sha(data/row['color_path']))
 if split_hashes['calibration'] & split_hashes['test']:raise ValueError('Cross-split duplicate image')
 torch.manual_seed(20260926);torch.set_num_threads(4)
 torch.backends.cudnn.benchmark=False
 processor=AutoImageProcessor.from_pretrained(model_dir,local_files_only=True,use_fast=False)
 model=AutoModelForDepthEstimation.from_pretrained(model_dir,local_files_only=True).to(args.device).eval()
 photo_model=DepthAnythingV2Backbone(device='cpu');assert photo_model.pipe is None
 start=time.monotonic();q_all=[];d_all=[];g_all=[]
 for i,row in enumerate(frames):
  bgr=cv2.imread(str(data/row['color_path']));rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
  t=time.monotonic();photo=photo_model.predict(rgb);photo_seconds=time.monotonic()-t
  inputs=processor(images=rgb,return_tensors='pt').to(args.device)
  if args.device.startswith('cuda'):torch.cuda.synchronize()
  t=time.monotonic()
  with torch.inference_mode():
   raw=model(**inputs).predicted_depth
   relative=torch.nn.functional.interpolate(raw.unsqueeze(1),size=rgb.shape[:2],mode='bilinear',align_corners=False).squeeze().float().cpu().numpy()
  if args.device.startswith('cuda'):torch.cuda.synchronize()
  network_seconds=time.monotonic()-t
  if not np.isfinite(relative).all():raise ValueError('Nonfinite learned depth')
  np.savez_compressed(out/'raw_predictions'/f"{row['scene']}_{row['frame_id']}.npz",photometric=photo,relative_depth_anything=relative,photometric_seconds=photo_seconds,network_seconds=network_seconds)
  if row['split']=='calibration':
   gt,valid=read_reference(data/row['depth_path']);indices=np.flatnonzero(valid)
   idx=indices[np.linspace(0,len(indices)-1,8192,dtype=int)]
   q_all.append(photo.ravel()[idx]);d_all.append(relative.ravel()[idx]);g_all.append(gt.ravel()[idx])
  if (i+1)%12==0:print(f'Inferred {i+1}/{len(frames)} frames, {time.monotonic()-start:.1f}s',flush=True)
 fit=fit_calibration(np.concatenate(q_all),np.concatenate(d_all),np.concatenate(g_all))
 fit['n_calibration_frames']=sum(r['split']=='calibration' for r in frames)
 fit['n_calibration_pixels']=sum(len(x) for x in q_all)
 dump(out/'calibration.json',fit)
 rows=[]
 for row in frames:
  cached=np.load(out/'raw_predictions'/f"{row['scene']}_{row['frame_id']}.npz")
  gt,valid=read_reference(data/row['depth_path'])
  candidates,invalid_inv=predictors(cached['photometric'],cached['relative_depth_anything'],fit)
  for method,pred in candidates.items():
   clipped=(pred<1)|(pred>100)
   rows.append({**row,'method':method,**metrics(np.clip(pred,1,100),gt,valid),
                'prediction_clipped_fraction':float(clipped[valid].mean()),
                'nonpositive_inverse_fraction':float(invalid_inv[valid].mean()) if method.startswith('depth_anything') else 0.})
 csv_write(out/'per_frame.csv',rows)
 per_video=[]
 for scene in sorted({r['scene'] for r in rows}):
  for method in METHODS:
   group=[r for r in rows if r['scene']==scene and r['method']==method]
   per_video.append({'scene':scene,'split':group[0]['split'],'method':method,'n_frames':len(group),
     **{k:float(np.mean([r[k] for r in group])) for k in ('mae_mm','rmse_mm','abs_rel','bias_mm','delta1_fraction','prediction_clipped_fraction','nonpositive_inverse_fraction')}})
 csv_write(out/'per_video.csv',per_video)
 summary=[]
 for split in ('calibration','validation','test'):
  for method in METHODS:
   v=[r for r in per_video if r['split']==split and r['method']==method];e=np.asarray([r['mae_mm'] for r in v])
   rng=np.random.default_rng(20260926);means=rng.choice(e,(10000,len(e)),replace=True).mean(axis=1)
   summary.append({'split':split,'method':method,'n_videos':len(v),'n_frames':sum(r['n_frames'] for r in v),
                   **{k:float(np.mean([r[k] for r in v])) for k in ('mae_mm','rmse_mm','abs_rel','bias_mm','delta1_fraction')},
                   'sequence_bootstrap_mae_95ci':np.quantile(means,[.025,.975]).tolist() if len(v)>1 else None})
 payload={'summary':summary,'calibration':fit,'seconds':time.monotonic()-start,
          'model_parameter_count':sum(p.numel() for p in model.parameters()),
          'environment':{'python':sys.version,'numpy':np.__version__,'opencv':cv2.__version__,'torch':torch.__version__,'platform':platform.platform(),'device':args.device}}
 dump(out/'summary.json',payload)
 after={str(p.relative_to(ROOT)):sha(p) for p in srcs}
 if after!=source_hash:raise RuntimeError('Source changed during run')
 snapshot=out/'source_snapshot'
 for p in srcs:
  dest=snapshot/p.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
 dump(out/'manifest.json',{'source_sha256':source_hash,'outputs_sha256':{str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file() and p.name!='manifest.json'}})
 print(json.dumps([r for r in summary if r['split']=='test'],indent=2),flush=True)
if __name__=='__main__':main()
