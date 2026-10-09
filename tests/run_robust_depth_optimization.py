"""Develop on c1 only, freeze a head, then evaluate predeclared reused/new-sequence tests."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse,json,sys,time
from pathlib import Path
import cv2,numpy as np,torch
from transformers import AutoImageProcessor,AutoModelForDepthEstimation
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.robust_metric_depth import RobustMetricSpline,depth_feature
from src.calibrated_depth_measurement import MonotoneDepthCalibration
from c3vdv2_external_depth_validation import sha,dump,csv_write,read_reference,predictors,metrics


def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--evaluate',action='store_true');ap.add_argument('--output',type=Path,default=ROOT/'outputs/robust_depth_optimization');args=ap.parse_args()
 out=args.output;dev=ROOT/'datasets/external/C3VDv2_robust_development';old=ROOT/'datasets/external/C3VDv2_native_subset'
 manifest=json.loads((dev/'download_manifest.json').read_text());old_manifest=json.loads((old/'download_manifest.json').read_text())
 frames=[{**r,'root':dev,'cache':out/'raw_predictions'/f"{r['scene']}_{r['frame_id']}.npz"} for r in manifest['frames']]
 old_frames=[{**r,'root':old,'cache':ROOT/'outputs/c3vdv2_external_validation/raw_predictions'/f"{r['scene']}_{r['frame_id']}.npz"} for r in old_manifest['frames']]
 fit_frames=[r for r in frames if r['split']=='fit']+[{**r,'split':'fit'} for r in old_frames if r['split']=='calibration']
 val_frames=[r for r in frames if r['split']=='validation']
 specs=[(mode,reg,temp) for mode in ['raw','frame_quantile'] for reg in [0.,.02,.2] for temp in [None,2.]]
 def ident(mode,reg,temp):return f"{mode}_smooth{reg:g}_{'mean' if temp is None else 'dro2'}"
 def load(row,gt=True):
  with np.load(row['cache']) as c:n=c['relative_depth_anything'].copy()
  if not gt:return n
  g,v=read_reference(row['root']/row['depth_path']);return n,g,v
 if not args.evaluate:
  out.mkdir(parents=True,exist_ok=False);(out/'raw_predictions').mkdir()
  sources=[Path(__file__),ROOT/'src/robust_metric_depth.py']
  protocol={'source_sha256':{str(p.relative_to(ROOT)):sha(p) for p in sources},'fit_scenes':sorted({r['scene'] for r in fit_frames}),
    'validation_scenes':sorted({r['scene'] for r in val_frames}),'fit_frames':len(fit_frames),'validation_frames':len(val_frames),
    'samples_per_fit_frame':2048,'scene_weighting':'Equal-scene pseudo-Huber mean, or entropic worst-scene soft maximum, temperature 2mm',
    'candidate_grid':[{'name':ident(*s),'mode':s[0],'smoothness':s[1],'scene_temperature':s[2]} for s in specs],
    'spline_knots':12,'delta_mm':.5,'selection':'Lowest equal-video full-resolution MAE on c1 transverse1/transverse2; includes matched-data inverse affine and 64-bin monotone baselines',
    'test':'No c2 or original C3VD labels used to fit/select. Evaluate once after frozen_selection.json. Old c2/C3VD are reused tests; c2 sigmoid is a new video on previously observed anatomy.',
    'predeclared_test_ablations':['raw_smooth0_mean','raw_smooth0.02_mean','raw_smooth0_dro2','raw_smooth0.02_dro2'],
    'network_training':False,'data_manifest_sha256':sha(dev/'download_manifest.json'),'command':[sys.executable,*sys.argv]}
  dump(out/'protocol.json',protocol)
  for record in old_manifest['files']:
   if sha(old/record['path'])!=record['sha256']:raise ValueError('Changed original data')
  cache_hashes=json.loads((ROOT/'outputs/c3vdv2_external_validation/manifest.json').read_text())['outputs_sha256']
  for row in old_frames:
   if sha(row['cache'])!=cache_hashes['raw_predictions/'+row['cache'].name]:raise ValueError('Changed old cache')
  for record in manifest['files']:
   if sha(dev/record['path'])!=record['sha256']:raise ValueError('Changed input')
  # Byte-level split guard includes previously used test scenes.
  split_hashes={}
  for rows,key in [(fit_frames,'fit'),(val_frames,'validation'),([r for r in frames if r['split']=='new_sequence_test']+[r for r in old_frames if r['split']=='test'],'test')]:
   split_hashes[key]={sha(r['root']/r['color_path']) for r in rows}
  if any(split_hashes[a]&split_hashes[b] for a,b in [('fit','validation'),('fit','test'),('validation','test')]):raise ValueError('Cross-split duplicate RGB')
  modeldir=ROOT/'datasets/external_models/Depth-Anything-V2-Small-hf'
  mm=json.loads((modeldir/'download_manifest.json').read_text())
  for r in mm['files']:
   if sha(modeldir/r['path'])!=r['sha256']:raise ValueError('Model hash mismatch')
  torch.manual_seed(20260927);torch.set_num_threads(4);torch.backends.cudnn.benchmark=False
  processor=AutoImageProcessor.from_pretrained(modeldir,local_files_only=True,use_fast=False)
  model=AutoModelForDepthEstimation.from_pretrained(modeldir,local_files_only=True).to('cuda:0').eval()
  for i,row in enumerate(frames):
   rgb=cv2.cvtColor(cv2.imread(str(row['root']/row['color_path'])),cv2.COLOR_BGR2RGB)
   with torch.inference_mode():
    raw=model(**processor(images=rgb,return_tensors='pt').to('cuda:0')).predicted_depth
    n=torch.nn.functional.interpolate(raw.unsqueeze(1),size=rgb.shape[:2],mode='bilinear',align_corners=False).squeeze().cpu().numpy()
   np.savez_compressed(row['cache'],relative_depth_anything=n)
   if (i+1)%8==0:print('Inferred',i+1,len(frames),flush=True)
  del model;torch.cuda.empty_cache()
  arrays={mode:[] for mode in ['raw','frame_quantile']};ys=[];groups=[]
  for row in fit_frames:
   n,g,v=load(row);idx=np.flatnonzero(v);idx=idx[np.linspace(0,len(idx)-1,2048,dtype=int)]
   for mode in arrays:arrays[mode].append(depth_feature(n,mode).ravel()[idx])
   ys.append(g.ravel()[idx]);groups.extend([row['scene']]*len(idx))
  y=np.concatenate(ys);groups=np.array(groups);arrays={k:np.concatenate(v) for k,v in arrays.items()}
  _,gi,gc=np.unique(groups,return_inverse=True,return_counts=True);weights=1/gc[gi];design=np.column_stack([arrays['raw'],np.ones(len(y))]);w=np.sqrt(weights)
  a,b=np.linalg.lstsq(design*w[:,None],(1/y)*w,rcond=None)[0]
  affine={'a':float(a),'b':float(b)}
  mono=MonotoneDepthCalibration.fit(arrays['raw'],y,increasing=False)
  models={}
  for spec in specs:
   name=ident(*spec);t=time.monotonic()
   models[name]=RobustMetricSpline.fit(arrays[spec[0]],y,groups,mode=spec[0],smoothness=spec[1],scene_temperature=spec[2])
   print('Fitted',name,models[name].diagnostics['iterations'],'iterations',round(time.monotonic()-t,2),'seconds',flush=True)
  bundle={'affine':affine,'monotone':mono.to_dict(),'splines':{k:v.to_dict() for k,v in models.items()}}
  dump(out/'fitted_heads.json',bundle)
 else:
  protocol=json.loads((out/'protocol.json').read_text())
  for name,h in protocol['source_sha256'].items():
   if sha(ROOT/name)!=h:raise ValueError('Source changed since fitting')
  bundle=json.loads((out/'fitted_heads.json').read_text())
  frozen=json.loads((out/'frozen_selection.json').read_text())
  if sha(out/'fitted_heads.json')!=frozen['heads_sha256']:raise ValueError('Head changed after selection')
  models={k:RobustMetricSpline.from_dict(v) for k,v in bundle['splines'].items()}
  affine=bundle['affine'];m=bundle['monotone'];mono=MonotoneDepthCalibration(np.array(m['x']),np.array(m['y']))
 def predict(name,n):
  if name=='matched_affine':
   d=affine['a']*n+affine['b'];return np.clip(np.divide(1.,d,out=np.full_like(n,100.),where=d>0),1,100)
  if name=='matched_monotone':return mono.predict(n)
  return models[name].predict(n)
 if not args.evaluate:
  rows=[]
  for row in val_frames:
   n,g,v=load(row)
   for name in ['matched_affine','matched_monotone',*models]:rows.append({'scene':row['scene'],'frame':row['frame_id'],'method':name,**metrics(predict(name,n),g,v)})
  csv_write(out/'validation_per_frame.csv',rows)
  scores={name:float(np.mean([np.mean([r['mae_mm'] for r in rows if r['method']==name and r['scene']==scene]) for scene in sorted({r['scene'] for r in rows})])) for name in ['matched_affine','matched_monotone',*models]}
  selected=min(scores,key=scores.get)
  dump(out/'frozen_selection.json',{'selected':selected,'validation_mae':scores,'heads_sha256':sha(out/'fitted_heads.json'),'protocol_sha256':sha(out/'protocol.json')})
  print('FROZEN',selected,json.dumps(scores,indent=2),flush=True);return
 # Evaluate only frozen selection and the predeclared ablations, never choose by test.
 protocol=json.loads((out/'protocol.json').read_text());selected=frozen['selected']
 names=list(dict.fromkeys(['matched_affine','matched_monotone',selected,*protocol['predeclared_test_ablations']]))
 cases=[{**r,'evaluation_set':'new_c2_sigmoid_video'} for r in frames if r['split']=='new_sequence_test']
 cases += [{**r,'evaluation_set':'reused_c3vdv2_test'} for r in old_frames if r['split']=='test']
 transfer=ROOT/'outputs/c3vd_frozen_transfer'
 for p in sorted(transfer.glob('*_color.npz')):
  cases.append({'scene':'trans_t1_a','frame_id':p.stem,'evaluation_set':'reused_c3vd_test','transfer_cache':p,
    'root':ROOT/'datasets/external/C3VD_fresh_subset','depth_path':p.stem.replace('_color','_depth')+'.tiff'})
 rows=[]
 oldfit=json.loads((ROOT/'outputs/c3vdv2_external_validation/calibration.json').read_text())
 for row in cases:
  if 'transfer_cache' in row:
   with np.load(row['transfer_cache']) as c:n=c['Small'].copy()
   g,v=read_reference(row['root']/row['depth_path'])
  else:n,g,v=load(row)
  base=oldfit['depth_anything_inverse_affine_a']*n+oldfit['depth_anything_inverse_affine_b']
  baseline=np.clip(np.divide(1.,base,out=np.full_like(n,100.),where=base>0),1,100)
  predictions={name:predict(name,n) for name in names};predictions['original_small_affine']=baseline
  for name,pred in predictions.items():rows.append({'evaluation_set':row['evaluation_set'],'scene':row['scene'],'frame':row['frame_id'],'method':name,**metrics(pred,g,v)})
 csv_write(out/'test_per_frame.csv',rows)
 summary=[]
 for dataset in sorted({r['evaluation_set'] for r in rows}):
  for name in [*names,'original_small_affine']:
   group=[r for r in rows if r['evaluation_set']==dataset and r['method']==name];scenes=sorted({r['scene'] for r in group})
   summary.append({'evaluation_set':dataset,'method':name,'n_videos':len(scenes),'n_frames':len(group),
     **{k:float(np.mean([np.mean([r[k] for r in group if r['scene']==scene]) for scene in scenes])) for k in ['mae_mm','rmse_mm','abs_rel','delta1_fraction']}})
 dump(out/'test_summary.json',summary)
 dump(out/'manifest.json',{str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file() and p.name!='manifest.json'})
 print(json.dumps([r for r in summary if r['method'] in [selected,'matched_affine','original_small_affine']],indent=2))
if __name__=='__main__':main()
