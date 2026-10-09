"""Auditable calibration ablation with frozen, previously unscored sequence test."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['HF_HUB_OFFLINE']='1'
import sys,json,csv,hashlib,argparse,datetime
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tests'),str(ROOT/'scripts')]
from src.radial_affine_calibration import fit,predict,radius_squared
from c3vdv2_external_depth_validation import read_reference,metrics
OUT=ROOT/'outputs/radial_confirmatory';PUB=ROOT/'outputs/publication_study'
FAMILIES=['affine_ls','affine_ridge','radial_ls','radial_ridge']
def dump(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4<<20),b''):h.update(b)
 return h.hexdigest()
def write_csv(p,rows):
 with p.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def read(n):return json.loads((OUT/n).read_text())
def sources():return {str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),ROOT/'src/radial_affine_calibration.py',ROOT/'tests/c3vdv2_external_depth_validation.py']}
def raw(row,phase):
 p=PUB/'predictions/development/small'/f"{row['scene']}_{row['frame_id']}.npz" if phase=='development' else OUT/'predictions'/f"{row['scene']}_{row['frame_id']}.npz"
 with np.load(p) as c:return c['prediction'].astype(float)
def sample(frames):
 pool={}
 for row in frames:
  gt,v=read_reference(Path(row['depth_path']));d=raw(row,'development');idx=np.flatnonzero(v);idx=idx[np.linspace(0,len(idx)-1,8192,dtype=int)]
  pool[(row['scene'],row['frame_id'])]={'d':d.ravel()[idx],'z':gt.ravel()[idx].astype(float),'r':radius_squared(d.shape).ravel()[idx]}
 return pool
def balanced(pool,scenes):
 groups=[];blocks=[];scenes=sorted(scenes)
 for j,s in enumerate(scenes):
  a=[v for (scene,_),v in pool.items() if scene==s];a={k:np.concatenate([x[k] for x in a]) for k in ['d','z','r']};n=16384//len(scenes)+(j<16384%len(scenes));idx=np.linspace(0,len(a['d'])-1,n,dtype=int)
  blocks.append({k:v[idx] for k,v in a.items()});groups.extend([s]*n)
 return {**{k:np.concatenate([v[k] for v in blocks]) for k in ['d','z','r']},'groups':np.array(groups)}
def score(head,pool,scene):return float(np.mean([np.abs(predict(head,a['d'],a['r'])-a['z']).mean() for (s,_),a in pool.items() if s==scene]))
def summarize(rows):
 videos=[];summaries=[]
 for m in sorted({r['method'] for r in rows}):
  for s in sorted({r['scene'] for r in rows}):
   a=[r for r in rows if r['method']==m and r['scene']==s]
   videos.append({'method':m,'scene':s,'frames':len(a),**{k:float(np.mean([r[k] for r in a])) for k in ['mae_mm','rmse_mm','abs_rel','delta1_fraction','bound_fraction']}})
  a=[r for r in videos if r['method']==m];summaries.append({'method':m,'videos':len(a),**{k:float(np.mean([r[k] for r in a])) for k in ['mae_mm','rmse_mm','abs_rel','delta1_fraction','bound_fraction']}})
 return videos,summaries
def evaluate(frames,phase,heads):
 rows=[];strata=[]
 for i,row in enumerate(frames):
  d=raw(row,phase);gt,v=read_reference(Path(row['depth_path']));rr=radius_squared(d.shape)
  if d.shape!=gt.shape:raise ValueError('Shape mismatch')
  rb=np.digitize(np.sqrt(rr),[.33,.66]);db=np.digitize(gt,[10,30],right=True)
  for m,h in heads[row['scene']].items():
   p=predict(h,d,rr);err=p-gt
   rows.append({'scene':row['scene'],'frame':row['frame_id'],'method':m,**metrics(p,gt,v),'bound_fraction':float(((p<=1)|(p>=100))[v].mean())})
   for radial_bin in range(3):
    for depth_bin in [-1,0,1,2]:
     mask=v&(rb==radial_bin)
     if depth_bin>=0:mask &= db==depth_bin
     if mask.any():strata.append({'scene':row['scene'],'frame':row['frame_id'],'method':m,'radius_bin':radial_bin,'depth_bin':depth_bin,'pixels':int(mask.sum()),'mae_mm':float(np.abs(err[mask]).mean()),'bias_mm':float(err[mask].mean())})
  if (i+1)%8==0:print('SCORED',phase,i+1,len(frames),flush=True)
 videos,summaries=summarize(rows);write_csv(OUT/f'{phase}_per_frame.csv',rows);write_csv(OUT/f'{phase}_per_video.csv',videos);write_csv(OUT/f'{phase}_strata.csv',strata);dump(OUT/f'{phase}_summary.json',summaries)
 print(json.dumps(summaries,indent=2),flush=True)
def verify_freeze():
 f=read('frozen_heads.json')
 if f['source_sha256']!=sources() or f['protocol_sha256']!=sha(OUT/'protocol.json'):raise ValueError('Frozen code/protocol changed')
 return f
def development():
 if (OUT/'frozen_heads.json').exists():raise ValueError('Refuse overwrite frozen study')
 frames=json.loads((PUB/'development_frames.json').read_text());pool=sample(frames);scenes=sorted({r['scene'] for r in frames});grid=read('protocol.json')['lambda_grid'];memo={};cv=[];outer=[];heads={}
 def trained(train,family,lam):
  key=(tuple(sorted(train)),family,lam)
  if key not in memo:
   a=balanced(pool,train);memo[key]=fit(a['d'],a['r'],a['z'],a['groups'],family.startswith('radial'),lam)
  return memo[key]
 for held in scenes:
  train=[s for s in scenes if s!=held];heads[held]={};choices={}
  for family in FAMILIES:
   candidates=[0.] if family.endswith('_ls') else grid;scores={}
   for lam in candidates:
    losses=[]
    for inner in train:
     h=trained([s for s in train if s!=inner],family,lam);value=score(h,pool,inner);losses.append(value);cv.append({'outer':held,'inner':inner,'family':family,'lambda':lam,'mae_mm':value})
    scores[lam]=float(np.mean(losses))
   chosen=min(scores,key=scores.get);heads[held][family]=trained(train,family,chosen);choices[family]={'lambda':chosen,'inner_scores':scores}
  outer.append({'held':held,'train':train,'choices':choices});print('OUTER',held,{k:v['lambda'] for k,v in choices.items()},flush=True)
 write_csv(OUT/'nested_selection.csv',cv);dump(OUT/'outer_folds.json',outer);dump(OUT/'outer_heads.json',heads)
 final={};selection={}
 for family in FAMILIES:
  candidates=[0.] if family.endswith('_ls') else grid;scores={lam:float(np.mean([score(trained([s for s in scenes if s!=held],family,lam),pool,held) for held in scenes])) for lam in candidates};chosen=min(scores,key=scores.get);final[family]=trained(scenes,family,chosen);selection[family]={'lambda':chosen,'development_loso_scores':scores}
 final['historical_affine']={'coefficients_a0_a1_b0_b1':[.00904776,0,.01545231,0],'provenance':'prior fixed scoring script'}
 final['historical_radial']={'coefficients_a0_a1_b0_b1':[.00768324,.00426751,.01458983,.00127847],'provenance':'prior fixed scoring script; original fit not auditable'}
 # Freeze before full OOF reporting and before acquiring new test pixels.
 dump(OUT/'frozen_heads.json',{'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'heads':final,'selection':selection,'source_sha256':sources(),'protocol_sha256':sha(OUT/'protocol.json'),'development_frames_sha256':sha(PUB/'development_frames.json'),'test_used_for_selection':False})
 print('FROZEN',selection,flush=True);evaluate(frames,'development',heads)
def acquire():
 verify_freeze();import acquire_publication_resources as a
 a.OUT=OUT
 from concurrent.futures import ThreadPoolExecutor
 with ThreadPoolExecutor(max_workers=4) as pool:
  for f in [pool.submit(a.acquire_test,item) for item in read('protocol.json')['new_test'].items()]:f.result()
def inference():
 verify_freeze();import torch,cv2
 from transformers import AutoImageProcessor,AutoModelForDepthEstimation
 frames=[]
 for scene in read('protocol.json')['new_test']:
  manifest=read(scene+'_manifest.json')
  for entry in manifest['files']:
   if sha(ROOT/entry['path'])!=entry['sha256']:raise ValueError('Input changed')
  for row in manifest['frames']:frames.append({**row,'color_path':str(ROOT/row['color_path']),'depth_path':str(ROOT/row['depth_path'])})
 seen=set()
 for folder in ['C3VDv2_native_subset','C3VDv2_robust_development','C3VD_fresh_subset']:
  for p in (ROOT/'datasets/external'/folder).rglob('*color.png'):seen.add(sha(p))
 for row in json.loads((PUB/'final_frames.json').read_text()):seen.add(sha(Path(row['color_path'])))
 current=[]
 for row in frames:row['rgb_sha256']=sha(Path(row['color_path']));current.append(row['rgb_sha256'])
 if seen&set(current) or len(current)!=len(set(current)):raise ValueError('Duplicate RGB detected')
 dump(OUT/'test_frames.json',frames);dump(OUT/'duplicate_check.json',{'prior_rgb_hashes':len(seen),'test_frames':len(frames),'overlap':0,'scope':'Exact byte hashes only; no pretraining or near-duplicate guarantee'})
 modeldir=ROOT/'datasets/external_models/Depth-Anything-V2-Small-hf'
 for rec in json.loads((modeldir/'download_manifest.json').read_text())['files']:
  if sha(modeldir/rec['path'])!=rec['sha256']:raise ValueError('Model changed')
 torch.manual_seed(20260927);torch.set_num_threads(4);torch.backends.cudnn.benchmark=False
 processor=AutoImageProcessor.from_pretrained(modeldir,local_files_only=True,use_fast=False);model=AutoModelForDepthEstimation.from_pretrained(modeldir,local_files_only=True).cuda().eval();(OUT/'predictions').mkdir(exist_ok=True)
 for i,row in enumerate(frames):
  dest=OUT/'predictions'/f"{row['scene']}_{row['frame_id']}.npz"
  if dest.exists():continue
  rgb=cv2.cvtColor(cv2.imread(row['color_path']),cv2.COLOR_BGR2RGB)
  with torch.inference_mode():
   p=model(**processor(images=rgb,return_tensors='pt').to('cuda:0')).predicted_depth;p=torch.nn.functional.interpolate(p.unsqueeze(1),size=rgb.shape[:2],mode='bilinear',align_corners=False).squeeze().cpu().numpy()
  np.savez_compressed(dest,prediction=p)
  if (i+1)%8==0:print('INFERRED',i+1,len(frames),flush=True)
 dump(OUT/'inference_complete.json',{'frames':len(frames),'parameters':sum(p.numel() for p in model.parameters()),'gt_loaded_for_inference':False})
def final():
 f=verify_freeze()
 if (OUT/'test_summary.json').exists():raise ValueError('Final evaluated already; refusing overwrite')
 frames=read('test_frames.json');dump(OUT/'final_evaluation_start.json',{'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'frozen_heads_sha256':sha(OUT/'frozen_heads.json'),'frames_sha256':sha(OUT/'test_frames.json')});evaluate(frames,'test',{r['scene']:f['heads'] for r in frames})
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('phase',choices=['development','acquire','inference','final']);a=p.parse_args();globals()[a.phase]()
