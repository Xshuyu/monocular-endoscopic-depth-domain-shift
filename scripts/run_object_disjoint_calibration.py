"""Exploratory recalibration on already observed SERV-CT; not fresh validation."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import sys,json,csv,hashlib,datetime
import numpy as np,cv2
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from src.radial_affine_calibration import fit,radius_squared
O=R/'outputs/object_disjoint_calibration';B=R/'outputs/independent_validation'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def csvout(p,rows):
 with p.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def predict(h,d,r,cap):
 a,b,c,e=h['coefficients_a0_a1_b0_b1'];q=(a+b*r)*d+c+e*r
 return np.clip(np.divide(1.,q,out=np.full_like(q,float(cap)),where=q>0),1,cap)
def main():
 O.mkdir(exist_ok=False)
 protocol={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'Exploratory, designed after SERV-CT results were observed. NOT an untouched external validation.','folds':'Fit all eight views of one specimen, evaluate all eight of the other; reverse. Neither view nor pixel splitting crosses objects. Camera configuration and object remain confounded.','models':['small','large','endodac'],'heads':['affine_ls','radial_ls'],'budgets':[1024,4096,16384],'seeds':[202610010,202610011,202610012,202610013,202610014],'sampling':'Uniform without replacement from 8192 fixed valid pixels per training image, equal image quotas; shared samples across models/heads, nested budgets.','selection':'None; all budgets/seeds/families reported; lambda=0 throughout','reference':'uint16/256 mm, positive and not official blue outside-reference region','endpoint':'Full-native MAE/RMSE/AbsRel/delta1; mean image then equal two held-out specimens','caps_mm':[100,250],'caps_status':'100 primary matches original protocol;250 secondary range sensitivity, within uint16/256 encoding. Apply each range to BOTH unchanged source heads and recalibrated heads; never select best cap.','uncertainty':'Seed SD represents calibration-pixel randomness only; two objects, no population inference','source_sha256':{str(Path(__file__).relative_to(R)):sha(Path(__file__)), 'src/radial_affine_calibration.py':sha(R/'src/radial_affine_calibration.py')}}
 dump(O/'protocol.json',protocol)
 protected={str(p.relative_to(R)):sha(p) for p in B.glob('*.json')};protected.update({str(p.relative_to(R)):sha(p) for p in B.glob('*.csv')})
 ff=json.loads((B/'frames.json').read_text());source=json.loads((B/'frozen_heads.json').read_text())['heads'];data={};inputs={}
 for row in ff:
  dp=R/row['depth'];op=R/row['occlusion'];gt=cv2.imread(str(dp),cv2.IMREAD_UNCHANGED).astype(float)/256.;occ=cv2.imread(str(op));v=(gt>0)&~np.all(occ==(255,0,0),axis=-1);idx=np.flatnonzero(v);pool=idx[np.linspace(0,len(idx)-1,8192,dtype=int)];r=radius_squared(gt.shape).ravel();d={}
  for model in protocol['models']:
   pp=B/'predictions'/model/(row['object']+'_'+row['frame_id']+'.npz');inputs[str(pp.relative_to(R))]=sha(pp)
   with np.load(pp) as z:d[model]=z['prediction'].astype(float).ravel()
  for pp in [dp,op]:inputs[str(pp.relative_to(R))]=sha(pp)
  data[row['object'],row['frame_id']]={'gt':gt.ravel(),'idx':idx,'pool':pool,'r':r,'d':d}
 dump(O/'input_hashes.json',inputs)
 rows=[];heads=[]
 def score(head,model,held,kind,budget,seed):
  for (obj,frame),a in data.items():
   if obj!=held:continue
   ix=a['idx'];z=a['gt'][ix]
   for cap in protocol['caps_mm']:
    p=predict(head,a['d'][model][ix],a['r'][ix],cap);err=p-z
    rows.append({'held_object':held,'frame':frame,'model':model,'head':kind,'budget':budget,'seed':seed,'cap_mm':cap,'mae_mm':float(np.abs(err).mean()),'rmse_mm':float(np.sqrt(np.mean(err**2))),'abs_rel':float(np.mean(np.abs(err)/z)),'delta1':float(np.mean(np.maximum(p/z,z/p)<1.25)),'boundary_fraction':float(np.mean((p<=1)|(p>=cap)))})
 for held in ['Experiment_1','Experiment_2']:
  train=[k for k in data if k[0]!=held];assert len(train)==8 and all(k[0]!=held for k in train)
  for model in protocol['models']:
   for kind in protocol['heads']:score(source[model+'_'+kind],model,held,'source_'+kind,0,-1)
  for seed in protocol['seeds']:
   rng=np.random.default_rng(np.random.SeedSequence([seed,int(held[-1])]))
   perm={k:rng.permutation(len(data[k]['pool'])) for k in train}
   for budget in protocol['budgets']:
    chosen={k:data[k]['pool'][perm[k][:budget//8]] for k in train};z=np.concatenate([data[k]['gt'][chosen[k]] for k in train]);rr=np.concatenate([data[k]['r'][chosen[k]] for k in train]);groups=np.concatenate([np.repeat(k[1],len(chosen[k])) for k in train]);assert len(z)==budget
    for model in protocol['models']:
     d=np.concatenate([data[k]['d'][model][chosen[k]] for k in train])
     for kind in protocol['heads']:
      h=fit(d,rr,z,groups,kind=='radial_ls',0.);heads.append({'held_object':held,'training_object':train[0][0],'budget':budget,'seed':seed,'model':model,'head_kind':kind,'fit':h});score(h,model,held,'adapted_'+kind,budget,seed)
   print('Scored',held,'seed',seed,flush=True)
 csvout(O/'per_frame.csv',rows);dump(O/'heads.json',heads)
 keys=['mae_mm','rmse_mm','abs_rel','delta1','boundary_fraction'];groups={}
 for row in rows:
  key=tuple(row[k] for k in ['held_object','model','head','budget','seed','cap_mm']);groups.setdefault(key,[]).append(row)
 objects=[]
 for key,rr in groups.items():objects.append({**dict(zip(['held_object','model','head','budget','seed','cap_mm'],key)),**{k:float(np.mean([x[k] for x in rr])) for k in keys}})
 csvout(O/'per_object.csv',objects);summary=[]
 combos=sorted({(x['model'],x['head'],x['budget'],x['cap_mm']) for x in objects})
 for model,kind,budget,cap in combos:
  rr=[x for x in objects if (x['model'],x['head'],x['budget'],x['cap_mm'])==(model,kind,budget,cap)];seeds=sorted({x['seed'] for x in rr});means=[np.mean([x['mae_mm'] for x in rr if x['seed']==s]) for s in seeds]
  summary.append({'model':model,'head':kind,'budget':budget,'cap_mm':cap,'objects':2,'seeds':len(seeds),'mae_mm':float(np.mean(means)),'seed_sd_mm':float(np.std(means,ddof=1)) if len(means)>1 else None,**{k:float(np.mean([x[k] for x in rr])) for k in keys if k!='mae_mm'}})
 dump(O/'summary.json',summary)
 checks={'180_fits':len(heads)==180,'train_test_objects_disjoint':all(h['held_object']!=h['training_object'] for h in heads),'old_results_unchanged':all(sha(R/n)==h for n,h in protected.items()),'full_frame_scores':len(rows)==3072,'finite_results':all(np.isfinite(x['mae_mm']) for x in rows)}
 dump(O/'validation.json',checks);assert all(checks.values()),checks
 print(json.dumps([x for x in summary if x['budget'] in [0,16384]],indent=2))
if __name__=='__main__':main()
