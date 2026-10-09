"""Post-hoc official EndoDAC+PCC baseline; C3VD-only calibration, no target alignment."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['HF_HUB_OFFLINE']='1';os.environ['XFORMERS_DISABLED']='1'
from pathlib import Path
import sys,json,hashlib,datetime,csv,argparse,re
import numpy as np,cv2
R=Path(__file__).resolve().parents[1];sys.path[:0]=[str(R),str(R.parent),str(R/'scripts'),str(R/'tests'),'/tmp/lesion_publication_deps',str(R/'third_party/EndoDAC')]
O=R/'outputs/recent_baselines';W=R/'datasets/external_models/EndoDAC_PCC/depth_model.pth'
from src.radial_affine_calibration import fit,predict,radius_squared
from run_radial_confirmatory import balanced
from c3vdv2_external_depth_validation import read_reference,metrics

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4<<20),b''):h.update(b)
 return h.hexdigest()
def dump(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def rows(phase):
 if phase in ['development','c3vd']:
  p=R/('outputs/publication_study/development_frames.json' if phase=='development' else 'outputs/radial_confirmatory/test_frames.json')
  return [{**r,'id':r['scene']+'_'+r['frame_id'],'object':r['scene'],'rgb':r['color_path']} for r in json.loads(p.read_text())]
 return [{**r,'id':r['object']+'_'+r['frame_id']} for r in json.loads((R/'outputs/independent_validation/frames.json').read_text())]
def freeze():
 assert not (O/'pcc_protocol.json').exists()
 dump(O/'pcc_protocol.json',{'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'checkpoint_sha256':sha(W),'script_sha256':sha(Path(__file__)),'checkpoint_source':'Official MonoPCC README EndoDAC+PCC link, Drive file 14I14aEFSRbHLnZG4GiHMqMjItqPci9Bj','source':json.loads((O/'monopcc_source.json').read_text()),'design':'Post-hoc added baseline on previously inspected test cohorts; not a new blind confirmatory test','calibration':'Same 52 C3VD development frames, 16384 equal-scene deterministic samples, train-only standardization, global and radial LS, no target tuning','prediction_range_mm':[1,100],'metrics':'Same native pixels, MAE/RMSE/AbsRel/bias/delta1, equal frames within object and equal objects; C3VD scene grouping is not independent physical specimens','cohorts':{'c3vd':len(rows('c3vd')),'servct':len(rows('servct'))},'overlap':'Published MonoPCC training list includes all nine SCARED objects; no independent SCARED claim. SERV-CT/C3VD absence from SCARED list is not an independent audit of all pretraining.','training_budget':'Original pretrained checkpoints; equal calibration data, not equal original training compute','test_gt_alignment':False})
def check():
 p=json.loads((O/'pcc_protocol.json').read_text());assert p['checkpoint_sha256']==sha(W);assert p['script_sha256']==sha(Path(__file__));return p

def inference():
 check();import torch
 from PIL import Image
 from models.endodac import endodac
 torch.manual_seed(20261002);torch.set_num_threads(4);torch.backends.cudnn.benchmark=False
 state=torch.load(W,map_location='cpu',weights_only=True);height=int(state.pop('height'));width=int(state.pop('width'));state.pop('use_stereo',None)
 model=endodac(backbone_size='base',r=4,image_shape=(224,280),lora_type='dvlora',pretrained_path=None,residual_block_indexes=[2,5,8,11],include_cls_token=True)
 model.load_state_dict(state,strict=True);model.cuda().eval()
 for phase in ['development','c3vd','servct']:
  d=O/'pcc_predictions'/phase;d.mkdir(parents=True,exist_ok=True)
  for row in rows(phase):
   out=d/(row['id']+'.npz')
   if out.exists():continue
   rgb=cv2.cvtColor(cv2.imread(str(R/row['rgb'])),cv2.COLOR_BGR2RGB)
   im=Image.fromarray(rgb).resize((width,height),Image.Resampling.LANCZOS)
   t=torch.from_numpy(np.asarray(im).copy()).permute(2,0,1).float().div(255).unsqueeze(0).cuda()
   with torch.inference_mode():disp=model(t)[('disp',0)][0,0].float().cpu().numpy()
   raw=cv2.resize(1/150+(1/.1-1/150)*disp,(rgb.shape[1],rgb.shape[0]),interpolation=cv2.INTER_LINEAR)
   assert np.isfinite(raw).all();np.savez_compressed(out,prediction=raw);print(phase,row['id'],flush=True)
 dump(O/'pcc_inference.json',{'parameters':sum(p.numel() for p in model.parameters()),'checkpoint_height':height,'checkpoint_width':width,'strict_state_loading':True,'reference_values_read':False,'rgb_sha256':{x['rgb']:sha(R/x['rgb']) for p in ['development','c3vd','servct'] for x in rows(p)}})
def raw(phase,row):
 with np.load(O/'pcc_predictions'/phase/(row['id']+'.npz')) as z:return z['prediction'].astype(float)
def calibrate():
 check();assert not (O/'pcc_heads.json').exists();pool={}
 for row in rows('development'):
  gt,v=read_reference(Path(row['depth_path']));d=raw('development',row);ix=np.flatnonzero(v);ix=ix[np.linspace(0,len(ix)-1,8192,dtype=int)]
  pool[row['scene'],row['frame_id']]={'d':d.ravel()[ix],'z':gt.ravel()[ix].astype(float),'r':radius_squared(d.shape).ravel()[ix]}
 a=balanced(pool,sorted({r['scene'] for r in rows('development')}))
 hh={('pcc_radial_ls' if radial else 'pcc_affine_ls'):fit(a['d'],a['r'],a['z'],a['groups'],radial,0.) for radial in [False,True]}
 dump(O/'pcc_heads.json',{'heads':hh,'protocol_sha256':sha(O/'pcc_protocol.json'),'test_values_used':False})
def score():
 check();assert not (O/'pcc_summary.json').exists();hh=json.loads((O/'pcc_heads.json').read_text())['heads'];rr=[]
 for phase in ['c3vd','servct']:
  for row in rows(phase):
   if phase=='c3vd':gt,v=read_reference(Path(row['depth_path']))
   else:
    gt=cv2.imread(str(R/row['depth']),cv2.IMREAD_UNCHANGED).astype(float)/256
    occ=cv2.imread(str(R/row['occlusion']));v=np.isfinite(gt)&(gt>0)&~np.all(occ==(255,0,0),axis=-1)
   d=raw(phase,row)
   for method,h in hh.items():
    pred=predict(h,d);assert pred.shape==gt.shape
    for support,mask in [('all_reference',v),('reference_1_100',v&(gt>1)&(gt<100))]:
     m=metrics(pred,gt,mask);assert np.isclose(m['mae_mm'],np.abs(pred[mask]-gt[mask]).mean(),rtol=1e-9)
     rr.append({'cohort':phase,'object':row['object'],'frame':row['id'],'method':method,'support':support,'pixels':int(mask.sum()),**m})
 with (O/'pcc_per_frame.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
 keys=['mae_mm','rmse_mm','abs_rel','bias_mm','delta1_fraction'];objs=[];s=[]
 for phase in ['c3vd','servct']:
  for method in hh:
   for support in ['all_reference','reference_1_100']:
    oo=[]
    for obj in sorted({r['object'] for r in rr if r['cohort']==phase}):
     a=[r for r in rr if (r['cohort'],r['method'],r['support'],r['object'])==(phase,method,support,obj)]
     v={'cohort':phase,'method':method,'support':support,'object':obj,**{k:float(np.mean([r[k] for r in a])) for k in keys}};objs.append(v);oo.append(v)
    s.append({'cohort':phase,'method':method,'support':support,'groups':len(oo),**{k:float(np.mean([r[k] for r in oo])) for k in keys}})
 dump(O/'pcc_per_object.json',objs);dump(O/'pcc_summary.json',s);print(json.dumps(s,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('phase',choices=['freeze','inference','calibrate','score']);a=p.parse_args();globals()[a.phase]()
