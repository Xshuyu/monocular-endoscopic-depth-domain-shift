"""Independent supervised domain adaptation, generic initialization, fixed compute and seeds."""
import os
os.environ['HF_HUB_OFFLINE']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import sys,json,hashlib,datetime,argparse,copy
import numpy as np,cv2,tifffile,torch
from torch import nn
from transformers import AutoModelForDepthEstimation,AutoImageProcessor
R=Path(__file__).resolve().parents[1];sys.path[:0]=[str(R),str(R/'tests')]
from c3vdv2_external_depth_validation import read_reference
O=R/'outputs/major_revision';M=R/'datasets/external_models/Depth-Anything-V2-Small-hf';SHAPE=(224,280)
def dump(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4<<20),b''):h.update(b)
 return h.hexdigest()
def data():
 source=[{'id':'source_'+r['scene']+'_'+r['frame_id'],'rgb':r['color_path'],'gt':r['depth_path'],'kind':'source','object':r['scene']} for r in json.loads((R/'outputs/publication_study/development_frames.json').read_text())]
 target=[]
 for n in [1,2,3,6]:
  for p in sorted((R/f'datasets/external/SCARED_official/measured_keyframes/dataset_{n}').glob('keyframe_*/Left_Image.png')):target.append({'id':f'target_{n}_{p.parent.name}','rgb':str(p),'gt':str(p.parent/'left_depth_map.tiff'),'kind':'target','object':n})
 assert len(source)==52 and len(target)==20
 return source+target
def freeze():
 assert not (O/'training_protocol.json').exists()
 rows=data();hashes={r['id']:sha(r['rgb']) for r in rows};assert len(set(hashes.values()))==len(rows)
 test=json.loads((R/'outputs/scared_expansion/frames.json').read_text());test=[r for r in test if r['subject'] in [8,9]]
 assert not set(hashes.values())&{sha(r['rgb']) for r in test}
 manifest=json.loads((M/'download_manifest.json').read_text())
 for x in manifest['files']:assert sha(M/x['path'])==x['sha256']
 dump(O/'training_protocol.json',{'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'script_sha256':sha(Path(__file__)),'type':'Supervised low-data domain adaptation baseline; not a claimed reproduction of EndoDAC or an unsupervised method','initialization':'Generic DA V2 Small only; no SCARED-adapted network checkpoint','frozen':'22M-parameter generic backbone','trainable':'Pretrained neck and depth head; final scalar conv reinitialized for log-mm prediction; output ReLU replaced by Identity','source_objects':'Original 6 C3VD development scenes, 52 frames','adaptation_objects':[1,2,3,6],'target_frames':20,'unused_target_subject':7,'target_test_subjects':[8,9],'other_tests':'48 prior C3VD test frames,16 SERV-CT views,new DRENDS sequences','seeds':[11,23,47],'steps':600,'batch_size':2,'optimizer':'AdamW lr=1e-4 weight_decay=.01 gradient_clip=1','loss':'Per-image mean SmoothL1(log-depth, log-reference), beta=.1; then equal images','source_only_sampling':'Two uniformly sampled source frames per step','adapted_sampling':'One source and one adaptation frame per step','resolution':list(SHAPE),'image_processing':'Official normalization, resized directly to224x280, keep_aspect_ratio=False; cached frozen backbone features','gt_resample':'Nearest neighbor for depth and valid masks; positive finite axial mm','checkpoint_selection':'Final step600 only; no validation or test selection; all three seeds reported','primary_output_bounds_mm':[1,100],'secondary_output_bounds_mm':[1,300],'training_gt_bounds':'finite positive reference, log target clamped1–300 mm','pretraining_limit':'Generic foundation pretraining cannot be independently exhaustively audited','historical_test_exposure':'Existing C3VD/SCARED/SERV-CT previously inspected; new runs are post-hoc, DRENDS new external acquisition','training_rgb_sha256':hashes,'training_gt_sha256':{r['id']:sha(r['gt']) for r in rows},'test_rgb_disjoint':True,'rows':rows,'generic_weights_manifest':manifest})
def model():
 m=AutoModelForDepthEstimation.from_pretrained(M,local_files_only=True)
 m.head.activation2=nn.Identity()
 return m

def cache():
 p=json.loads((O/'training_protocol.json').read_text());assert p['script_sha256']==sha(Path(__file__))
 torch.set_num_threads(4);m=model().cuda().eval();proc=AutoImageProcessor.from_pretrained(M,local_files_only=True,use_fast=False);d=O/'training_cache';d.mkdir(exist_ok=True)
 for row in p['rows']:
  out=d/(row['id']+'.pt')
  if out.exists():continue
  rgb=cv2.cvtColor(cv2.imread(row['rgb']),cv2.COLOR_BGR2RGB)
  inp=proc(images=rgb,return_tensors='pt',size={'height':224,'width':280},keep_aspect_ratio=False).to('cuda');assert tuple(inp.pixel_values.shape[-2:])==SHAPE
  with torch.no_grad():f=m.backbone.forward_with_filtered_kwargs(inp.pixel_values).feature_maps
  if row['kind']=='source':gt,v=read_reference(Path(row['gt']))
  else:
   xyz=tifffile.imread(row['gt']);gt=xyz[...,2];v=np.isfinite(xyz).all(-1)&(gt>0)
  gt=cv2.resize(np.where(v,gt,1).astype('float32'),SHAPE[::-1],interpolation=cv2.INTER_NEAREST);v=cv2.resize(v.astype('uint8'),SHAPE[::-1],interpolation=cv2.INTER_NEAREST).astype(bool)
  torch.save({'features':[x.cpu() for x in f],'log_gt':torch.from_numpy(np.log(np.clip(gt,1,300))),'valid':torch.from_numpy(v)},out)
  print('CACHE',row['id'],flush=True)

def train():
 p=json.loads((O/'training_protocol.json').read_text());assert p['script_sha256']==sha(Path(__file__))
 torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.use_deterministic_algorithms(True)
 cached={r['id']:torch.load(O/'training_cache'/(r['id']+'.pt'),weights_only=True) for r in p['rows']}
 src=[r['id'] for r in p['rows'] if r['kind']=='source'];tgt=[r['id'] for r in p['rows'] if r['kind']=='target'];d=O/'trained_models';d.mkdir(exist_ok=True)
 for seed in p['seeds']:
  for regime in ['source_only','adapted']:
   dest=d/f'{regime}_{seed}.pt'
   if dest.exists():continue
   torch.manual_seed(seed);rng=np.random.default_rng(seed);m=model().cuda()
   for v in m.backbone.parameters():v.requires_grad_(False)
   nn.init.normal_(m.head.conv3.weight,0,.001);nn.init.constant_(m.head.conv3.bias,np.log(50))
   m.neck.train();m.head.train();pars=[v for v in m.parameters() if v.requires_grad];opt=torch.optim.AdamW(pars,lr=1e-4,weight_decay=.01);log=[]
   for step in range(600):
    ids=[src[rng.integers(len(src))],(tgt if regime=='adapted' else src)[rng.integers(len(tgt) if regime=='adapted' else len(src))]];batch=[cached[x] for x in ids]
    features=[torch.cat([x['features'][i] for x in batch]).cuda() for i in range(len(batch[0]['features']))]
    pred=m.head(m.neck(features,16,20),16,20);gt=torch.stack([x['log_gt'] for x in batch]).cuda();valid=torch.stack([x['valid'] for x in batch]).cuda()
    loss=torch.stack([torch.nn.functional.smooth_l1_loss(pred[i][valid[i]],gt[i][valid[i]],beta=.1) for i in range(2)]).mean()
    assert torch.isfinite(loss);opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(pars,1.);opt.step()
    if step%50==0 or step==599:log.append({'step':step+1,'loss':float(loss.detach())});print(regime,seed,step+1,float(loss.detach()),flush=True)
   torch.save({'neck':m.neck.state_dict(),'head':m.head.state_dict(),'seed':seed,'regime':regime,'steps':600,'trainable_parameters':sum(v.numel() for v in pars),'protocol_sha256':sha(O/'training_protocol.json')},dest)
   dump(d/f'{regime}_{seed}_log.json',log);del m,opt;torch.cuda.empty_cache()
 dump(O/'training_complete.json',{'checkpoints':{p.name:sha(p) for p in d.glob('*.pt')},'script_sha256':sha(Path(__file__))})
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('phase',choices=['freeze','cache','train']);args=a.parse_args();globals()[args.phase]()
