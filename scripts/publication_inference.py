"""Frozen-backbone inference for the calibration/generalization paper."""
import os,sys,json,hashlib,argparse,time
from pathlib import Path
import cv2,numpy as np,torch
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT.parent))
OUT=ROOT/'outputs/publication_study'
os.environ['HF_HUB_OFFLINE']='1';os.environ['HF_HUB_ENABLE_DOWNLOAD']='0';os.environ['XFORMERS_DISABLED']='1'
from transformers import AutoImageProcessor,AutoModelForDepthEstimation
from lesion_3d_measurement.src.depth_anything_backbone import DepthAnythingV2Backbone

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for block in iter(lambda:f.read(4<<20),b''):h.update(block)
 return h.hexdigest()

def development_frames():
 frames=[]
 pairs=[('datasets/external/C3VDv2_native_subset','calibration','outputs/c3vdv2_external_validation'),('datasets/external/C3VDv2_robust_development',None,'outputs/robust_depth_optimization')]
 for folder,split,cache in pairs:
  manifest=json.loads((ROOT/folder/'download_manifest.json').read_text())
  for row in manifest['frames']:
   if split is not None and row['split']!=split:continue
   if split is None and not row['scene'].startswith('c1_'):continue
   frames.append({'scene':row['scene'],'frame_id':row['frame_id'],'color_path':str(ROOT/folder/row['color_path']),
    'depth_path':str(ROOT/folder/row['depth_path']),'small_cache':str(ROOT/cache/'raw_predictions'/f"{row['scene']}_{row['frame_id']}.npz")})
 return frames

class EndoDACPredictor:
 def __init__(self):
  sys.path.insert(0,'/tmp/lesion_publication_deps');sys.path.insert(0,str(ROOT/'third_party/EndoDAC'))
  from models.endodac import endodac
  path=ROOT/'datasets/external_models/EndoDAC/depth_model.pth'
  manifest=json.loads((path.parent/'manifest.json').read_text())
  if sha(path)!=manifest['sha256']:raise ValueError('Checkpoint hash mismatch')
  state=torch.load(path,map_location='cpu',weights_only=True)
  self.height=int(state.pop('height'));self.width=int(state.pop('width'));state.pop('use_stereo')
  self.model=endodac(backbone_size='base',r=4,image_shape=(224,280),lora_type='dvlora',pretrained_path=None,residual_block_indexes=[2,5,8,11],include_cls_token=True)
  self.model.load_state_dict(state,strict=True);self.model.to('cuda:0').eval()
  self.parameters=sum(p.numel() for p in self.model.parameters())
 def predict(self,rgb):
  # Author pipeline: PIL LANCZOS to checkpoint W,H, ToTensor [0,1];
  # model internally bilinear-resizes to 224x280 with align_corners=True.
  resized=Image.fromarray(rgb).resize((self.width,self.height),Image.Resampling.LANCZOS)
  tensor=torch.from_numpy(np.asarray(resized).copy()).permute(2,0,1).float().div(255).unsqueeze(0).to('cuda:0')
  with torch.inference_mode():
   disp=self.model(tensor)[('disp',0)][0,0].float().cpu().numpy()
  scaled=1/150.+(1/.1-1/150.)*disp
  return cv2.resize(scaled,(rgb.shape[1],rgb.shape[0]),interpolation=cv2.INTER_LINEAR)


def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--phase',choices=['development','final'],required=True);args=ap.parse_args()
 if args.phase=='development':frames=development_frames()
 else:
  if not (OUT/'frozen_heads.json').exists():raise ValueError('Freeze calibration choices before final inference')
  frames=[]
  for scene in json.loads((OUT/'protocol.json').read_text())['final_test']:
   m=json.loads((OUT/(scene+'_manifest.json')).read_text())
   for row in m['frames']:frames.append({'scene':scene,'frame_id':row['frame_id'],'color_path':str(ROOT/row['color_path']),'depth_path':str(ROOT/row['depth_path'])})
 # Reference values are never loaded in inference.
 for row in frames:row['rgb_sha256']=sha(row['color_path'])
 if len({r['rgb_sha256'] for r in frames})!=len(frames):raise ValueError('Duplicate RGB in phase')
 if args.phase=='final':
  old=set()
  for source in ['datasets/external/C3VDv2_native_subset','datasets/external/C3VDv2_robust_development']:
   for r in json.loads((ROOT/source/'download_manifest.json').read_text())['frames']:old.add(sha(ROOT/source/r['color_path']))
  for p in (ROOT/'datasets/external/C3VD_fresh_subset').glob('*_color.png'):old.add(sha(p))
  if old & {r['rgb_sha256'] for r in frames}:raise ValueError('Final RGB duplicates observed data')
 (OUT/(args.phase+'_frames.json')).write_text(json.dumps(frames,indent=2)+'\n')
 torch.manual_seed(20260927);torch.set_num_threads(4);torch.backends.cudnn.benchmark=False
 metadata=[]
 for name in ['photo','small','large','endodac']:
  folder=OUT/'predictions'/args.phase/name;folder.mkdir(parents=True,exist_ok=True)
  pending=[r for r in frames if not (folder/f"{r['scene']}_{r['frame_id']}.npz").exists()]
  if not pending:continue
  start=time.monotonic();model=None;processor=None;count=None
  if name=='photo':model=DepthAnythingV2Backbone(device='cpu');assert model.pipe is None
  elif name=='endodac':model=EndoDACPredictor();count=model.parameters
  elif name=='large' or args.phase=='final':
   modeldir=ROOT/f'datasets/external_models/Depth-Anything-V2-{name.title()}-hf'
   manifest=json.loads((modeldir/'download_manifest.json').read_text())
   for f in manifest['files']:
    if sha(modeldir/f['path'])!=f['sha256']:raise ValueError('DA checkpoint changed')
   processor=AutoImageProcessor.from_pretrained(modeldir,local_files_only=True,use_fast=False)
   model=AutoModelForDepthEstimation.from_pretrained(modeldir,local_files_only=True).to('cuda:0').eval();count=sum(p.numel() for p in model.parameters())
  for i,row in enumerate(pending):
   if name=='small' and args.phase=='development':
    with np.load(row['small_cache']) as c:prediction=c['relative_depth_anything'].copy()
   else:
    rgb=cv2.cvtColor(cv2.imread(row['color_path']),cv2.COLOR_BGR2RGB)
    if name in ['photo','endodac']:prediction=model.predict(rgb)
    else:
     with torch.inference_mode():
      raw=model(**processor(images=rgb,return_tensors='pt').to('cuda:0')).predicted_depth
      prediction=torch.nn.functional.interpolate(raw.unsqueeze(1),size=rgb.shape[:2],mode='bilinear',align_corners=False).squeeze().float().cpu().numpy()
   if prediction.ndim!=2 or not np.isfinite(prediction).all():raise ValueError('Invalid predicted map')
   np.savez_compressed(folder/f"{row['scene']}_{row['frame_id']}.npz",prediction=prediction)
   if (i+1)%8==0:print(args.phase,name,i+1,'/',len(pending),flush=True)
  metadata.append({'model':name,'frames':len(pending),'seconds':time.monotonic()-start,'parameters':count})
  del model;torch.cuda.empty_cache()
 (OUT/(args.phase+'_inference.json')).write_text(json.dumps({'runs':metadata,'source_sha256':sha(Path(__file__)),'endodac':'Original architecture and strict full state load; author preprocessing; no test median scaling; inverse affine calibration is downstream, not author paper replication'},indent=2)+'\n')
 print('Inference complete',args.phase,flush=True)
if __name__=='__main__':main()
