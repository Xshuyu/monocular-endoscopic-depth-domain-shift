"""New official C3VD sequence: frozen C3VDv2 calibration, no target-label fitting."""
import argparse,json,sys,zipfile,os
from pathlib import Path
import cv2,numpy as np,torch
from transformers import AutoImageProcessor,AutoModelForDepthEstimation
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tests'))
from c3vdv2_external_depth_validation import sha,dump,csv_write,read_reference,metrics,predictors
from src.calibrated_depth_measurement import MonotoneDepthCalibration
from src.depth_anything_backbone import DepthAnythingV2Backbone

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,default=ROOT/'outputs/c3vd_frozen_transfer');args=ap.parse_args()
 out=args.output;out.mkdir(parents=True,exist_ok=False);data=ROOT/'datasets/external/C3VD_fresh_subset';data.mkdir(exist_ok=True)
 archive=ROOT/'datasets/external/C3VD_primary/trans_t1_a.zip'
 download=json.loads((ROOT/'outputs/next_validation/trans_t1_a_download.json').read_text())
 if sha(archive)!=download['sha256']:raise ValueError('Archive checksum mismatch')
 with zipfile.ZipFile(archive) as z:
  names=sorted(n for n in z.namelist() if n.endswith('_color.png') and '__MACOSX' not in n)
  if len(names)<12:raise ValueError('Unexpected archive naming/support')
  chosen=[names[i] for i in np.linspace(0,len(names)-1,12,dtype=int)]
  frames=[{'rgb':n,'depth':n.replace('_color.png','_depth.tiff')} for n in chosen]
  dump(out/'protocol.json',{'frames':frames,'selection':'12 uniform filename indices, before loading images or GT','archive_sha256':sha(archive),
    'calibration':'Frozen C3VDv2 c1 only; no fit or selection on this sequence','primary_metric':'mean frame axial depth MAE mm',
    'limits':['one phantom video; no population CI','unverified pretrained-corpus overlap','cross-dataset transfer, not original-paper replication'],
    'source_sha256':sha(Path(__file__)), 'frozen_fits_sha256':{name:sha(ROOT/name) for name in ['outputs/c3vdv2_external_validation/calibration.json','outputs/c3vdv2_large_validation/calibration.json','outputs/calibrated_depth_candidate/frozen_calibration.json']}})
  for row in frames:
   for key in ('rgb','depth'):
    name=row[key];dest=data/Path(name).name
    if name not in z.namelist():raise ValueError('Missing pair')
    dest.write_bytes(z.read(name));row[key]=str(dest)
 old_manifest=json.loads((ROOT/'datasets/external/C3VDv2_native_subset/download_manifest.json').read_text())
 old_rgb_hashes={sha(ROOT/'datasets/external/C3VDv2_native_subset'/r['color_path']) for r in old_manifest['frames']}
 if any(sha(Path(r['rgb'])) in old_rgb_hashes for r in frames):raise ValueError('Duplicate RGB between old and new datasets')
 torch.manual_seed(20260926);torch.set_num_threads(4);torch.backends.cudnn.benchmark=False
 allpred={};photo=DepthAnythingV2Backbone(device='cpu');assert photo.pipe is None
 for row in frames:
  rgb=cv2.cvtColor(cv2.imread(row['rgb']),cv2.COLOR_BGR2RGB)
  allpred[row['rgb']]={'photo':photo.predict(rgb)}
 for label in ('Small','Large'):
  modeldir=ROOT/f'datasets/external_models/Depth-Anything-V2-{label}-hf'
  manifest=json.loads((modeldir/'download_manifest.json').read_text())
  for f in manifest['files']:
   if sha(modeldir/f['path'])!=f['sha256']:raise ValueError('Model checksum mismatch')
  processor=AutoImageProcessor.from_pretrained(modeldir,local_files_only=True,use_fast=False)
  model=AutoModelForDepthEstimation.from_pretrained(modeldir,local_files_only=True).to('cuda:0').eval()
  for row in frames:
   rgb=cv2.cvtColor(cv2.imread(row['rgb']),cv2.COLOR_BGR2RGB)
   with torch.inference_mode():
    raw=model(**processor(images=rgb,return_tensors='pt').to('cuda:0')).predicted_depth
    pred=torch.nn.functional.interpolate(raw.unsqueeze(1),size=rgb.shape[:2],mode='bilinear',align_corners=False).squeeze().cpu().numpy()
   allpred[row['rgb']][label]=pred
  del model;torch.cuda.empty_cache();print('Inferred',label,flush=True)
 fit_small=json.loads((ROOT/'outputs/c3vdv2_external_validation/calibration.json').read_text())
 fit_large=json.loads((ROOT/'outputs/c3vdv2_large_validation/calibration.json').read_text())
 frozen=json.loads((ROOT/'outputs/calibrated_depth_candidate/frozen_calibration.json').read_text())
 def calibrated(name,pred):
  state=frozen[name];return MonotoneDepthCalibration(np.array(state['x']),np.array(state['y'])).predict(pred)
 rows=[]
 for row in frames:
  p=allpred[row['rgb']];g,v=read_reference(Path(row['depth']))
  if g.shape!=p['photo'].shape:raise ValueError('Image/reference shape mismatch')
  small,_=predictors(p['photo'],p['Small'],fit_small);large,_=predictors(p['photo'],p['Large'],fit_large)
  w=frozen['neural_weight']
  predictions={'photometric':p['photo'],'small_affine':small['depth_anything_v2_calibration_inverse_affine'],
    'large_affine':large['depth_anything_v2_calibration_inverse_affine'],
    'frozen_candidate':w*calibrated('neural',p['Small'])+(1-w)*calibrated('photometric',p['photo'])}
  np.savez_compressed(out/(Path(row['rgb']).stem+'.npz'),**p)
  for method,pred in predictions.items():rows.append({'frame':Path(row['rgb']).name,'method':method,**metrics(np.clip(pred,1,100),g,v)})
 csv_write(out/'per_frame.csv',rows)
 summary=[{'method':m,**{k:float(np.mean([r[k] for r in rows if r['method']==m])) for k in ['mae_mm','rmse_mm','abs_rel','delta1_fraction']}} for m in predictions]
 dump(out/'summary.json',summary)
 dump(out/'manifest.json',{'inputs':{row[k]:sha(Path(row[k])) for row in frames for k in ('rgb','depth')},'outputs':{str(p.relative_to(out)):sha(p) for p in out.iterdir() if p.is_file()}})
 print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
