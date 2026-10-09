"""Image-to-millimetre depth with a frozen, validation-selected calibration head.

Optional masks/rays produce a geometric quantile extent, not clinical caliper size.
"""
import argparse,json,hashlib,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.robust_metric_depth import RobustMetricSpline
from src.calibrated_depth_measurement import MonotoneDepthCalibration,measure_axial_depth

def main():
 ap=argparse.ArgumentParser(description=__doc__);inputs=ap.add_mutually_exclusive_group(required=True)
 inputs.add_argument('--image',type=Path);inputs.add_argument('--relative',type=Path)
 ap.add_argument('--head-dir',type=Path,default=ROOT/'outputs/robust_depth_optimization')
 ap.add_argument('--device',default='cuda:0');ap.add_argument('--mask',type=Path);ap.add_argument('--rays',type=Path);ap.add_argument('--output',type=Path,required=True)
 args=ap.parse_args();args.backbone=ROOT/'datasets/external_models/Depth-Anything-V2-Small-hf'
 if bool(args.mask)!=bool(args.rays):ap.error('--mask and --rays must be provided together')
 frozen=json.loads((args.head_dir/'frozen_selection.json').read_text());raw=(args.head_dir/'fitted_heads.json').read_bytes()
 if hashlib.sha256(raw).hexdigest()!=frozen['heads_sha256']:raise ValueError('Changed frozen head')
 bundle=json.loads(raw);name=frozen['selected']
 if args.image:
  import cv2,torch
  from transformers import AutoImageProcessor,AutoModelForDepthEstimation
  torch.manual_seed(20260927);torch.set_num_threads(4);torch.backends.cudnn.benchmark=False
  image=cv2.imread(str(args.image))
  if image is None:raise ValueError('Unreadable image')
  rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
  manifest=json.loads((args.backbone/'download_manifest.json').read_text())
  for row in manifest['files']:
   if hashlib.sha256((args.backbone/row['path']).read_bytes()).hexdigest()!=row['sha256']:raise ValueError('Backbone checksum mismatch')
  processor=AutoImageProcessor.from_pretrained(args.backbone,local_files_only=True,use_fast=False)
  model=AutoModelForDepthEstimation.from_pretrained(args.backbone,local_files_only=True).to(args.device).eval()
  with torch.inference_mode():
   raw=model(**processor(images=rgb,return_tensors='pt').to(args.device)).predicted_depth
   relative=torch.nn.functional.interpolate(raw.unsqueeze(1),size=rgb.shape[:2],mode='bilinear',align_corners=False).squeeze().cpu().numpy()
 else:relative=np.load(args.relative,allow_pickle=False)
 if relative.ndim!=2 or not np.isfinite(relative).all():raise ValueError('Finite H,W relative-depth map required')
 if name=='matched_affine':
  fit=bundle['affine'];den=fit['a']*relative+fit['b'];depth=np.clip(np.divide(1.,den,out=np.full_like(relative,100.),where=den>0),1,100)
 elif name=='matched_monotone':
  m=bundle['monotone'];depth=MonotoneDepthCalibration(np.array(m['x']),np.array(m['y'])).predict(relative)
 else:depth=RobustMetricSpline.from_dict(bundle['splines'][name]).predict(relative)
 result={'method':name,'depth_unit':'mm','depth_type':'axial Z','head_sha256':frozen['heads_sha256'],
         'validation':'Phantom calibration; clinical absolute scale and size accuracy unverified'}
 if args.mask:
  result['measurement']=measure_axial_depth(depth,np.load(args.mask,allow_pickle=False),np.load(args.rays,allow_pickle=False))
 args.output.mkdir(parents=True,exist_ok=False);np.save(args.output/'depth_mm.npy',depth);np.save(args.output/'relative_depth.npy',relative)
 (args.output/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
