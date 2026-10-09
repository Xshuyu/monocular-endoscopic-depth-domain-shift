"""Apply a frozen radial head; optional sizing requires calibrated camera rays."""
import argparse,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.radial_affine_calibration import predict
from src.calibrated_depth_measurement import measure_axial_depth

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--relative',type=Path,required=True);p.add_argument('--heads',type=Path,required=True);p.add_argument('--method',default='radial_ridge');p.add_argument('--mask',type=Path);p.add_argument('--rays',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if bool(a.mask)!=bool(a.rays):p.error('mask and calibrated rays must be supplied together')
 d=np.load(a.relative,allow_pickle=False)
 if d.ndim!=2 or not np.isfinite(d).all():raise ValueError('Finite native H,W relative map required')
 h=json.loads(a.heads.read_text())['heads'][a.method];z=predict(h,d)
 record={'method':a.method,'head':h,'status':'Phantom-calibrated research output; no clinical size validation','endpoint':'axial Z in mm'}
 if a.mask:record['measurement']=measure_axial_depth(z,np.load(a.mask,allow_pickle=False),np.load(a.rays,allow_pickle=False))
 a.output.mkdir(parents=True,exist_ok=False);np.save(a.output/'depth_mm.npy',z);(a.output/'measurement.json').write_text(json.dumps(record,indent=2)+'\n')
if __name__=='__main__':main()
