"""Apply frozen depth calibration; optional measurement needs mask and calibrated rays.

Input NPY files contain native-resolution raw DA-v2 relative depth and photometric
millimetre depth, respectively. Rays must correspond to the same pixel grid.
"""
import argparse,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.calibrated_depth_measurement import MonotoneDepthCalibration,measure_axial_depth

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--relative',type=Path,required=True)
    ap.add_argument('--photometric',type=Path,required=True)
    ap.add_argument('--calibration',type=Path,required=True)
    ap.add_argument('--mask',type=Path)
    ap.add_argument('--rays',type=Path)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    if bool(args.mask)!=bool(args.rays):ap.error('--mask and --rays must be provided together')
    fit=json.loads(args.calibration.read_text())
    weight=float(fit['neural_weight'])
    if not 0<=weight<=1:raise ValueError('Invalid fusion weight')
    def apply(name,path):
        state=fit[name];model=MonotoneDepthCalibration(np.array(state['x']),np.array(state['y']))
        return model.predict(np.load(path,allow_pickle=False))
    n=apply('neural',args.relative);p=apply('photometric',args.photometric)
    if n.ndim!=2 or n.shape!=p.shape:raise ValueError('Matched H,W maps required')
    depth=weight*n+(1-weight)*p
    result={'depth_units':'mm','depth_type':'axial Z','neural_weight':weight,
            'status':'experimental; calibrated on a silicone phantom; not validated clinical sizing'}
    if args.mask:
        mask=np.load(args.mask,allow_pickle=False);rays=np.load(args.rays,allow_pickle=False)
        result['measurement']=measure_axial_depth(depth,mask,rays)
        measures=[measure_axial_depth(x,mask,rays)['major_q01_q99_mm'] for x in (p,depth,n)]
        result['component_sensitivity_range_mm']=[min(measures),max(measures)]
        result['sensitivity_note']='Variation between component depth predictions; NOT a confidence interval or error bound'
    args.output.mkdir(parents=True,exist_ok=False)
    np.save(args.output/'depth_mm.npy',depth)
    (args.output/'measurement.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
