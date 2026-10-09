"""Balanced analytic surface benchmark with explicit, consistent reference units.

This geometry-only supplement removes the old renderer's nominal/rendered-size
mismatch. It does not replace RGB rendering or provide clinical size ground truth.
"""
import argparse,csv,hashlib,json,sys,time
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT.parent))
from lesion_3d_measurement.src.robust_obb_methods import pca_obb,mad_filtered_pca_obb
from lesion_3d_measurement.src.lie_algebra_obb import compute_robust_obb_lie

def surface(shape,major,seed,n=500):
 rng=np.random.default_rng(seed);theta=rng.uniform(0,2*np.pi,n);r=np.sqrt(rng.uniform(0,1,n))
 a=major/2;b=.7*a;x=a*r*np.cos(theta);y=b*r*np.sin(theta)
 # Explicit support samples make physical full x/y extents exactly known.
 x=np.r_[x,[-a,a,0,0,0]];y=np.r_[y,[0,0,-b,b,0]]
 if shape=='disc':z=np.zeros_like(x)
 elif shape=='dome':z=.25*major*np.sqrt(np.maximum(0,1-(x/a)**2-(y/b)**2))
 elif shape=='saddle':z=.15*major*((x/a)**2-(y/b)**2)
 else:raise ValueError(shape)
 local=np.column_stack([x,y,z]);rotation=Rotation.random(random_state=rng).as_matrix()
 points=local@rotation.T+np.array([3.,-2.,45.])
 assert abs(np.ptp(local,axis=0).max()-major)<1e-10
 q=np.quantile(local,.99,axis=0)-np.quantile(local,.01,axis=0)
 return points,float(q.max()),rng

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=ROOT/'outputs/consistent_surface_reference');args=p.parse_args()
 out=args.output;out.mkdir(parents=True,exist_ok=False)
 protocol={'shapes':['disc','dome','saddle'],'major_mm':[4.,8.,16.,24.],'seeds':list(range(4100,4105)),
           'outlier_fraction_added':[0.,.05,.10],'inlier_points':505,'rotation':'random SO(3) per seed and shape/size cell',
           'primary_reference':'clean pre-contamination 1st–99th percentile extent along known intrinsic model axes',
           'secondary_reference':'known full intrinsic-axis major diameter; reported separately, not conflated with quantile extent',
           'design':'full crossing of shape, size, noise and seed; no parameter fitting',
           'comparisons':'PCA vs SO(3) with identical no filtering, or identical always-on radial MAD; same reported quantiles',
           'limitations':'analytic visible-surface point clouds, not images or clinical lesion masks; no illumination or depth prediction evaluated'}
 (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
 rows=[];start=time.monotonic()
 for seed in protocol['seeds']:
  for si,shape in enumerate(protocol['shapes']):
   for di,major in enumerate(protocol['major_mm']):
    pts,truth,rng=surface(shape,major,seed*100+si*10+di)
    # Fixed absolute-mm contamination, shared as a nested prefix across levels.
    bad=rng.uniform([-35,-35,10],[35,35,80],(int(.1*len(pts)),3))
    for noise in protocol['outlier_fraction_added']:
     cloud=np.vstack([pts,bad[:int(noise*len(pts))]])
     estimates={'pca_q':pca_obb(cloud,quantiles=(.01,.99))[0],
                'mad_pca_q':mad_filtered_pca_obb(cloud,quantiles=(.01,.99))[0]}
     for mode in ['never','always']:
      estimates['so3_'+mode]=compute_robust_obb_lie(cloud,outlier_filter=mode).major_diameter_mm
     for method,pred in estimates.items():
      rows.append({'seed':seed,'shape':shape,'full_major_mm':major,'outlier_fraction':noise,
                   'quantile_reference_mm':truth,'method':method,'prediction_mm':float(pred),
                   'primary_quantile_abs_error_mm':float(abs(pred-truth)),
                   'secondary_full_extent_abs_error_mm':float(abs(pred-major))})
  print('Completed geometry seed',seed,flush=True)
 with (out/'trials.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 summary=[]
 for shape in protocol['shapes']:
  for noise in protocol['outlier_fraction_added']:
   for method in ['pca_q','mad_pca_q','so3_never','so3_always']:
    group=[r for r in rows if r['shape']==shape and r['outlier_fraction']==noise and r['method']==method]
    seed_means=[np.mean([r['primary_quantile_abs_error_mm'] for r in group if r['seed']==s]) for s in protocol['seeds']]
    draws=np.random.default_rng(20260926).choice(seed_means,(10000,len(seed_means)),replace=True).mean(axis=1)
    summary.append({'shape':shape,'outlier_fraction':noise,'method':method,'n_cases':len(group),
                    'quantile_mae_mm':float(np.mean(seed_means)),'seed_block_95ci':np.quantile(draws,[.025,.975]).tolist(),
                    'full_extent_mae_mm':float(np.mean([r['secondary_full_extent_abs_error_mm'] for r in group]))})
 payload={'n_unique_cases':len(rows)//4,'n_rows':len(rows),'summary':summary,'seconds':time.monotonic()-start}
 (out/'summary.json').write_text(json.dumps(payload,indent=2))
 sources=[Path(__file__).resolve(),ROOT/'src/lie_algebra_obb.py',ROOT/'src/robust_obb_methods.py']
 (out/'manifest.json').write_text(json.dumps({'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},'output_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file()}},indent=2))
 print('Completed',len(rows)//4,'balanced cases',flush=True)
if __name__=='__main__':main()
