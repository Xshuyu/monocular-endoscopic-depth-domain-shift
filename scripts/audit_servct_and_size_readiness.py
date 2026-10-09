"""Audit independent depth validation and enumerate missing clinical inputs."""
from pathlib import Path
import sys,json,csv,hashlib,datetime
import numpy as np,cv2
R=Path(__file__).resolve().parents[1];O=R/'outputs/independent_validation'
sys.path[:0]=[str(R/'scripts'),str(O/'official_toolkit')]
from run_independent_servct import check,frames
from servcttk.iotools import parse_occlusion_image

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4<<20),b''):h.update(b)
 return h.hexdigest()
def matrix(c,k):
 a=c[k];m=np.asarray(a['data'],float).reshape(a['rows'],a['cols']);assert np.isfinite(m).all();return m

def main():
 frozen=check();acq=json.loads((O/'acquisition.json').read_text());start=json.loads((O/'evaluation_start.json').read_text())
 checks={'freeze_before_acquisition':frozen['utc']<acq['utc']<start['utc'],'frozen_head_unchanged':sha(O/'frozen_heads.json')==start['frozen_heads_sha256'],'archive_members_unchanged':all(sha(R/n)==h for n,h in acq['members'].items()),'six_training_scenes_only':all(len(h['training_scenes'])==6 and not any('Experiment' in s for s in h['training_scenes']) for h in frozen['heads'].values()),'same_calibration_budget':all(h['n_samples']==16384 for h in frozen['heads'].values())}
 with (O/'per_frame.csv').open() as f:fr=list(csv.DictReader(f))
 with (O/'per_object.csv').open() as f:ob=list(csv.DictReader(f))
 summary=json.loads((O/'summary.json').read_text())
 checks['expected_score_counts']=len(fr)==16*8*3 and len(ob)==2*8*3 and len(summary)==8*3
 checks['aggregations_match']=True
 for row in ob:
  vals=[x for x in fr if all(x[k]==row[k] for k in ['method','support','object'])]
  for k in ['mae_mm','rmse_mm','abs_rel','bias_mm','delta1_fraction','bound_fraction']:checks['aggregations_match'] &= abs(np.mean([float(x[k]) for x in vals])-float(row[k]))<1e-10
 for row in summary:
  vals=[x for x in ob if all(x[k]==row[k] for k in ['method','support'])]
  for k in ['mae_mm','rmse_mm','abs_rel','bias_mm','delta1_fraction','bound_fraction']:checks['aggregations_match'] &= abs(np.mean([float(x[k]) for x in vals])-row[k])<1e-10
 checks['official_mask_matches']=True;camera=[];exact=[]
 dev=json.loads((R/'outputs/publication_study/development_frames.json').read_text());prior={sha(Path(x['color_path'])) for x in dev}
 prior|={x['rgb_sha256'] for x in json.loads((R/'outputs/radial_confirmatory/test_frames.json').read_text())}
 for row in frames():
  rgbsha=sha(R/row['rgb']);exact.append(rgbsha)
  c=json.loads((R/row['calibration']).read_text());p1,p2,q=[matrix(c,k) for k in ['P1','P2','Q']]
  assert p1.shape==(3,4) and p2.shape==(3,4) and q.shape==(4,4) and p1[0,0]>0 and p1[1,1]>0
  k=p1[:,:3];assert abs(np.linalg.det(k))>1e-10
  raw=cv2.imread(str(R/row['depth']),cv2.IMREAD_UNCHANGED);gt=raw.astype(float)/256.
  occ=cv2.imread(str(R/row['occlusion']));valid=(gt>0)&~np.all(occ==(255,0,0),axis=-1)
  official=parse_occlusion_image(R/row['occlusion'],False)&(gt>0)
  checks['official_mask_matches'] &= bool(np.array_equal(official,valid))
  disp_path=(R/row['depth']).parent.parent/'Disparity'/Path(row['depth']).name
  disp=cv2.imread(str(disp_path),cv2.IMREAD_UNCHANGED).astype(float)/256.
  mask=valid&(disp>0);ii=np.flatnonzero(mask)[::100];yy,xx=np.unravel_index(ii,gt.shape)
  homog=np.column_stack([xx,yy,disp.ravel()[ii],np.ones(len(ii))])@q.T;xyz=homog[:,:3]/homog[:,3,None]
  delta=np.abs(xyz[:,2]-gt.ravel()[ii]);rays=np.column_stack([xx,yy,np.ones(len(ii))])@np.linalg.inv(k).T
  points=rays/rays[:,2,None]*gt.ravel()[ii,None];uv=points@k.T;uv=uv[:,:2]/uv[:,2,None];reproj=np.linalg.norm(uv-np.column_stack([xx,yy]),axis=1)
  camera.append({'object':row['object'],'frame':row['frame_id'],'P1':p1.tolist(),'P2':p2.tolist(),'Q':q.tolist(),'rectified_ray_source':'inverse of P1 left 3x3; already rectified, no additional distortion applied','stereo_baseline_mm':float(abs(p2[0,3]/p2[0,0])),'depth_disparity_consistency_median_mm':float(np.median(delta)),'depth_disparity_consistency_p95_mm':float(np.quantile(delta,.95)),'reprojection_max_px':float(reproj.max()),'samples':len(ii)})
 checks['no_exact_rgb_duplicates']=len(set(exact))==16 and not set(exact)&prior
 checks={k:bool(v) for k,v in checks.items()};(O/'validation.json').write_text(json.dumps({'checks':checks,'all_passed':all(checks.values()),'limits':'Camera checks verify supplied matrix/units consistency, not new physical calibration accuracy; exact hashes cannot exclude pretraining overlap.'},indent=2)+'\n');assert all(checks.values())
 (O/'camera_calibration_audit.json').write_text(json.dumps(camera,indent=2)+'\n')
 clinical=json.loads((R/'outputs/next_validation/polyp_size_holdout_manifest.json').read_text());out=O/'physical_size_readiness';out.mkdir(exist_ok=True)
 rows=[]
 for row in clinical:rows.append({'video_id':row['video_id'],'reference_mm':row['label_size_mm'],'reference_endpoint':'post-resection maximum caliper diameter','camera_calibration_available':False,'lesion_mask_available':False,'raw_observer_records_available':False,'patient_id_available':False,'eligible_for_geometry_size_validation':False})
 with (out/'clinical_readiness.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 templates={'observer_measurements_template.csv':['patient_id','object_id','rater_id','session_id','measurement_mm','endpoint','reference_device_id','timestamp','blinded'], 'objects_template.csv':['patient_id','object_id','device_id','split','reference_endpoint','reference_size_mm','reference_method','measurement_timing','annotation_path'], 'camera_calibrations_template.csv':['device_id','calibration_id','image_width','image_height','coordinate_frame','intrinsics_path','distortion_model','distortion_coefficients_path','calibration_images_path','target_square_size_mm','heldout_reprojection_error_px','video_crop_resize_transform_path']}
 for name,header in templates.items():
  path=out/name
  if not path.exists():
   with path.open('w',newline='') as f:csv.writer(f).writerow(header)
 (out/'status.json').write_text(json.dumps({'real_mm_reference_video_count':42,'independently_verified_patient_count':None,'camera_ready':0,'raw_observer_ready':0,'geometry_eligible_count':0,'clinical_size_mae_mm':None,'recomputed_icc':None,'status':'Blocked by absent paired calibration/masks/patient IDs/raw observer measurements; values not imputed.','polyp_size_source':'https://doi.org/10.1038/s41597-025-05251-x','polyp_sense_source':'https://doi.org/10.7910/DVN/LKDIEK','polyp_sense_current_access':'Primary API returned 403; available mirror inventory contains virtual archive only.'},indent=2)+'\n')
 print(json.dumps({'checks':checks,'calibration_frames':len(camera),'max_depth_disparity_p95_mm':max(x['depth_disparity_consistency_p95_mm'] for x in camera),'size_ready':0},indent=2))
if __name__=='__main__':main()
