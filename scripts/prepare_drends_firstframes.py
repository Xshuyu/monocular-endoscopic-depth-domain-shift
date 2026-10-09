"""Fixed first-frame external validation, official geometry with CPU axial raycasting."""
from pathlib import Path
import os,sys,json,hashlib,argparse
R=Path(__file__).resolve().parents[1];sys.path[:0]=[str(R/'third_party/major_revision_deps'),str(R/'third_party/sam2'),str(R/'third_party/DRENDS/Processing')]
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='4'
import numpy as np,cv2,torch,h5py,tifffile,open3d as o3d
import rendering as geo
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
O=R/'outputs/major_revision';D=R/'datasets/external/DRENDS/DRENDS';OUT=R/'datasets/external/DRENDS/firstframes';OUT.mkdir(exist_ok=True)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4<<20),b''):h.update(b)
 return h.hexdigest()
def first(p):
 cap=cv2.VideoCapture(str(p));ok,bgr=cap.read();cap.release();assert ok,p;return bgr

def render(mesh,K,w,h):
 scene=o3d.t.geometry.RaycastingScene(nthreads=4);scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
 y,x=np.indices((h,w));dirs=np.stack([(x+.5-K[0,2])/K[0,0],(y+.5-K[1,2])/K[1,1],np.ones((h,w))],-1).astype('float32')
 rays=np.concatenate([np.zeros_like(dirs),dirs],-1)
 # Unnormalized direction with z=1: t_hit is axial Z, not Euclidean ray length.
 return scene.cast_rays(o3d.core.Tensor(rays))['t_hit'].numpy()
def geometry_test():
 mesh=o3d.geometry.TriangleMesh();mesh.vertices=o3d.utility.Vector3dVector([[-10,-10,2],[10,-10,2],[-10,10,2],[10,10,2]]);mesh.triangles=o3d.utility.Vector3iVector([[0,1,2],[1,3,2]])
 z=render(mesh,np.array([[40,0,31.5],[0,40,23.5],[0,0,1.]]),64,48);assert np.max(np.abs(z-2))<1e-5;return float(np.max(np.abs(z-2)))
def main(modality):
 torch.set_num_threads(4);torch.manual_seed(20261002)
 protocol=O/'drends_processing_protocol.json'
 if not protocol.exists():protocol.write_text(json.dumps({'first_frame':0,'modality_groups':['Phantom','ExVivo'],'selection':'All nonconsecutive Seq folders, no score selection','official_processing_commit':'077ec79fd4c414508bea1de1ef1e14b94911c942','modifications':'Fixed first frames; SAM2 image predictor with original first-frame prompts; CPU raycasting replaces Linux offscreen renderer with unnormalized rays (axial Z); no temporal change at first frame; all other geometry/filter parameters retained. Not full published benchmark reproduction.','depth_mm':'XYZ stored mm -> meters for official transform -> axial renderedZ *1000, float32 retained','tools':'SAM2 Hiera Large, author prompts for left and intensity, thresholds0; intensity mask cropped then removed from mesh; rectified left mask removed from evaluation','spatial_filter':'Official d12 sigma_s4 sigma_r_mm5 passed unchanged to official function; name suggests mm but code operates on meter coordinates, retained for comparability and explicitly flagged','range_mm':[10,300],'occlusion_tolerance_mm':5,'camera_compensation':'Author supplied fixed transform, no manual fitting to model predictions','independence':'Phantom organ pieces and ex vivo organ preparations; no verified donor IDs or camera serials; do not count sequences as independent donors','script_sha256':sha(Path(__file__))},indent=2))
 assert json.loads(protocol.read_text())['script_sha256']==sha(Path(__file__))
 predictor=SAM2ImagePredictor(build_sam2('configs/sam2/sam2_hiera_l.yaml',str(R/'datasets/external_models/SAM2/sam2_hiera_large.pt'),device='cuda'))
 folder=D/modality;cal={k:np.asarray(v) for k,v in json.loads((folder/'calibration.json').read_text()).items()};prompts=json.loads((folder/'tool_prompts.json').read_text());comp=json.loads((folder/'fine_transform.json').read_text());ws=cv2.imread(str(folder/'workspace_mask.png'),0)>0;ys,xs=np.where(ws);x0,x1,y0,y1=xs.min(),xs.max(),ys.min(),ys.max();K,rot,trans=geo.decompose_projection_matrix(cal['left_P']);w,h=map(int,cal['img_size'])
 maps=cv2.initUndistortRectifyMap(cal['left_K'],cal['left_D'],cal['left_R'],cal['left_P'],(w,h),cv2.CV_32FC1)
 records=[];audit=[];plane_error=geometry_test()
 for seq in sorted(folder.glob('Seq[0-9][0-9]_*')):
  if not (seq/'left.mp4').exists():continue
  dest=OUT/(modality+'_'+seq.name);dest.mkdir(exist_ok=True)
  rec={'cohort':'drends','object':modality+'_'+seq.name.split('_')[1],'setup':modality,'illumination':seq.name.split('_')[-1],'sequence':seq.name,'id':'drends_'+dest.name,'rgb':str((dest/'rgb.png').relative_to(R)),'gt':str((dest/'depth_mm.tiff').relative_to(R)),'mask':str((dest/'mask.png').relative_to(R))}
  if (dest/'audit.json').exists():records.append(rec);audit.append(json.loads((dest/'audit.json').read_text()));continue
  images={k:first(seq/(k+'.mp4')) for k in ['left','intensity']}
  if modality=='ExVivo':images['left']=cv2.fastNlMeansDenoisingColored(images['left'],None,8,8,7,21)
  masks={}
  for kind,im in images.items():
   predictor.set_image(cv2.cvtColor(im,cv2.COLOR_BGR2RGB));mask=np.zeros(im.shape[:2],bool)
   for points in prompts[seq.name+'/'+kind]:
    pts=np.asarray(points,dtype=np.float32)
    with torch.inference_mode():mm,_,_=predictor.predict(point_coords=pts,point_labels=np.ones(len(pts),np.int32),multimask_output=False)
    mask|=mm[0].astype(bool)
   masks[kind]=mask;cv2.imwrite(str(dest/(kind+'_tool_mask.png')),mask.astype('uint8')*255)
  rect=cv2.remap(images['left'],*maps,cv2.INTER_LINEAR);tool=cv2.remap(masks['left'].astype('uint8')*255,*maps,cv2.INTER_LINEAR)>0;cv2.imwrite(str(dest/'rgb.png'),rect)
  with h5py.File(seq/'point_clouds.hdf5','r') as f:pc,valid=geo.get_depth_from_hdf5(f,0,supress_outliers=True,crop=True,crop_coords=(x0,x1,y0,y1))
  pc=geo.spatial_filtering(pc,d=12,sigma_s=4.,sigma_r_mm=5.);mesh,material=geo.create_mesh_from_grid(pc);norm=geo.filter_normals(mesh,theta=60,bidirectional=False).reshape(valid.shape)
  mesh=geo.transform_mesh(mesh,cal['he_lap_R'],cal['he_lap_T'],compensation=comp[seq.name]);mesh=geo.transform_mesh(mesh,rot,trans)
  norm &= geo.filter_camera_view(mesh,cal,K).reshape(valid.shape)
  full=geo.remove_points_from_mesh(mesh,valid.ravel());ref=render(full,K,w,h)
  keep=valid&norm&~masks['intensity'][y0:y1,x0:x1];filt=geo.remove_points_from_mesh(mesh,keep.ravel());depth=render(filt,K,w,h)
  mask=np.isfinite(depth)&(depth>.01)&(depth<.3)&(depth<ref+.005)&~tool
  assert mask.any(),seq
  tifffile.imwrite(dest/'depth_mm.tiff',np.where(np.isfinite(depth),depth*1000,0).astype('float32'));cv2.imwrite(str(dest/'mask.png'),mask.astype('uint8')*255)
  a={'sequence':dest.name,'valid_pixels':int(mask.sum()),'valid_fraction':float(mask.mean()),'depth_range_mm':[float(depth[mask].min()*1000),float(depth[mask].max()*1000)],'mean_depth_mm':float(depth[mask].mean()*1000),'tool_fraction':float(tool.mean()),'calibration_sha256':sha(folder/'calibration.json'),'fine_transform_sha256':sha(folder/'fine_transform.json'),'rgb_sha256':sha(dest/'rgb.png'),'reference_sha256':sha(dest/'depth_mm.tiff'),'plane_axial_test_error_m':plane_error}
  (dest/'audit.json').write_text(json.dumps(a,indent=2));records.append(rec);audit.append(a);print('PROCESSED',dest.name,int(mask.sum()),flush=True)
 (O/f'drends_frames_{modality}.json').write_text(json.dumps(records,indent=2));(O/f'drends_geometry_{modality}.json').write_text(json.dumps(audit,indent=2))
 allrows=[]
 for p in sorted(O.glob('drends_frames_*.json')):allrows+=json.loads(p.read_text())
 (O/'drends_frames.json').write_text(json.dumps(allrows,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('modality',choices=['Phantom','ExVivo']);a=p.parse_args();main(a.modality)
