"""Explicit reference-processing sensitivity to official sigma unit mismatch."""
from pathlib import Path
import sys,json
R=Path(__file__).resolve().parents[1];sys.path[:0]=[str(R/'scripts'),str(R/'third_party/major_revision_deps'),str(R/'third_party/sam2'),str(R/'third_party/DRENDS/Processing'),'/tmp/lesion_publication_deps']
from prepare_drends_firstframes import render
import rendering as geo
import cv2,numpy as np,h5py,tifffile
O=R/'outputs/major_revision';D=R/'datasets/external/DRENDS/DRENDS';rows=json.loads((O/'drends_frames.json').read_text());audit=[]
for row in rows:
 folder=D/row['setup'];seq=folder/row['sequence'];dest=R/row['rgb'];dest=dest.parent
 cal={k:np.asarray(v) for k,v in json.loads((folder/'calibration.json').read_text()).items()};comp=json.loads((folder/'fine_transform.json').read_text())[row['sequence']];ws=cv2.imread(str(folder/'workspace_mask.png'),0)>0;y,x=np.where(ws);x0,x1,y0,y1=x.min(),x.max(),y.min(),y.max();K,rot,trans=geo.decompose_projection_matrix(cal['left_P']);w,h=map(int,cal['img_size'])
 with h5py.File(seq/'point_clouds.hdf5') as f:pc,v=geo.get_depth_from_hdf5(f,0,supress_outliers=True,crop=True,crop_coords=(x0,x1,y0,y1))
 pc=geo.spatial_filtering(pc,d=12,sigma_s=4.,sigma_r_mm=.005);mesh,_=geo.create_mesh_from_grid(pc);norm=geo.filter_normals(mesh,theta=60).reshape(v.shape);mesh=geo.transform_mesh(mesh,cal['he_lap_R'],cal['he_lap_T'],comp);mesh=geo.transform_mesh(mesh,rot,trans);norm &= geo.filter_camera_view(mesh,cal,K).reshape(v.shape)
 ref=render(geo.remove_points_from_mesh(mesh,v.ravel()),K,w,h);tm=cv2.imread(str(dest/'intensity_tool_mask.png'),0)>0;keep=v&norm&~tm[y0:y1,x0:x1];z=render(geo.remove_points_from_mesh(mesh,keep.ravel()),K,w,h)
 maps=cv2.initUndistortRectifyMap(cal['left_K'],cal['left_D'],cal['left_R'],cal['left_P'],(w,h),cv2.CV_32FC1);tool=cv2.remap(cv2.imread(str(dest/'left_tool_mask.png'),0),*maps,cv2.INTER_LINEAR)>0
 mask=np.isfinite(z)&(z>.01)&(z<.3)&(z<ref+.005)&~tool;z=np.where(np.isfinite(z),z*1000,0).astype('float32');tifffile.imwrite(dest/'depth_mm_sigma5mm.tiff',z);cv2.imwrite(str(dest/'mask_sigma5mm.png'),mask.astype('uint8')*255)
 gt=tifffile.imread(R/row['gt']);old=cv2.imread(str(R/row['mask']),0)>0;common=old&mask;diff=np.abs(z[common]-gt[common]);a={'id':row['id'],'setup':row['setup'],'common_pixels':int(common.sum()),'old_valid':int(old.sum()),'new_valid':int(mask.sum()),'reference_mae_difference_mm':float(diff.mean()),'reference_p95_difference_mm':float(np.quantile(diff,.95))};audit.append(a);print(a,flush=True)
(O/'reference_sensitivity.json').write_text(json.dumps(audit,indent=2))
