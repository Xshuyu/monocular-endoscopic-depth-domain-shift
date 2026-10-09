"""Prespecified five-subject expansion and stereo RGB acquisition."""
from pathlib import Path,PurePosixPath
import requests,json,zipfile,io,hashlib,datetime,time,concurrent.futures,zlib
R=Path(__file__).resolve().parents[1];O=R/'outputs/scared_expansion';O.mkdir(exist_ok=True);D=R/'datasets/external/SCARED_official/measured_keyframes';REV='c1246e4f9f52e1ffd51fa45f8544288771785189'
# Reuse the verified HTTP-range class without executing the original acquisition loop.
src=(R/'scripts/acquire_scared_keyframes.py').read_text();exec(src[src.index('class Remote'):src.index('records=json.loads')])
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
if not (O/'protocol.json').exists():
 (O/'protocol.json').write_text(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'new_subjects':[1,2,3,6,7],'previously_observed_subjects':[8,9],'exclusion':'4/5 excluded before acquisition: calibration problems documented in Allan et al 2101.01133 IV-A; no score-based exclusion','frames':'all measured left keyframes, no interpolated frames','heads_sha256':sha(R/'outputs/independent_validation/frozen_heads.json'),'heads':'six Small/Large C3VD-fitted heads unchanged; no EndoDAC independent claim','reference':'native XYZ finite, Z>0 in mm, primary all support; secondary 1<Z<100','aggregation':'frame means within subject then equal subject; report five newly acquired and seven cumulative separately; no confirmatory p-values after previous two observed','primary_contrast':'Small radial ridge minus Small affine LS','stereo_reference':'RAFT-Stereo official SceneFlow-only checkpoint, frozen 32 updates, half-resolution rectified input, Q geometry back to native left axial Z; SGBM fixed numDisparities256 block5 uniqueness10 speckle100/2; more input than monocular, not matched-input comparison','stereo_failures':'nonfinite/nonpositive disparity or depth or outside rectified grid ->100mm; every method clipped1–100mm; score identical GT support, separately report valid coverage and matched stereo coverage as diagnostic only','uncertainty':'subject bootstrap seed20261002 10000 draws exploratory seven-subject CI; new five separately','pretraining':'generic monocular pretraining not fully auditable; EndoDAC split overlap all SCARED, segregated','revision':REV},indent=2)+'\n')
records=json.loads((R/'outputs/independent_object_expansion/original_tree.txt').read_text())
def acquire(n):
 rec=next(x for x in records if x['path']==f'dataset_{n}.zip');remote=Remote(f'https://huggingface.co/datasets/maxhallan7/scared/resolve/{REV}/dataset_{n}.zip',rec['size']);result=[]
 with zipfile.ZipFile(remote) as z:
  selected=[x for x in z.infolist() if PurePosixPath(x.filename).name in {'Left_Image.png','Right_Image.png','left_depth_map.tiff','endoscope_calibration.yaml','camera_calibration.yaml'}]
  for x in selected:
   p=PurePosixPath(x.filename);assert not p.is_absolute() and '..' not in p.parts
   dest=D/p;dest.parent.mkdir(parents=True,exist_ok=True)
   if not dest.exists():dest.write_bytes(z.read(x))
   data=dest.read_bytes();assert len(data)==x.file_size and zlib.crc32(data)&0xffffffff==x.CRC
   result.append({'local':str(dest.relative_to(R)),'sha256':sha(dest),'crc32':x.CRC,'bytes':len(data),'archive':rec['path'],'revision':REV,'full_archive_hash_verified':False})
  (O/f'acquisition_{n}.json').write_text(json.dumps(result,indent=2)+'\n')
 print('ACQUIRED',n,len(result),'files',remote.transferred,'network bytes',flush=True)
 return result
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
 data=[]
 for result in pool.map(acquire,[1,2,3,6,7,8,9]):data.extend(result)
(O/'acquisition.json').write_text(json.dumps(data,indent=2)+'\n')
