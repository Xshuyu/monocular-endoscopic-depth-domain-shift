"""Acquire publisher SCARED test archives, verify SHA256 and safe ZIP contents."""
from pathlib import Path, PurePosixPath
import requests,json,hashlib,zipfile,datetime
R=Path(__file__).resolve().parents[1]; O=R/'outputs/scared_validation';D=R/'datasets/external/SCARED_official';D.mkdir(parents=True,exist_ok=True)
REV='c1246e4f9f52e1ffd51fa45f8544288771785189'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4<<20),b''):h.update(b)
 return h.hexdigest()
protocol=O/'protocol.json'
if not protocol.exists():
 protocol.write_text(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'locked before test archive download and scoring','selection':'All original measured left keyframes in SCARED datasets8/9; interpolation excluded. If compact test archives lack original measured keyframes obtain same members from full publisher archives. No error-based selection.','heads_source':'outputs/independent_validation/frozen_heads.json','heads_sha256':sha(R/'outputs/independent_validation/frozen_heads.json'),'methods':['small_affine_ls','small_affine_ridge','small_radial_ls','small_radial_ridge','large_affine_ls','large_radial_ls','endodac_affine_ls','endodac_radial_ls'],'fitting':'None; frozen C3VD heads, no target median scaling or alignment','primary':'All finite XYZ with positive axial Z in mm; native left RGB coordinate system. MAE per frame then equal subject means; paired radial-affine differences, descriptive N=2 SCARED subjects.','secondary':'1<reference Z<100mm, separately reported. Frozen prediction clip1–100mm unchanged for all.','reference':'Original structured-light keyframes only, Z channel from XYZ, not Euclidean range. Verify units from publisher evaluator.','identity_source':'Allan et al arXiv2101.01133 sectionII: each dataset corresponds to a single porcine subject. Do not assume independence across SERV-CT and SCARED without cross-study provenance.','training_overlap':'Audit EndoDAC released splits against dataset8/9; foundation pretraining not fully auditable.','scope':'Frozen calibrated monocular transfer; not replication of stereo challenge ranking or lesion-size validation.','revision':REV},indent=2)+'\n')
records=json.loads((R/'outputs/independent_object_expansion/original_tree.txt').read_text());manifest=[]
for name in ['code.zip','test_dataset_8.zip','test_dataset_9.zip']:
 rec=next(x for x in records if x['path']==name);dest=D/name
 if not dest.exists():
  u=f'https://huggingface.co/datasets/maxhallan7/scared/resolve/{REV}/{name}?download=true'
  print('DOWNLOAD',name,rec['size'],flush=True)
  with requests.get(u,stream=True,timeout=(30,90)) as r:
   r.raise_for_status();assert r.status_code==200
   with dest.with_suffix('.partial').open('wb') as f:
    for b in r.iter_content(4<<20): f.write(b)
  assert dest.with_suffix('.partial').stat().st_size==rec['size']
  dest.with_suffix('.partial').rename(dest)
 digest=sha(dest);assert digest==rec['lfs']['oid'],name
 with zipfile.ZipFile(dest) as z:
  for n in z.namelist():
   p=PurePosixPath(n.replace('\\','/'));assert not p.is_absolute() and '..' not in p.parts and ':' not in n
  assert z.testzip() is None
  folder=D/name.removesuffix('.zip');folder.mkdir(exist_ok=True);z.extractall(folder)
  members=z.namelist()
 manifest.append({'archive':name,'sha256':digest,'bytes':dest.stat().st_size,'members':members})
 (O/'acquisition.json').write_text(json.dumps(manifest,indent=2)+'\n')
 print('VERIFIED',name,len(members),'members',members[:12],flush=True)
