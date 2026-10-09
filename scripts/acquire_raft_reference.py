"""Restore pinned official RAFT-Stereo source and SceneFlow checkpoint."""
from pathlib import Path,PurePosixPath
import requests,zipfile,io,hashlib,json
R=Path(__file__).resolve().parents[1];P=R/'datasets/external_models/RAFT-Stereo';P.mkdir(parents=True,exist_ok=True)
COMMIT='6e93ed2169bd858dbb43033988563f3b0bb49506'
URL='https://www.dropbox.com/scl/fi/5khx1bhz84dapi8vtwapg/models.zip?rlkey=ggddrn1du1iiq6mgc2dsdpmwi&dl=1'
EXPECTED='ba75bdcb379b4c3942d1e714565d0e66e6b2450a3da8129f54073302d67e0a6d'
if not (P/'models.zip').exists():
 with requests.get(URL,stream=True,timeout=(30,90)) as response:
  response.raise_for_status()
  with (P/'models.zip.partial').open('wb') as f:
   for b in response.iter_content(4<<20):f.write(b)
 (P/'models.zip.partial').rename(P/'models.zip')
with zipfile.ZipFile(P/'models.zip') as z:
 names=[n for n in z.namelist() if PurePosixPath(n).name=='raftstereo-sceneflow.pth' and '__MACOSX' not in PurePosixPath(n).parts];assert len(names)==1
 data=z.read(names[0]);assert hashlib.sha256(data).hexdigest()==EXPECTED
 (P/'raftstereo-sceneflow.pth').write_bytes(data)
root=R/'third_party/RAFT-Stereo'
if not (root/'core/raft_stereo.py').exists():
 response=requests.get(f'https://api.github.com/repos/princeton-vl/RAFT-Stereo/zipball/{COMMIT}',timeout=90);response.raise_for_status()
 with zipfile.ZipFile(io.BytesIO(response.content)) as z:
  for n in z.namelist():
   rel=PurePosixPath(*PurePosixPath(n).parts[1:]);assert not rel.is_absolute() and '..' not in rel.parts
   if not n.endswith('/') and rel.parts:
    p=root/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(z.read(n))
manifest={'source':URL,'code_commit':COMMIT,'checkpoint':'raftstereo-sceneflow.pth','sha256':EXPECTED,'bytes':len(data),'training_claim':'Official SceneFlow checkpoint; train_stereo.py default dataset sceneflow; original training logs not independently audited','code_sha256':{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('*.py')}}
(P/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print('Verified pinned RAFT-Stereo reference.')
