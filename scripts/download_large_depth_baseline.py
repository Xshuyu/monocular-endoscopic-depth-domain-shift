"""Download a fixed public official DA V2 Large HF checkpoint without changing environment."""
from pathlib import Path
import hashlib,json,requests,time
ROOT=Path(__file__).resolve().parents[1]
REPO='depth-anything/Depth-Anything-V2-Large-hf'
REV='7581137eff8d4e94f6e796d3baea0e9fa79b22d2'
out=ROOT/'datasets/external_models/Depth-Anything-V2-Large-hf';out.mkdir(parents=True,exist_ok=True)
files=[]
for name in ['config.json','preprocessor_config.json','README.md','model.safetensors']:
 url=f'https://huggingface.co/{REPO}/resolve/{REV}/{name}'
 dest=out/name
 if not dest.exists():
  temp=out/(name+'.partial')
  with requests.get(url,stream=True,timeout=(30,120)) as response:
   response.raise_for_status()
   with temp.open('wb') as f:
    count=0;start=time.monotonic()
    for chunk in response.iter_content(4<<20):
     f.write(chunk);count+=len(chunk)
     if count%(128<<20)<(4<<20):print(name,count,'bytes',round(time.monotonic()-start,1),'seconds',flush=True)
  temp.rename(dest)
 h=hashlib.sha256()
 with dest.open('rb') as f:
  for block in iter(lambda:f.read(4<<20),b''):h.update(block)
 files.append({'path':name,'sha256':h.hexdigest(),'size':dest.stat().st_size,'url':url})
 print('Ready',name,dest.stat().st_size,flush=True)
(out/'download_manifest.json').write_text(json.dumps({'repo':REPO,'revision':REV,'files':files},indent=2)+'\n')
