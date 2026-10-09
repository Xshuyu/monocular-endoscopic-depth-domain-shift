"""Public HTTP range transfer with official MD5 verification."""
import requests,concurrent.futures,hashlib,json
from pathlib import Path
root=Path(__file__).resolve().parents[1];out=root/'datasets/external';parts=out/'polyp_size_parts';parts.mkdir(exist_ok=True)
url='https://ndownloader.figshare.com/files/53623187';size=362536394
ranges=[(i*size//8,(i+1)*size//8-1) for i in range(8)]
def fetch(item):
 i,(a,b)=item;p=parts/f'{i}.part'
 if p.exists() and p.stat().st_size==b-a+1:return p
 offset=p.stat().st_size if p.exists() else 0
 a+=offset
 with requests.get(url,headers={'Range':f'bytes={a}-{b}'},stream=True,timeout=(30,60)) as r:
  if r.status_code!=206 or not r.headers.get('Content-Range','').startswith(f'bytes {a}-{b}/'):raise ValueError('Server did not honor range')
  with p.open('ab') as f:
   for chunk in r.iter_content(1<<20):f.write(chunk)
 if p.stat().st_size!=b-a+1+offset:raise ValueError('Incomplete range')
 print('Finished part',i,flush=True);return p
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:paths=list(pool.map(fetch,enumerate(ranges)))
dest=out/'Polyp_Size_Dataset.rar';temp=out/'Polyp_Size_Dataset.assembled'
h=hashlib.sha256();md5=hashlib.md5()
with temp.open('wb') as f:
 for p in paths:
  with p.open('rb') as g:
   for block in iter(lambda:g.read(4<<20),b''):f.write(block);h.update(block);md5.update(block)
if md5.hexdigest()!='c4f2fb4b2bbc207b618a155f2b2eeac1':raise ValueError('Official MD5 mismatch')
temp.rename(dest)
(root/'outputs/next_validation/Polyp_Size_Dataset_download.json').write_text(json.dumps({'url':url,'bytes':size,'sha256':h.hexdigest(),'md5':md5.hexdigest(),'official_md5_verified':True},indent=2)+'\n')
print('Official Polyp-Size archive verified',flush=True)
