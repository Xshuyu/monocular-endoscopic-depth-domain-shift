"""Fetch only measured keyframe members through HTTP ranges of official ZIPs."""
from pathlib import Path,PurePosixPath
import requests,json,zipfile,io,hashlib,datetime,time
R=Path(__file__).resolve().parents[1];O=R/'outputs/scared_validation';D=R/'datasets/external/SCARED_official/measured_keyframes';D.mkdir(parents=True,exist_ok=True)
REV='c1246e4f9f52e1ffd51fa45f8544288771785189'
class Remote(io.RawIOBase):
 def __init__(self,url,size):self.url=url;self.size=size;self.pos=0;self.transferred=0
 def seekable(self):return True
 def readable(self):return True
 def tell(self):return self.pos
 def seek(self,offset,whence=0):
  self.pos=offset if whence==0 else self.pos+offset if whence==1 else self.size+offset
  return self.pos
 def read(self,n=-1):
  if n<0:n=self.size-self.pos
  n=min(n,self.size-self.pos)
  if n<=0:return b''
  start=self.pos;end=start+n-1
  for attempt in range(4):
   try:
    r=requests.get(self.url+f'?range_start={start}&range_end={end}',headers={'Range':f'bytes={start}-{end}'},timeout=(30,90))
    r.raise_for_status();assert r.status_code==206,(r.status_code,n)
    assert r.headers.get('Content-Range')==f'bytes {start}-{end}/{self.size}',r.headers.get('Content-Range')
    assert len(r.content)==n;self.pos+=n;self.transferred+=n;return r.content
   except Exception:
    if attempt==3:raise
    time.sleep(2)
records=json.loads((R/'outputs/independent_object_expansion/original_tree.txt').read_text());manifest=[]
for number in [8,9]:
 name=f'dataset_{number}.zip';rec=next(x for x in records if x['path']==name)
 remote=Remote(f'https://huggingface.co/datasets/maxhallan7/scared/resolve/{REV}/{name}',rec['size'])
 with zipfile.ZipFile(remote) as z:
  infos=z.infolist();(O/f'dataset_{number}_zip_directory.json').write_text(json.dumps([{'name':x.filename,'bytes':x.file_size,'compressed':x.compress_size,'crc':x.CRC} for x in infos],indent=2))
  selected=[x for x in infos if PurePosixPath(x.filename).name in {'Left_Image.png','left_depth_map.tiff','endoscope_calibration.yaml','camera_calibration.yaml','README.md','README.txt'}]
  print('SELECTED',number,len(selected),[(x.filename,x.file_size) for x in selected],flush=True)
  for x in selected:
   p=PurePosixPath(x.filename);assert not p.is_absolute() and '..' not in p.parts
   dest=D/p;dest.parent.mkdir(parents=True,exist_ok=True)
   if not dest.exists():
    data=z.read(x);assert len(data)==x.file_size
    dest.write_bytes(data)
   import zlib
   data=dest.read_bytes();assert zlib.crc32(data)&0xffffffff==x.CRC
   manifest.append({'archive':name,'archive_lfs_sha256':rec['lfs']['oid'],'archive_full_hash_verified':False,'revision':REV,'member':x.filename,'crc32_verified':True,'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data),'local':str(dest.relative_to(R))})
   (O/'keyframe_acquisition.json').write_text(json.dumps(manifest,indent=2)+'\n')
   print('SAVED',x.filename,len(data),flush=True)
 print('TRANSFERRED',number,remote.transferred,flush=True)
