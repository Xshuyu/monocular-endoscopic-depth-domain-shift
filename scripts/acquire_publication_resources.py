"""Read selected members of official public ZIPs via standard HTTP byte ranges."""
import io,json,re,hashlib,zipfile,requests,tarfile,concurrent.futures
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/publication_study'
class RemoteZipFile(io.RawIOBase):
 def __init__(self,file_id):
  self.session=requests.Session();self.origin=f'https://drive.google.com/uc?export=download&id={file_id}'
  r=self.session.get(self.origin,stream=True,timeout=(30,90));r.raise_for_status()
  if 'text/html' not in r.headers.get('Content-Type',''):raise ValueError('Expected public download confirmation')
  params=dict(re.findall(r'type="hidden" name="([^"]+)" value="([^"]+)"',r.text));r.close()
  if not params:raise ValueError('No public download form; not bypassing restriction')
  response=self.session.get('https://drive.usercontent.google.com/download',params=params,headers={'Range':'bytes=0-0'},timeout=(30,90))
  if response.status_code!=206:raise ValueError(f'Range unsupported: {response.status_code} {response.headers.get("Content-Type")}')
  self.url=response.url;self.size=int(response.headers['Content-Range'].split('/')[-1]);self.pos=0
 def readable(self):return True
 def seekable(self):return True
 def tell(self):return self.pos
 def seek(self,offset,whence=0):
  self.pos=offset if whence==0 else self.pos+offset if whence==1 else self.size+offset
  if self.pos<0:raise ValueError('Negative seek')
  return self.pos
 def read(self,n=-1):
  if n is None or n<0:n=self.size-self.pos
  n=min(n,self.size-self.pos)
  if n<=0:return b''
  start=self.pos;end=start+n-1
  for attempt in range(3):
   try:
    r=self.session.get(self.url,headers={'Range':f'bytes={start}-{end}'},timeout=(30,90));r.raise_for_status()
    if r.status_code!=206 or not r.headers.get('Content-Range','').startswith(f'bytes {start}-{end}/') or len(r.content)!=n:raise ValueError('Invalid byte range response')
    self.pos+=n;return r.content
   except (requests.RequestException,ValueError):
    if attempt==2:raise
 def close(self):
  if hasattr(self,'session'):self.session.close()
  super().close()

def acquire_test(item):
 scene,file_id=item;dest=ROOT/'datasets/external/C3VD_publication_final'/scene;dest.mkdir(parents=True,exist_ok=True)
 with RemoteZipFile(file_id) as remote,zipfile.ZipFile(remote) as z:
  depths={int(Path(n).stem.split('_')[0]):n for n in z.namelist() if n.endswith('_depth.tiff') and '__MACOSX' not in n}
  colors=sorted((n for n in z.namelist() if n.endswith('_color.png') and '__MACOSX' not in n and int(Path(n).stem.split('_')[0]) in depths),key=lambda n:int(Path(n).stem.split('_')[0]))
  chosen=[colors[round(k*(len(colors)-1)/11)] for k in range(12)]
  frames=[{'scene':scene,'frame_id':Path(n).stem.split('_')[0],'color_member':n,'depth_member':depths[int(Path(n).stem.split('_')[0])]} for n in chosen]
  (OUT/(scene+'_selection.json')).write_text(json.dumps({'source':remote.origin,'archive_bytes':remote.size,'frames':frames,'selection_before_extraction':True},indent=2)+'\n')
  files=[]
  for row in frames:
   for kind in ['color','depth']:
    member=row[kind+'_member'];info=z.getinfo(member);p=dest/Path(member).name
    if not p.exists():p.write_bytes(z.read(member))
    data=p.read_bytes()
    import zlib
    if zlib.crc32(data)!=info.CRC:raise ValueError('CRC mismatch')
    files.append({'path':str(p.relative_to(ROOT)),'sha256':hashlib.sha256(data).hexdigest(),'zip_crc32':info.CRC,'bytes':len(data),'member':member})
    row[kind+'_path']=str(p.relative_to(ROOT))
  (OUT/(scene+'_manifest.json')).write_text(json.dumps({'official_source':remote.origin,'files':files,'frames':frames,'integrity':'ZIP CRC and local SHA256; complete archive was not downloaded or hashed'},indent=2)+'\n')
 print('Ready final sequence',scene,flush=True)

def acquire_endodac():
 metadata=json.loads((OUT/'EndoDAC_tree.json').read_text());commit=metadata['commit'];dest=ROOT/'third_party/EndoDAC'
 if not dest.exists():
  r=requests.get(f'https://codeload.github.com/BeileiCui/EndoDAC/tar.gz/{commit}',timeout=90);r.raise_for_status()
  parent=ROOT/'third_party';parent.mkdir(exist_ok=True)
  with tarfile.open(fileobj=io.BytesIO(r.content),mode='r:gz') as t:
   for m in t.getmembers():
    if m.issym() or m.islnk() or '..' in Path(m.name).parts or Path(m.name).is_absolute():raise ValueError('Unsafe code archive')
   t.extractall(parent,filter='data')
  (parent/f'EndoDAC-{commit}').rename(dest)
 modeldir=ROOT/'datasets/external_models/EndoDAC';modeldir.mkdir(exist_ok=True)
 with RemoteZipFile('1qzAYBtwYJDN7hEi6pApqBOOz6pUhyY70') as remote,zipfile.ZipFile(remote) as z:
  inventory=[{'name':x.filename,'size':x.file_size,'compressed':x.compress_size} for x in z.infolist()]
  (OUT/'endodac_checkpoint_inventory.json').write_text(json.dumps(inventory,indent=2)+'\n')
  print('EndoDAC checkpoint members',[(x['name'],x['size']) for x in inventory],flush=True)
  names=[n for n in z.namelist() if n.endswith('depth_model.pth')]
  if len(names)!=1:raise ValueError('Ambiguous checkpoint selection')
  name=names[0];p=modeldir/'depth_model.pth'
  if not p.exists():p.write_bytes(z.read(name))
  import zlib
  data=p.read_bytes()
  if zlib.crc32(data)!=z.getinfo(name).CRC:raise ValueError('Model CRC mismatch')
  record={'code_commit':commit,'checkpoint_source':remote.origin,'member':name,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'zip_crc32':z.getinfo(name).CRC}
  (modeldir/'manifest.json').write_text(json.dumps(record,indent=2)+'\n')
 print('Ready EndoDAC model',flush=True)
if __name__=='__main__':
 protocol=json.loads((OUT/'protocol.json').read_text())
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
  futures=[pool.submit(acquire_test,item) for item in protocol['final_test'].items()]+[pool.submit(acquire_endodac)]
  for f in futures:f.result()
