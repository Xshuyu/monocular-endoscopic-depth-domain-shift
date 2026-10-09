from pathlib import Path
import requests,json,hashlib,zipfile,io,concurrent.futures
R=Path(__file__).resolve().parents[1];O=R/'outputs/major_revision';D=R/'datasets/external/DRENDS';D.mkdir(parents=True,exist_ok=True)
meta=json.loads((O/'drends_zenodo.json').read_text())
def fetch(rec):
 p=D/rec['key']
 if not p.exists():
  with requests.get(rec['links']['self'],stream=True,timeout=(30,120)) as q:
   q.raise_for_status()
   with p.with_suffix('.partial').open('wb') as f:
    for b in q.iter_content(4<<20):f.write(b)
  p.with_suffix('.partial').rename(p)
 h=hashlib.md5()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4<<20),b''):h.update(b)
 assert 'md5:'+h.hexdigest()==rec['checksum'];print('VERIFIED',p.name,p.stat().st_size,flush=True)
 with zipfile.ZipFile(p) as z:
  assert all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in z.namelist())
  (O/(p.stem+'_members.json')).write_text(json.dumps([{'name':x.filename,'bytes':x.file_size} for x in z.infolist()],indent=2));z.extractall(D if rec["key"]=="DRENDS.zip" else D/"DRENDS"/("Phantom" if "Phantom" in rec["key"] else "ExVivo"))
 return {'file':str(p.relative_to(R)),'checksum':rec['checksum'],'bytes':p.stat().st_size}
if __name__=='__main__':
 wanted=['DRENDS.zip','DRENDS_Phantom_Sequences.zip','DRENDS_ExVivo_Sequences.zip']
 with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:out=list(ex.map(fetch,[x for x in meta['files'] if x['key'] in wanted]))
 (O/'drends_acquisition.json').write_text(json.dumps(out,indent=2))
