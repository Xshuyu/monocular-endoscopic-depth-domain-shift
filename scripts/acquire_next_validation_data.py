"""Acquire official public data and record a frozen new-data protocol before extraction."""
import requests,re,json,hashlib,concurrent.futures,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];out=ROOT/'outputs/next_validation';out.mkdir(exist_ok=True)
protocol={'date':'2026-09-27','c3vd':'trans_t1_a official archive; 12 uniform sorted RGB frames selected without labels; external transfer test only; no refitting C3VDv2 calibration',
'polyp_size':'All 42 videos reserved as external clinical evaluation cohort, no training/tuning; inspect schema/provenance and mappings only until valid camera calibration and lesion annotation available',
'frozen_candidate_sha256':hashlib.sha256((ROOT/'outputs/calibrated_depth_candidate/frozen_calibration.json').read_bytes()).hexdigest(),
'limits':['one new phantom sequence is not a multi-patient test','backbone pretraining overlap is unverified','clinical caliper endpoint must be checked; no size score without matching measurement definition']}
if not (out/'protocol.json').exists():(out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
def download(name,url,expected_md5=None):
 dest=ROOT/'datasets/external'/name;dest.parent.mkdir(parents=True,exist_ok=True)
 if not dest.exists():
  r=requests.get(url,stream=True,timeout=(30,120));r.raise_for_status()
  if 'text/html' in r.headers.get('Content-Type',''):
   params=dict(re.findall(r'type="hidden" name="([^"]+)" value="([^"]+)"',r.text))
   if not params:raise ValueError('No public download confirmation form')
   r=requests.get('https://drive.usercontent.google.com/download',params=params,stream=True,timeout=(30,120));r.raise_for_status()
  if 'text/html' in r.headers.get('Content-Type',''):raise ValueError('Server returned HTML, not archive')
  tmp=dest.with_suffix(dest.suffix+'.partial');n=0;t=time.monotonic()
  with tmp.open('wb') as f:
   for chunk in r.iter_content(4<<20):
    f.write(chunk);n+=len(chunk)
    if n%(128<<20)<(4<<20):print(name,n,round(time.monotonic()-t,1),flush=True)
  tmp.rename(dest)
 h=hashlib.sha256();m=hashlib.md5()
 with dest.open('rb') as f:
  for block in iter(lambda:f.read(4<<20),b''):h.update(block);m.update(block)
 if expected_md5 and m.hexdigest()!=expected_md5:raise ValueError('MD5 mismatch')
 row={'path':str(dest.relative_to(ROOT)),'url':url,'bytes':dest.stat().st_size,'sha256':h.hexdigest(),'md5':m.hexdigest(),'official_md5_verified':bool(expected_md5)}
 (out/(dest.stem+'_download.json')).write_text(json.dumps(row,indent=2)+'\n');print('Ready',name,flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
 futures=[pool.submit(download,'C3VD_primary/trans_t1_a.zip','https://drive.google.com/uc?export=download&id=1urFuVo8ZalwPmsXEZg3xzhuqhpgWV8hw'),pool.submit(download,'Polyp_Size_Dataset.rar','https://ndownloader.figshare.com/files/53623187','c4f2fb4b2bbc207b618a155f2b2eeac1')]
 for f in futures:f.result()
