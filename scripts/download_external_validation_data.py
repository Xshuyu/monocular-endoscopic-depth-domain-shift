"""Fetch fixed, documented public mirrors; preserve checksums and selection.

Mirror hashes establish download integrity, not equivalence to inaccessible
primary Dataverse files. No credentials or restricted endpoints are used.
"""
import argparse,concurrent.futures,hashlib,json,time
from pathlib import Path
from urllib.parse import quote
import requests
ROOT=Path(__file__).resolve().parents[1]

def get_json(url):
    r=requests.get(url,timeout=45); r.raise_for_status(); return r.json()

def fetch(url,path,expected=None,size=None):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        h=hashlib.sha256(path.read_bytes()).hexdigest()
        if (not expected or h==expected) and (size is None or path.stat().st_size==size):return h
    part=path.with_suffix(path.suffix+'.partial')
    for attempt in range(3):
        try:
            h=hashlib.sha256(); total=0
            with requests.get(url,stream=True,timeout=(30,90)) as r:
                r.raise_for_status()
                with part.open('wb') as f:
                    for chunk in r.iter_content(1<<20):
                        if chunk:f.write(chunk); h.update(chunk); total+=len(chunk)
            if size is not None and total!=size:raise ValueError(f'Size mismatch {total} != {size}')
            if expected and h.hexdigest()!=expected:raise ValueError('SHA256 mismatch')
            part.replace(path); return h.hexdigest()
        except Exception:
            if attempt==2:raise
            time.sleep(2)

def main():
    p=argparse.ArgumentParser();p.add_argument('dataset',choices=['polypsense','c3vdv2','c3vdv2_native']);args=p.parse_args()
    if args.dataset=='polypsense':
        repo='BeileiCui/PolypSense3D_archived'; meta=get_json('https://huggingface.co/api/datasets/'+repo)
        rev=meta['sha']; out=ROOT/'datasets/external/PolypSense3D_archive';out.mkdir(parents=True,exist_ok=True)
        name='PolypSense3D.7z.001'; url=f'https://huggingface.co/datasets/{repo}/resolve/{rev}/{name}'
        expected='5d9f2e71b564c3e0fb227b88d055c2bb8ea9bb705c749855f634678ca08e3ad0'
        print('Downloading PolypSense3D mirror archive: 1.106 GB',flush=True)
        h=fetch(url,out/name,expected,1106240863)
        record={'primary_doi':'10.7910/DVN/LKDIEK','license_from_primary_doi':'CC0-1.0','mirror_repo':repo,'revision':rev,'url':url,'sha256':h,'primary_mirror_equivalence':'not verified against inaccessible primary repository'}
        (out/'download_manifest.json').write_text(json.dumps(record,indent=2));print(record,flush=True);return
    native = args.dataset == 'c3vdv2_native'
    repo='BeileiCui/C3VDv2' if native else 'kangqi-ni/c3vdv2';meta=get_json('https://huggingface.co/api/datasets/'+repo);rev=meta['sha']
    out=ROOT/('datasets/external/C3VDv2_native_subset' if native else 'datasets/external/C3VDv2_subset');out.mkdir(parents=True,exist_ok=True)
    # Anatomies separated; fixed first texture where available. Clean/debris
    # pairs on c2 share anatomy/trajectory and must not count as independent subjects.
    scenes={'c1_cecum_t1_v2':'calibration','c1_sigmoid1_t1_v3':'validation',
            'c2_ascending_t4_v2':'test','c2_ascending_t4_v3':'test',
            'c2_transverse1_t3_v2':'test','c2_transverse1_t3_v3':'test',
            'c2_transverse2_t1_v2':'test','c2_transverse2_t1_v3':'test'}
    if native:
        scenes={'c1_ascending_t1_v1':'calibration','c1_ascending_t1_v2':'validation',
                'c2_ascending_t1_v3':'test','c2_cecum_t1_v3':'test',
                'c2_descending_t3_v3':'test','c2_rectum_t1_v3':'test',
                'c2_transverse1_t1_v3':'test','c2_transverse2_t1_v3':'test'}
    color_dir='rgb' if native else 'color'
    files=[];frames=[]
    for scene,split in scenes.items():
        tree=get_json(f'https://huggingface.co/api/datasets/{repo}/tree/{rev}/{scene}/{color_dir}?limit=1000')
        colors=sorted((x for x in tree if x['type']=='file'),key=lambda x:x['path'])
        depths=get_json(f'https://huggingface.co/api/datasets/{repo}/tree/{rev}/{scene}/depth?limit=1000')
        depth_by_id={Path(x['path']).stem.split('_')[0]:x for x in depths if x['type']=='file'}
        # Fixed 12 uniformly spaced frames per video. Selection uses only names.
        indices=[round(i*(len(colors)-1)/11) for i in range(12)]
        for i in indices:
            c=colors[i];fid=Path(c['path']).stem.split('_')[0];d=depth_by_id[fid]
            for x in (c,d):
                files.append({'path':x['path'],'size':x['size'],'expected_sha256':x.get('lfs',{}).get('oid')})
            frames.append({'scene':scene,'split':split,'frame_id':fid,'color_path':c['path'],'depth_path':d['path']})
        files.append({'path':scene+'/pose.txt','size':None,'expected_sha256':None})
    manifest={'primary_doi':'10.7281/T1/JC64MK','primary_url':'https://durrlab.github.io/C3VDv2/',
              'license_from_primary_doi':'CC-BY-4.0','mirror_repo':repo,'revision':rev,
              'selection':'12 uniform filename-order frames per fixed video, selected before running predictions',
              'primary_mirror_equivalence':'not verified bytewise against inaccessible Dataverse; encoding checked against primary documentation',
              'frames':frames,'files':files}
    (out/'selection.json').write_text(json.dumps(manifest,indent=2))
    def download(row):
        url=f'https://huggingface.co/datasets/{repo}/resolve/{rev}/'+quote(row['path'])
        return {**row,'sha256':fetch(url,out/row['path'],row['expected_sha256'],row['size']),'url':url}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        results=[]
        for i,row in enumerate(pool.map(download,files)):
            results.append(row)
            if i%20==0:print(f'C3VDv2 files {i+1}/{len(files)}',flush=True)
    manifest['files']=results;(out/'download_manifest.json').write_text(json.dumps(manifest,indent=2))
    print(f'Complete: {len(frames)} frames from {len(scenes)} scenes',flush=True)
if __name__=='__main__':main()
