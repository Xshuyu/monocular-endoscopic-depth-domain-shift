"""Fixed additional development videos and one new evaluation video, chosen before predictions."""
import sys,json,concurrent.futures
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from download_external_validation_data import get_json,fetch
repo='BeileiCui/C3VDv2';rev='80ff074dd81d7b9c4ae4e74c264b494338f5f3d1'
out=ROOT/'datasets/external/C3VDv2_robust_development';out.mkdir(parents=True,exist_ok=True)
scenes={'c1_cecum_t1_v3':'fit','c1_rectum_t1_v3':'fit','c1_sigmoid1_t1_v3':'fit','c1_transverse1_t1_v3':'validation','c1_transverse2_t1_v3':'validation','c2_sigmoid_t1_v3':'new_sequence_test'}
frames=[];files=[]
for scene,split in scenes.items():
 colors=sorted([x for x in get_json(f'https://huggingface.co/api/datasets/{repo}/tree/{rev}/{scene}/rgb?limit=1000') if x['type']=='file'],key=lambda x:x['path'])
 depths={Path(x['path']).stem.split('_')[0]:x for x in get_json(f'https://huggingface.co/api/datasets/{repo}/tree/{rev}/{scene}/depth?limit=1000') if x['type']=='file'}
 for i in [round(k*(len(colors)-1)/7) for k in range(8)]:
  color=colors[i];fid=Path(color['path']).stem.split('_')[0];depth=depths[fid]
  frames.append({'scene':scene,'split':split,'frame_id':fid,'color_path':color['path'],'depth_path':depth['path']})
  for x in (color,depth):files.append({'path':x['path'],'size':x['size'],'expected_sha256':x.get('lfs',{}).get('oid')})
manifest={'repo':repo,'revision':rev,'frames':frames,'files':files,'selection':'8 uniform filename indices per preselected video, before pixels or labels','existing_fit_addition':'12 old c1_ascending_t1_v1 calibration frames','limits':'New test video, already observed anatomy c2; not new patients. Mirror origin unverified against primary bytes.'}
selection=out/'selection.json'
if selection.exists() and json.loads(selection.read_text())!=manifest:raise ValueError('Selection changed')
selection.write_text(json.dumps(manifest,indent=2)+'\n')
def download(row):
 url=f"https://huggingface.co/datasets/{repo}/resolve/{rev}/{row['path']}"
 return {**row,'sha256':fetch(url,out/row['path'],row['expected_sha256'],row['size'])}
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:downloaded=list(pool.map(download,files))
manifest['files']=downloaded;(out/'download_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('Ready',len(frames),'frames',flush=True)
