"""Artifact, split, provenance and arithmetic checks without modifying frozen results."""
from pathlib import Path
import json,csv,hashlib,zlib,math,sys,platform,importlib.metadata as md,shutil
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/radial_confirmatory';PUB=ROOT/'outputs/publication_study'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4<<20),b''):h.update(b)
 return h.hexdigest()
def read(n):return json.loads((OUT/n).read_text())
def csvread(n):
 with (OUT/n).open() as f:return list(csv.DictReader(f))
f=read('frozen_heads.json');start=read('final_evaluation_start.json');protocol=read('protocol.json');checks={}
checks['source_frozen']=all(sha(ROOT/n)==h for n,h in f['source_sha256'].items());checks['protocol_frozen']=sha(OUT/'protocol.json')==f['protocol_sha256'];checks['head_frozen']=sha(OUT/'frozen_heads.json')==start['frozen_heads_sha256'];checks['test_frames_frozen']=sha(OUT/'test_frames.json')==start['frames_sha256'];checks['no_test_selection_flag']=f['test_used_for_selection'] is False
checks['outer_group_exclusion']=all(x['held'] not in x['train'] and len(x['train'])==5 for x in read('outer_folds.json'))
outer=read('outer_heads.json');checks['fit_sample_budget']=all(h['n_samples']==16384 for heads in outer.values() for h in heads.values()) and all(f['heads'][m]['n_samples']==16384 for m in protocol['families'])
checks['outer_fitted_groups']=all(set(h['training_scenes'])==set(x['train']) for x in read('outer_folds.json') for h in outer[x['held']].values())
cv=csvread('nested_selection.csv');checks['inner_records']=len(cv)==540 and all(r['outer']!=r['inner'] for r in cv)
checks['new_members_valid']=True;checks['acquired_after_freeze']=True
for scene in protocol['new_test']:
 checks['acquired_after_freeze'] &= (OUT/(scene+'_selection.json')).stat().st_mtime >= (OUT/'frozen_heads.json').stat().st_mtime
 for rec in read(scene+'_manifest.json')['files']:
  b=(ROOT/rec['path']).read_bytes();checks['new_members_valid'] &= len(b)==rec['bytes'] and hashlib.sha256(b).hexdigest()==rec['sha256'] and zlib.crc32(b)==rec['zip_crc32']
checks['exact_rgb_overlap_absent']=read('duplicate_check.json')['overlap']==0
for phase,frames,methods,videos in [('development',52,4,6),('test',48,6,4)]:
 a=csvread(phase+'_per_frame.csv');b=csvread(phase+'_per_video.csv');c=read(phase+'_summary.json');checks[phase+'_counts']=len(a)==frames*methods and len(b)==videos*methods and len(c)==methods;checks[phase+'_aggregation']=True
 for row in b:
  group=[r for r in a if r['method']==row['method'] and r['scene']==row['scene']]
  for k in ['mae_mm','rmse_mm','abs_rel','delta1_fraction']:
   checks[phase+'_aggregation'] &= math.isclose(sum(float(r[k]) for r in group)/len(group),float(row[k]),abs_tol=1e-10)
 for row in c:
  group=[r for r in b if r['method']==row['method']]
  for k in ['mae_mm','rmse_mm','abs_rel','delta1_fraction']:
   checks[phase+'_aggregation'] &= math.isclose(sum(float(r[k]) for r in group)/len(group),float(row[k]),abs_tol=1e-10)
checks['test_scene_names']=set(r['scene'] for r in read('test_frames.json'))==set(protocol['new_test'])
report={'checks':checks,'all_passed':all(checks.values()),'limits':'No reconstruction of historical fit provenance, no near-duplicate or pretraining audit, no clinical validation'}
(OUT/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
(OUT/'environment.json').write_text(json.dumps({'python':sys.executable,'version':platform.python_version(),'packages':{d.metadata['Name']:d.version for d in md.distributions() if d.metadata['Name']}},indent=2)+'\n')
inputs={}
for row in json.loads((PUB/'development_frames.json').read_text()):
 for p in [Path(row['color_path']),Path(row['depth_path']),PUB/'predictions/development/small'/f"{row['scene']}_{row['frame_id']}.npz"]:inputs[str(p.relative_to(ROOT))]=sha(p)
(OUT/'development_input_hashes_postrun.json').write_text(json.dumps({'timing':'Post-run audit, not a retrospective claim of pre-fit hashing','sha256':inputs},indent=2)+'\n')
print(json.dumps(report,indent=2))
if not report['all_passed']:raise SystemExit(1)
