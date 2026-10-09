"""Read-only checks on frozen numerical evidence; writes a separate audit record."""
from pathlib import Path
import json,csv,hashlib,zlib,math,sys,platform,importlib.metadata as md
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/publication_study'
def read(n):return json.loads((OUT/n).read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rows(n):
 with (OUT/n).open() as f:return list(csv.DictReader(f))
checks={}
frozen=read('frozen_heads.json');start=read('final_evaluation_start.json')
checks['frozen_source_hashes']=all(sha(ROOT/p)==h for p,h in frozen['source_sha256'].items())
checks['frozen_head_hash']=sha(OUT/'frozen_heads.json')==start['frozen_heads_sha256']
checks['final_frame_manifest_hash']=sha(OUT/'final_frames.json')==start['frames_sha256']
checks['protocol_hash']=sha(OUT/'protocol.json')==frozen['protocol_sha256']
checks['outer_scene_isolation']=all(r['held_out_scene'] not in r['training_scenes'] and len(r['training_scenes'])==5 for r in read('outer_folds.json'))
checks['inner_folds']=len(rows('nested_inner_scores.csv'))==420 and all(r['outer_held']!=r['inner_held'] for r in rows('nested_inner_scores.csv'))
checks['final_gt_not_used_for_selection']=frozen['final_gt_used_for_selection'] is False
checks['budget']=frozen['budget_per_head']==16384
checks['final_members_hash_and_crc']=True
for scene in ['cecum_t1_a','desc_t4_a','sigmoid_t1_a']:
 for f in read(scene+'_manifest.json')['files']:
  b=(ROOT/f['path']).read_bytes()
  checks['final_members_hash_and_crc'] &= hashlib.sha256(b).hexdigest()==f['sha256'] and zlib.crc32(b)==f['zip_crc32'] and len(b)==f['bytes']
for phase,n,v in [('development',52,6),('final',36,3)]:
 frame=rows(phase+'_per_frame.csv');video=rows(phase+'_per_video.csv');summary=read(phase+'_summary.json')
 checks[phase+'_counts']=len(frame)==n*8 and len(video)==v*8 and len(summary)==8
 checks[phase+'_aggregation']=True
 for s in summary:
  for k in ['mae_mm','rmse_mm','abs_rel','delta1_fraction']:
   values=[float(r[k]) for r in video if r['method']==s['method']]
   checks[phase+'_aggregation'] &= math.isclose(sum(values)/len(values),s[k],abs_tol=1e-10)
 for r in video:
  for k in ['mae_mm','rmse_mm','abs_rel','delta1_fraction']:
   values=[float(x[k]) for x in frame if x['scene']==r['scene'] and x['method']==r['method']]
   checks[phase+'_aggregation'] &= math.isclose(sum(values)/len(values),float(r[k]),abs_tol=1e-10)
checks['pdf_exists']=(ROOT/'manuscript/calibration_generalization.pdf').stat().st_size>10000
report={'checks':checks,'all_passed':all(checks.values()),'scope':'Artifact integrity and aggregation; not independent re-inference or proof of all methodological claims'}
(OUT/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
aux={d.metadata['Name']:d.version for d in md.distributions(path=['/tmp/lesion_publication_deps'])}
(OUT/'aux_requirements.txt').write_text(''.join(f'{k}=={v}\n' for k,v in sorted(aux.items())))
env={'python':sys.executable,'python_version':platform.python_version(),'packages':{d.metadata['Name']:d.version for d in md.distributions() if d.metadata['Name']},'auxiliary_target':'/tmp/lesion_publication_deps','auxiliary_packages':aux}
(OUT/'environment.json').write_text(json.dumps(env,indent=2)+'\n')
print(json.dumps(report,indent=2))
if not report['all_passed']:raise SystemExit(1)
