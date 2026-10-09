"""Post-hoc development sampling sensitivity; never selects on test results."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
import sys,json,csv,hashlib,datetime
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from run_radial_confirmatory import sample,fit,predict,score
OUT=ROOT/'outputs/radial_sampling_sensitivity'
BASE=ROOT/'outputs/radial_confirmatory'
METHODS=['affine_ls','affine_ridge','radial_ls','radial_ridge']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def csvout(p,rows):
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
 OUT.mkdir(exist_ok=False)
 frozen=json.loads((BASE/'frozen_heads.json').read_text())
 assert all(sha(ROOT/p)==h for p,h in frozen['source_sha256'].items())
 protected={str(p.relative_to(ROOT)):sha(p) for p in BASE.glob('*.json')}
 protected.update({str(p.relative_to(ROOT)):sha(p) for p in BASE.glob('*.csv')})
 protocol={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'Post-hoc development-only sampling sensitivity, NOT new external validation','budgets':[4096,16384,65536],'seeds':list(range(202609280,202609290)),'folds':'Existing six outer scene folds','regularization':'Original outer-fold selected lambda held fixed; not repeating nested selection','sampling':'Uniform without replacement within each training scene from original 8192-valid-pixel/frame pool; equal scene quotas; identical pixels across families; nested subsets across budgets for each seed/fold','evaluation':'Original deterministic 8192 valid pixels/frame in held-out development scenes; mean frame then equal scene MAE; NOT full-native primary endpoint','limits':'Seed SD describes pixel sampling only, conditional on fixed scenes, cached predictions, lambda and pixel pool. No backbone repetitions or independent patient replication. Test results used only for leave-one-video-out descriptive arithmetic.','source_sha256':{str(Path(__file__).relative_to(ROOT)):sha(Path(__file__)),**frozen['source_sha256']}}
 dump(OUT/'protocol.json',protocol)
 frames=json.loads((ROOT/'outputs/publication_study/development_frames.json').read_text())
 pool=sample(frames);scenes=sorted({r['scene'] for r in frames});outer=json.loads((BASE/'outer_heads.json').read_text())
 grouped={s:{k:np.concatenate([a[k] for (ss,_),a in pool.items() if ss==s]) for k in ['d','r','z']} for s in scenes}
 rows=[];heads=[]
 for seed in protocol['seeds']:
  for held in scenes:
   train=[s for s in scenes if s!=held]
   rng=np.random.default_rng(np.random.SeedSequence([seed,scenes.index(held)]))
   order={s:rng.permutation(len(grouped[s]['d'])) for s in train}
   for budget in protocol['budgets']:
    ids={s:order[s][:budget//len(train)+(i<budget%len(train))] for i,s in enumerate(train)}
    a={k:np.concatenate([grouped[s][k][ids[s]] for s in train]) for k in ['d','r','z']}
    groups=np.concatenate([np.repeat(s,len(ids[s])) for s in train])
    assert len(groups)==budget and held not in groups
    for m in METHODS:
     lam=outer[held][m]['lambda'];h=fit(a['d'],a['r'],a['z'],groups,m.startswith('radial'),lam)
     rows.append({'seed':seed,'held':held,'budget':budget,'method':m,'lambda':lam,'sampled_mae_mm':score(h,pool,held)})
     heads.append({'seed':seed,'held':held,'budget':budget,'method':m,'head':h})
  print('Completed sampling seed',seed,flush=True)
 csvout(OUT/'fold_results.csv',rows);dump(OUT/'heads.json',heads)
 repeats=[];summary=[]
 for b in protocol['budgets']:
  for m in METHODS:
   vals=[]
   for seed in protocol['seeds']:
    x=[r['sampled_mae_mm'] for r in rows if r['budget']==b and r['method']==m and r['seed']==seed];assert len(x)==6
    val=float(np.mean(x));vals.append(val);repeats.append({'budget':b,'method':m,'seed':seed,'sampled_mae_mm':val})
   summary.append({'budget':b,'method':m,'mean_sampled_mae_mm':float(np.mean(vals)),'seed_sd_mm':float(np.std(vals,ddof=1)),'min_mm':min(vals),'max_mm':max(vals)})
 csvout(OUT/'seed_results.csv',repeats);dump(OUT/'summary.json',summary)
 paired=[]
 for b in protocol['budgets']:
  for a,c in [('radial_ls','affine_ls'),('radial_ridge','affine_ls'),('radial_ridge','radial_ls')]:
   ds=[]
   for seed in protocol['seeds']:
    get=lambda m:next(r['sampled_mae_mm'] for r in repeats if r['budget']==b and r['method']==m and r['seed']==seed)
    ds.append(get(a)-get(c))
   paired.append({'budget':b,'a':a,'b':c,'mean_delta_mm':float(np.mean(ds)),'seed_sd_mm':float(np.std(ds,ddof=1)),'min_delta_mm':min(ds),'max_delta_mm':max(ds),'improved_seeds':sum(d<0 for d in ds)})
 dump(OUT/'paired_sampling.json',paired)
 with (BASE/'test_per_video.csv').open() as f:test=list(csv.DictReader(f))
 vids=sorted({r['scene'] for r in test});lookup={(r['scene'],r['method']):float(r['mae_mm']) for r in test}
 influence=[]
 for m in ['radial_ls','radial_ridge']:
  for omit in vids:
   delta=float(np.mean([lookup[s,m]-lookup[s,'affine_ls'] for s in vids if s!=omit]))
   influence.append({'method':m,'omitted_video':omit,'remaining_videos':3,'delta_vs_affine_mm':delta})
 csvout(OUT/'test_video_influence.csv',influence)
 checks={'720_fits':len(rows)==720,'all_training_groups_exclude_held':all(h['held'] not in h['head']['training_scenes'] for h in heads),'unchanged_primary_artifacts':all(sha(ROOT/p)==h for p,h in protected.items()),'finite_scores':all(np.isfinite(r['sampled_mae_mm']) for r in rows)}
 dump(OUT/'validation.json',checks);assert all(checks.values())
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,ax=plt.subplots(figsize=(7,4.2))
 for m,label in zip(METHODS,['Affine LS','Affine ridge','Radial LS','Radial ridge']):
  rr=[r for r in summary if r['method']==m]
  ax.errorbar([r['budget'] for r in rr],[r['mean_sampled_mae_mm'] for r in rr],yerr=[r['seed_sd_mm'] for r in rr],marker='o',capsize=4,label=label,linestyle='--' if m.endswith('ridge') else '-')
 ax.set_xscale('log',base=2);ax.set_xticks(protocol['budgets'],[str(x) for x in protocol['budgets']]);ax.set_xlabel('Total training pixels per head');ax.set_ylabel('Held-out development sampled MAE (mm)');ax.legend(ncol=2);ax.grid(alpha=.2)
 fig.text(.5,.01,'Mean ± seed SD; 10 sampling seeds, fixed fold-specific regularization.',ha='center',fontsize=9);fig.tight_layout(rect=(0,.04,1,1))
 for ext in ['pdf','svg','png']:fig.savefig(OUT/f'sampling_sensitivity.{ext}',dpi=180)
 plt.close(fig)
 print(json.dumps({'summary':summary,'paired':paired,'test_influence':influence,'checks':checks},indent=2),flush=True)
if __name__=='__main__':main()
