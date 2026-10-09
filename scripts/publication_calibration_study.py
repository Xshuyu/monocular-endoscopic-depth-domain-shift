"""Nested scene CV, equal-budget coverage ablation, and one locked final test."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import sys,json,csv,hashlib,argparse,time,itertools
from pathlib import Path
import numpy as np,cv2
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tests'))
from src.robust_metric_depth import RobustMetricSpline,depth_feature
from src.calibrated_depth_measurement import MonotoneDepthCalibration
from c3vdv2_external_depth_validation import read_reference,metrics,sha,dump,csv_write
OUT=ROOT/'outputs/publication_study'
SPECS={'affine':{'kind':'affine'},'monotone':{'kind':'monotone'}}
for mode,reg,temp in itertools.product(['raw','frame_quantile'],[0.,.02,.2],[None,2.]):
 SPECS[f'{mode}_s{reg:g}_{"mean" if temp is None else "dro"}']={'kind':'spline','mode':mode,'smoothness':reg,'temperature':temp}
BUDGET=16384

def prediction(row,phase,backbone):
 with np.load(OUT/'predictions'/phase/backbone/f"{row['scene']}_{row['frame_id']}.npz") as f:return f['prediction'].copy()

def fit_head(spec,data,backbone='small'):
 x=data[backbone];y=data['gt'];groups=data['groups'];kind=spec['kind']
 state={'kind':kind,'backbone':backbone,'support':[float(x.min()),float(x.max())]}
 if kind=='affine':
  _,i,c=np.unique(groups,return_inverse=True,return_counts=True);w=np.sqrt(1/c[i])
  a,b=np.linalg.lstsq(np.column_stack([x,np.ones(len(x))])*w[:,None],1/y*w,rcond=None)[0]
  state.update({'a':float(a),'b':float(b)})
 elif kind=='monotone':state['model']=MonotoneDepthCalibration.fit(x,y,increasing=False).to_dict()
 elif kind=='constant':state['median_mm']=float(np.median(y))
 else:
  mode=spec['mode'];features=data['small_norm'] if mode=='frame_quantile' else x
  model=RobustMetricSpline.fit(features,y,groups,mode=mode,smoothness=spec['smoothness'],scene_temperature=spec['temperature'])
  state['model']=model.to_dict();state['support']=[float(features.min()),float(features.max())]
 return state

def apply_head(head,raw,normalized=None):
 kind=head['kind']
 if kind=='affine':
  d=head['a']*raw+head['b'];return np.clip(np.divide(1.,d,out=np.full_like(raw,100.),where=d>0),1,100)
 if kind=='constant':return np.full_like(raw,head['median_mm'])
 if kind=='photo':return np.clip(raw,1,100)
 model=head['model']
 if kind=='monotone':return np.clip(np.interp(raw,model['x'],model['y']),1,100)
 x=normalized if model['mode']=='frame_quantile' else raw
 if x is None:x=depth_feature(raw,'frame_quantile')
 return np.clip(np.interp(x,model['knots'],model['values']),1,100)

def prepare_samples(frames):
 samples={}
 for row in frames:
  gt,valid=read_reference(Path(row['depth_path']));indices=np.flatnonzero(valid)
  idx=indices[np.linspace(0,len(indices)-1,8192,dtype=int)]
  d={'gt':gt.ravel()[idx].astype(float)}
  for name in ['small','large','endodac']:
   p=prediction(row,'development',name)
   if p.shape!=gt.shape:raise ValueError('Prediction/reference shape mismatch')
   d[name]=p.ravel()[idx].astype(float)
   if name=='small':d['small_norm']=depth_feature(p,'frame_quantile').ravel()[idx]
  samples[(row['scene'],row['frame_id'])]=d
 return samples

def balanced_samples(samples,scenes):
 scenes=sorted(scenes);blocks=[];groups=[]
 for i,scene in enumerate(scenes):
  rows=[v for (s,f),v in samples.items() if s==scene];joined={k:np.concatenate([r[k] for r in rows]) for k in rows[0]}
  n=BUDGET//len(scenes)+(i<BUDGET%len(scenes));idx=np.linspace(0,len(joined['gt'])-1,n,dtype=int)
  blocks.append({k:v[idx] for k,v in joined.items()});groups.extend([scene]*n)
 result={k:np.concatenate([b[k] for b in blocks]) for k in blocks[0]};result['groups']=np.array(groups)
 if len(result['gt'])!=BUDGET:raise ValueError('Calibration budget mismatch')
 return result

def sampled_score(head,samples,scene):
 errors=[]
 for (s,f),d in samples.items():
  if s!=scene:continue
  pred=apply_head(head,d[head['backbone']],d['small_norm'])
  errors.append(float(np.abs(pred-d['gt']).mean()))
 return float(np.mean(errors))

def evaluate_full(frames,phase,heads_by_scene):
 rows=[];strata=[]
 for j,row in enumerate(frames):
  g,v=read_reference(Path(row['depth_path']));maps={k:prediction(row,phase,k) for k in ['small','large','endodac','photo']}
  rgb=cv2.cvtColor(cv2.imread(row['color_path']),cv2.COLOR_BGR2RGB)
  bright=(rgb.min(axis=-1)>=220)&(rgb.max(axis=-1)>=245);dark=rgb.max(axis=-1)<20
  masks={'near_0_10mm':v&(g<=10),'mid_10_30mm':v&(g>10)&(g<=30),'far_30_100mm':v&(g>30),
         'bright_neutral_proxy':v&bright,'dark_proxy':v&dark,'other_rgb':v&~bright&~dark}
  for method,head in heads_by_scene[row['scene']].items():
   raw=maps[head['backbone']];pred=apply_head(head,raw)
   support=head.get('support');feature=depth_feature(raw,'frame_quantile') if head.get('model',{}).get('mode')=='frame_quantile' else raw
   ood=float(((feature<support[0])|(feature>support[1]))[v].mean()) if support is not None else None
   rows.append({'scene':row['scene'],'frame':row['frame_id'],'method':method,**metrics(pred,g,v),
     'valid_pixels':int(v.sum()),'outside_calibration_support_fraction':ood,'at_output_bound_fraction':float(((pred<=1)|(pred>=100))[v].mean())})
   for region,mask in masks.items():
    if mask.any():strata.append({'scene':row['scene'],'frame':row['frame_id'],'method':method,'stratum':region,'pixels':int(mask.sum()),'mae_mm':float(np.abs(pred[mask].astype(float)-g[mask]).mean())})
  if (j+1)%8==0:print('Full-resolution scoring',phase,j+1,len(frames),flush=True)
 return rows,strata

def summarize(rows):
 videos=[]
 for scene in sorted({r['scene'] for r in rows}):
  for method in sorted({r['method'] for r in rows}):
   group=[r for r in rows if r['scene']==scene and r['method']==method]
   videos.append({'scene':scene,'method':method,'n_frames':len(group),**{k:float(np.mean([r[k] for r in group])) for k in ['mae_mm','rmse_mm','abs_rel','delta1_fraction']}})
 summary=[]
 for method in sorted({r['method'] for r in rows}):
  group=[r for r in videos if r['method']==method]
  summary.append({'method':method,'n_videos':len(group),**{k:float(np.mean([r[k] for r in group])) for k in ['mae_mm','rmse_mm','abs_rel','delta1_fraction']}})
 return videos,summary

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--phase',choices=['development','final'],required=True);args=ap.parse_args()
 sources=[Path(__file__),ROOT/'src/robust_metric_depth.py',ROOT/'src/calibrated_depth_measurement.py',ROOT/'scripts/publication_inference.py']
 if args.phase=='development':
  if (OUT/'frozen_heads.json').exists():raise ValueError('Already frozen; refusing overwrite')
  frames=json.loads((OUT/'development_frames.json').read_text());scenes=sorted({r['scene'] for r in frames});samples=prepare_samples(frames)
  dump(OUT/'cv_execution_protocol.json',{'source_sha256':{str(p.relative_to(ROOT)):sha(p) for p in sources},'budget':BUDGET,'candidate_specs':SPECS,'scenes':scenes,'final_labels_used':False})
  memo={};cv_rows=[];outer=[];outer_heads={}
  def trained(train,name):
   key=(tuple(sorted(train)),name)
   if key not in memo:memo[key]=fit_head(SPECS[name],balanced_samples(samples,train))
   return memo[key]
  for held in scenes:
   training=[s for s in scenes if s!=held];scores={}
   for name in SPECS:
    losses=[]
    for inner in training:
     head=trained([s for s in training if s!=inner],name);loss=sampled_score(head,samples,inner);losses.append(loss)
     cv_rows.append({'outer_held':held,'inner_held':inner,'candidate':name,'mae_mm':loss})
    scores[name]=float(np.mean(losses))
   selected=min(scores,key=scores.get);data=balanced_samples(samples,training)
   heads={'small_affine':trained(training,'affine'),'small_monotone':trained(training,'monotone'),'nested_selected':trained(training,selected),
    'large_affine':fit_head({'kind':'affine'},data,'large'),'endodac_affine':fit_head({'kind':'affine'},data,'endodac'),
    'constant_median':fit_head({'kind':'constant'},data),'photometric_raw':{'kind':'photo','backbone':'photo'},
    'small_single_scene_affine':fit_head({'kind':'affine'},balanced_samples(samples,[training[0]]))}
   outer_heads[held]=heads;outer.append({'held_out_scene':held,'training_scenes':training,'single_scene_anchor':training[0],'selected':selected,'inner_scores':scores})
   print('Outer fold',held,'selected',selected,'inner MAE',scores[selected],flush=True)
  csv_write(OUT/'nested_inner_scores.csv',cv_rows);dump(OUT/'outer_folds.json',outer);dump(OUT/'outer_heads.json',outer_heads)
  rows,strata=evaluate_full(frames,'development',outer_heads)
  videos,summary=summarize(rows);csv_write(OUT/'development_per_frame.csv',rows);csv_write(OUT/'development_strata.csv',strata);csv_write(OUT/'development_per_video.csv',videos);dump(OUT/'development_summary.json',summary)
  # Final choice uses six development-only leave-scene-out scores. No final GT.
  final_scores={name:float(np.mean([sampled_score(trained([s for s in scenes if s!=held],name),samples,held) for held in scenes])) for name in SPECS}
  selected=min(final_scores,key=final_scores.get);all_data=balanced_samples(samples,scenes)
  heads={'small_affine':fit_head(SPECS['affine'],all_data),'small_monotone':fit_head(SPECS['monotone'],all_data),'selected_calibration':fit_head(SPECS[selected],all_data),
    'large_affine':fit_head({'kind':'affine'},all_data,'large'),'endodac_affine':fit_head({'kind':'affine'},all_data,'endodac'),
    'constant_median':fit_head({'kind':'constant'},all_data),'photometric_raw':{'kind':'photo','backbone':'photo'},
    'small_single_scene_affine':fit_head({'kind':'affine'},balanced_samples(samples,[scenes[0]]))}
  frozen={'selected_candidate':selected,'development_loso_scores':final_scores,'heads':heads,'calibration_scenes':scenes,'budget_per_head':BUDGET,
   'source_sha256':{str(p.relative_to(ROOT)):sha(p) for p in sources},'protocol_sha256':sha(OUT/'protocol.json'),'final_gt_used_for_selection':False}
  dump(OUT/'frozen_heads.json',frozen);print('FROZEN FINAL HEAD',selected,flush=True)
 else:
  if (OUT/'final_summary.json').exists():raise ValueError('Final already evaluated; refusing overwrite')
  frozen=json.loads((OUT/'frozen_heads.json').read_text())
  for name,h in frozen['source_sha256'].items():
   if sha(ROOT/name)!=h:raise ValueError('Code changed after final freeze')
  frames=json.loads((OUT/'final_frames.json').read_text());heads_by_scene={r['scene']:frozen['heads'] for r in frames}
  dump(OUT/'final_evaluation_start.json',{'frozen_heads_sha256':sha(OUT/'frozen_heads.json'),'frames_sha256':sha(OUT/'final_frames.json'),'command':[sys.executable,*sys.argv]})
  rows,strata=evaluate_full(frames,'final',heads_by_scene);videos,summary=summarize(rows)
  csv_write(OUT/'final_per_frame.csv',rows);csv_write(OUT/'final_strata.csv',strata);csv_write(OUT/'final_per_video.csv',videos);dump(OUT/'final_summary.json',summary)
  print(json.dumps(summary,indent=2),flush=True)
if __name__=='__main__':main()
