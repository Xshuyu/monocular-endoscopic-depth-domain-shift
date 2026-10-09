"""Analyze real repeated-rater records. Empty/incomplete templates are rejected."""
from pathlib import Path
import argparse,csv,json,sys,itertools
import numpy as np
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from src.observer_agreement import absolute_agreement_icc,cluster_bootstrap_icc

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--rater-scope',choices=['fixed_panel','random_raters'],required=True);a=p.parse_args()
 rows=list(csv.DictReader(a.input.open()));required=['patient_id','object_id','rater_id','session_id','measurement_mm','endpoint']
 if not rows:raise ValueError('No real observer measurements provided; ICC cannot be computed')
 if any(not r.get(k,'').strip() for r in rows for k in required):raise ValueError('Missing identifiers, measurements, or endpoint')
 if len({r['endpoint'] for r in rows})!=1:raise ValueError('Do not mix measurement endpoints')
 index={}
 for row in rows:
  key=(row['object_id'],row['rater_id'],row['session_id'])
  if key in index:raise ValueError('Duplicate object/rater/session record')
  v=float(row['measurement_mm'])
  if not np.isfinite(v) or v<=0:raise ValueError('Measurements must be finite positive millimetres')
  index[key]=v
 objects=sorted({r['object_id'] for r in rows});raters=sorted({r['rater_id'] for r in rows});sessions=sorted({r['session_id'] for r in rows})
 if len(raters)<2:raise ValueError('Need at least two raters')
 if len(index)!=len(objects)*len(raters)*len(sessions):raise ValueError('Incomplete crossed panel; no silent imputation or complete-case deletion')
 groups=[]
 for obj in objects:
  ids={r['patient_id'] for r in rows if r['object_id']==obj}
  if len(ids)!=1:raise ValueError('An object maps to inconsistent patients')
  groups.append(next(iter(ids)))
 result={'endpoint':rows[0]['endpoint'],'rater_scope':a.rater_scope,'formula':'Two-way absolute agreement ICC(A,1) and ICC(A,k); random-rater interpretation additionally requires justified random rater sampling','interrater_by_session':[],'intrarater_by_rater':[],'limits':'Agreement assesses reference repeatability, not algorithm accuracy. ICC(A,k) applies to the rater mean, NOT the maximum measurement.'}
 def analyze(x):return {**absolute_agreement_icc(x),**cluster_bootstrap_icc(x,groups)}
 for session in sessions:
  x=np.array([[index[o,r,session] for r in raters] for o in objects]);pairs=[]
  for i,j in itertools.combinations(range(len(raters)),2):
   d=x[:,i]-x[:,j];sd=d.std(ddof=1);pairs.append({'rater_a':raters[i],'rater_b':raters[j],'bias_mm':float(d.mean()),'descriptive_limits_of_agreement_mm':[float(d.mean()-1.96*sd),float(d.mean()+1.96*sd)],'note':'Object differences, descriptive normal-approximation limits; no independent-row confidence interval'})
  result['interrater_by_session'].append({'session':session,**analyze(x),'pairwise_bias':pairs})
 if len(sessions)>1:
  for rater in raters:
   x=np.array([[index[o,rater,s] for s in sessions] for o in objects]);result['intrarater_by_rater'].append({'rater':rater,**analyze(x),'scope':'Occasion agreement for this rater, not interrater ICC'})
 if a.output.exists():raise ValueError('Refusing overwrite')
 a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print('Saved',a.output)
if __name__=='__main__':main()
