"""Native-mask scoring, group aggregation and depth-support controlled diagnostics."""
from cmig_revision_common import *
import argparse, cv2
from evaluate_major_revision import frames,reference
from src.radial_affine_calibration import radius_squared
from c3vdv2_external_depth_validation import metrics

KEYS=['mae_mm','rmse_mm','abs_rel','bias_mm','delta1_fraction']
def aggregate(rows,name,groupkeys):
    buckets={}
    for r in rows:buckets.setdefault(tuple(r[k] for k in groupkeys)+ (r['object'],),[]).append(r)
    objects=[]
    for key,rr in sorted(buckets.items()):
        objects.append({**dict(zip(groupkeys+['object'],key)),'frames':len(rr),'pixels':sum(r['pixels'] for r in rr),**{k:float(np.mean([r[k] for r in rr])) for k in KEYS}})
    buckets={}
    for r in objects:buckets.setdefault(tuple(r[k] for k in groupkeys),[]).append(r)
    summary=[{**dict(zip(groupkeys,key)),'groups':len(rr),'frames':sum(r['frames'] for r in rr),'pixels':sum(r['pixels'] for r in rr),**{k:float(np.mean([r[k] for r in rr])) for k in KEYS}} for key,rr in sorted(buckets.items())]
    writecsv(O/(name+'_per_frame.csv'),rows);dump(O/(name+'_per_object.json'),objects);dump(O/(name+'_summary.json'),summary)
    return summary

def neural_predictions(r,controls=True):
    for regime in ['source_only','adapted']+(REGIMES if controls else []):
        for seed in SEEDS:
            p=OLD/'metric_predictions'/f'{regime}_{seed}'/(r['id']+'_identity.npz') if regime in ['source_only','adapted'] else O/'control_predictions'/f'{regime}_{seed}'/(r['id']+'.npz')
            with np.load(p) as f:pred=f['prediction'].copy()
            yield regime,seed,pred

def score():
    check();hh=read(O/'calibration_heads.json');assert hh['protocol_sha256']==sha(O/'protocol.json')
    rows=[];strata=[];edges=[0,30,60,100,200,300]
    for r in frames():
        gt,v=reference(r);assert v.any();rad=radius_squared(gt.shape)
        predictions=[]
        for method,h in hh['heads'].items():
            backbone=method.split('_')[0]
            if backbone=='pcc' and r['cohort']=='scared':continue
            with np.load(O/'raw'/backbone/(r['id']+'.npz')) as f:d=cv2.resize(f['prediction'],gt.shape[::-1],interpolation=cv2.INTER_LINEAR).astype(float)
            a0,a1,b0,b1=h['coefficients_a0_a1_b0_b1'];inv=(a0+a1*rad)*d+b0+b1*rad
            raw=np.divide(1.,inv,out=np.full_like(inv,np.inf),where=inv>0)
            predictions.append((method,0,raw))
        # Score one model at a time; retain no dense prediction stack in the output.
        for method,seed,raw in list(neural_predictions(r))+predictions:
            raw=cv2.resize(raw,gt.shape[::-1],interpolation=cv2.INTER_LINEAR) if raw.shape!=gt.shape else raw
            for bound in [100,300]:
                pred=np.clip(raw,1,bound);assert np.isfinite(pred).all()
                base={'cohort':cohort(r),'object':str(r['object']),'frame':r['id'],'method':method,'seed':seed,'bound':bound}
                rows.append({**base,'pixels':int(v.sum()),**metrics(pred,gt,v)})
                for lo,hi in zip(edges[:-1],edges[1:]):
                    mask=v&(gt>lo)&(gt<=hi)
                    if mask.any():strata.append({**base,'bin':f'{lo}_{hi}','pixels':int(mask.sum()),**metrics(pred,gt,mask)})
        print('SCORE',r['id'],flush=True)
    aggregate(rows,'comparison',['cohort','method','seed','bound'])
    ss=aggregate(strata,'depth_strata',['cohort','method','seed','bound','bin'])
    standardized=[]
    for method,seed,bound in sorted({(x['method'],x['seed'],x['bound']) for x in ss}):
        a=[x for x in ss if (x['method'],x['seed'],x['bound'])==(method,seed,bound)]
        cohorts=sorted({x['cohort'] for x in a});bins=sorted(set.intersection(*[{x['bin'] for x in a if x['cohort']==c} for c in cohorts]))
        for c in cohorts:
            selected=[x for x in a if x['cohort']==c and x['bin'] in bins]
            standardized.append({'cohort':c,'method':method,'seed':seed,'bound':bound,'common_bins':bins,'equal_bin_mae_mm':float(np.mean([x['mae_mm'] for x in selected])) if selected else None,'bin_group_counts':{x['bin']:x['groups'] for x in selected},'interpretation':'Descriptive restricted support only; not primary performance or causal identification'})
    dump(O/'depth_standardized.json',standardized)

def factors():
    check();rows=[]
    for r in frames():
        gt,v=reference(r)
        for regime in ['source_only','adapted']:
            for seed in SEEDS:
                for factor in FACTORS:
                    with np.load(O/'factor_predictions'/f'{regime}_{seed}'/(r['id']+'_'+factor+'.npz')) as f:raw=cv2.resize(f['prediction'],gt.shape[::-1],interpolation=cv2.INTER_LINEAR)
                    pred=np.clip(raw,1,100)
                    rows.append({'cohort':cohort(r),'object':str(r['object']),'frame':r['id'],'method':regime,'seed':seed,'factor':factor,'pixels':int(v.sum()),**metrics(pred,gt,v)})
    ss=aggregate(rows,'input_factors',['cohort','method','seed','factor'])
    effects=[]
    for c,m,s in sorted({(r['cohort'],r['method'],r['seed']) for r in ss}):
        a={r['factor']:r['mae_mm'] for r in ss if (r['cohort'],r['method'],r['seed'])==(c,m,s)}
        e00,e10,e01,e11=[a[k] for k in ['identity','dark','blur','dark_blur']]
        effects.append({'cohort':c,'method':m,'seed':s,'dark_main_mm':((e10-e00)+(e11-e01))/2,'blur_main_mm':((e01-e00)+(e11-e10))/2,'interaction_mm':e11-e10-e01+e00,**a})
    dump(O/'input_factor_effects.json',effects)
    print('FACTOR EFFECTS',len(effects),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['score','factors']);a=p.parse_args();globals()[a.phase]()
