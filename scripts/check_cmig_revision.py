"""Independent numerical and provenance checks of the CMIG extension."""
from cmig_revision_common import *
import cv2, torch, importlib.metadata
from evaluate_major_revision import reference

def main():
    p=check();parent=read(OLD/'training_protocol.json')
    for r in p['train_rows']:
        assert sha(R/r['rgb'])==p['training_rgb_sha256'][r['id']]
        assert sha(R/r['gt'])==parent['training_gt_sha256'][r['id']]
    for r in p['test_rows']:assert sha(R/r['rgb'])==p['test_rgb_sha256'][r['id']]
    oldcomplete=read(OLD/'training_complete.json')
    for name,h in oldcomplete['checkpoints'].items():assert sha(OLD/'trained_models'/name)==h
    complete=read(O/'training_complete.json');assert len(complete['checkpoints'])==9
    counts=[]
    for name,h in complete['checkpoints'].items():
        path=O/'trained_models'/name;assert sha(path)==h
        s=torch.load(path,map_location='cpu',weights_only=True)
        assert s['protocol_sha256']==sha(O/'protocol.json') and s['trainable_parameters']==2728513
        log=read(path.with_name(path.stem+'_log.json'));draws=log['draws']
        src=sum(x.startswith('source_') for batch in draws for x in batch)
        target=sum(x.startswith('target_') for batch in draws for x in batch)
        expected={'source_half':(600,0,600),'target_only':(0,1200,600),'adapted_double':(1200,1200,1200)}[s['regime']]
        assert (src,target,len(draws))==expected
        assert all(np.isfinite(x['loss']) for x in log['loss'])
        counts.append({'checkpoint':name,'source_draws':src,'target_draws':target,'steps':len(draws),'seconds':log['elapsed_seconds']})
    for seed in SEEDS:
        half=read(O/'trained_models'/f'source_half_{seed}_log.json')['draws']
        double=read(O/'trained_models'/f'adapted_double_{seed}_log.json')['draws']
        assert [x[0] for x in half]==[x[0] for x in double[:600]]
    supports=read(O/'calibration_support.json')
    assert sum(r['n'] for r in supports['source_only'])==16384
    for kind in ['source','target']:assert sum(r['n'] for r in supports['source_target'] if r['kind']==kind)==8192
    hh=read(O/'calibration_heads.json');assert len(hh['heads'])==8 and not hh['test_reference_used']
    for h in hh['heads'].values():assert h['n_samples']==16384 and h['lambda']==0
    for name,gkeys in [('comparison',['cohort','method','seed','bound']),('depth_strata',['cohort','method','seed','bound','bin']),('input_factors',['cohort','method','seed','factor'])]:
        rr=list(csv.DictReader((O/(name+'_per_frame.csv')).open()));ss=read(O/(name+'_summary.json'))
        keys=[tuple(r[k] for k in gkeys+['frame']) for r in rr];assert len(set(keys))==len(keys)
        for s in ss:
            a=[r for r in rr if all(r[k]==str(s[k]) for k in gkeys)]
            objects=sorted({r['object'] for r in a})
            val=np.mean([np.mean([float(r['mae_mm']) for r in a if r['object']==obj]) for obj in objects])
            assert np.isclose(val,s['mae_mm'],rtol=1e-12) and len(objects)==s['groups']
        assert all(np.isfinite(float(r['mae_mm'])) for r in rr)
    comp=list(csv.DictReader((O/'comparison_per_frame.csv').open()))
    assert not any(r['cohort']=='scared' and r['method'].startswith('pcc_') for r in comp)
    # A different implementation directly recomputes MAE from native pixels for every neural model,
    # and for each calibration head, on the first frame of every cohort.
    representatives={}
    for r in p['test_rows']:representatives.setdefault(cohort(r),r)
    verified=0
    for c,r in representatives.items():
        gt,v=reference(r);y,x=np.indices(gt.shape,dtype=float)
        rad=((x-gt.shape[1]/2)**2+(y-gt.shape[0]/2)**2)/((gt.shape[1]/2)**2+(gt.shape[0]/2)**2)
        for row in [a for a in comp if a['frame']==r['id']]:
            method=row['method'];seed=int(row['seed']);bound=int(row['bound'])
            if seed:
                path=OLD/'metric_predictions'/f'{method}_{seed}'/(r['id']+'_identity.npz') if method in ['source_only','adapted'] else O/'control_predictions'/f'{method}_{seed}'/(r['id']+'.npz')
                with np.load(path) as f:pred=cv2.resize(f['prediction'],gt.shape[::-1],interpolation=cv2.INTER_LINEAR)
            else:
                with np.load(O/'raw'/method.split('_')[0]/(r['id']+'.npz')) as f:d=cv2.resize(f['prediction'],gt.shape[::-1],interpolation=cv2.INTER_LINEAR).astype(float)
                a,b,c0,d0=hh['heads'][method]['coefficients_a0_a1_b0_b1'];inverse=(a+b*rad)*d+c0+d0*rad
                pred=np.full_like(inverse,float(bound));ok=inverse>0;pred[ok]=1/inverse[ok]
            pred=np.clip(pred,1,bound);mae=np.mean(np.abs(pred[v].astype(float)-gt[v]))
            assert np.isclose(mae,float(row['mae_mm']),rtol=1e-9);verified+=1
    # The new factorial identity should agree numerically with the original input pipeline.
    identity=[]
    for c,r in representatives.items():
        for regime in ['source_only','adapted']:
            for seed in SEEDS:
                with np.load(OLD/'metric_predictions'/f'{regime}_{seed}'/(r['id']+'_identity.npz')) as f:a=f['prediction'].copy()
                with np.load(O/'factor_predictions'/f'{regime}_{seed}'/(r['id']+'_identity.npz')) as f:b=f['prediction'].copy()
                identity.append(float(np.max(np.abs(a-b))))
    assert max(identity)<.01, max(identity)
    checks={'protocol_hashes_verified':True,'nine_new_and_six_old_checkpoints_verified':True,'training_rgb_gt_and_test_rgb_hashes_verified':True,'calibration_budget_verified':True,'pcc_scared_test_excluded':True,'all_summaries_reaggregated':True,'independent_native_metric_checks':verified,'identity_max_prediction_difference_mm':max(identity),'training_counts':counts,'new_independent_donors_or_device_serials_verified':False,'clinical_size_or_observer_agreement_validated':False}
    dump(O/'validation_checks.json',checks)
    dump(O/'environment.json',{'python':sys.executable,'version':sys.version,'packages':{name:importlib.metadata.version(name) for name in ['torch','transformers','numpy','opencv-python','Pillow','tifffile','matplotlib']},'gpu':torch.cuda.get_device_name(0)})
    print(json.dumps(checks,indent=2))
if __name__=='__main__':main()
