"""One frozen exploratory trial; old test data are already observed, not a new holdout."""
import argparse, json, sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.calibrated_depth_measurement import MonotoneDepthCalibration
from c3vdv2_external_depth_validation import read_reference, metrics, sha, dump, csv_write, predictors

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,default=ROOT/'outputs/calibrated_depth_candidate')
    args=ap.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=False)
    data=ROOT/'datasets/external/C3VDv2_native_subset'
    previous=ROOT/'outputs/c3vdv2_external_validation'
    frames=json.loads((data/'download_manifest.json').read_text())['frames']
    weights=[0.,.25,.5,.75,1.]
    sources=[Path(__file__),ROOT/'src/calibrated_depth_measurement.py']
    dump(out/'protocol.json',{'candidate':'64 quantile-bin medians + weighted monotone regression; convex depth fusion',
        'fit_split':'calibration only','selection_split':'validation only, frame-mean MAE',
        'weights_neural':weights,'depth_range_mm':[1,100],
        'test_status':'EXPLORATORY: previously observed c2 anatomy, not a new independent holdout',
        'calibration_samples_per_frame':8192,'network_training':False,
        'source_sha256':{str(p.relative_to(ROOT)):sha(p) for p in sources},
        'data_manifest_sha256':sha(data/'download_manifest.json'),
        'previous_manifest_sha256':sha(previous/'manifest.json'),
        'command':[sys.executable,*sys.argv]})
    # Authenticate cached predictions and references before fitting or evaluation.
    hashes=json.loads((previous/'manifest.json').read_text())['outputs_sha256']
    for row in json.loads((data/'download_manifest.json').read_text())['files']:
        if sha(data/row['path']) != row['sha256']:raise ValueError('Changed data')
    for row in frames:
        name=f"raw_predictions/{row['scene']}_{row['frame_id']}.npz"
        if sha(previous/name)!=hashes[name]:raise ValueError('Changed prediction cache')
    def load(row):
        cache=np.load(previous/'raw_predictions'/f"{row['scene']}_{row['frame_id']}.npz")
        gt,valid=read_reference(data/row['depth_path'])
        return cache['photometric'],cache['relative_depth_anything'],gt,valid
    samples=[]
    for row in frames:
        if row['split']!='calibration':continue
        p,n,g,v=load(row);idx=np.flatnonzero(v);idx=idx[np.linspace(0,len(idx)-1,8192,dtype=int)]
        samples.append(np.stack([p.ravel()[idx],n.ravel()[idx],g.ravel()[idx]]))
    p,n,g=np.concatenate(samples,axis=1)
    photo=MonotoneDepthCalibration.fit(p,g,increasing=True)
    neural=MonotoneDepthCalibration.fit(n,g,increasing=False)
    scores={w:[] for w in weights}
    for row in frames:
        if row['split']!='validation':continue
        p,n,g,v=load(row);p=photo.predict(p);n=neural.predict(n)
        for w in weights:scores[w].append(metrics(w*n+(1-w)*p,g,v)['mae_mm'])
    selected=min(weights,key=lambda w:np.mean(scores[w]))
    # Freeze learned mapping and selected weight before loading any test GT.
    dump(out/'frozen_calibration.json',{'photometric':photo.to_dict(),'neural':neural.to_dict(),
        'neural_weight':selected,'validation_mae':{str(w):float(np.mean(scores[w])) for w in weights}})
    fit=json.loads((previous/'calibration.json').read_text());rows=[]
    for i,row in enumerate(frames):
        p,n,g,v=load(row);base,_=predictors(p,n,fit)
        candidates={'photometric_original':p,'neural_inverse_affine':base['depth_anything_v2_calibration_inverse_affine'],
            'monotone_neural':neural.predict(n),'monotone_photometric':photo.predict(p)}
        candidates['selected_fusion']=selected*candidates['monotone_neural']+(1-selected)*candidates['monotone_photometric']
        for method,pred in candidates.items():
            rows.append({**row,'method':method,**metrics(np.clip(pred,1,100),g,v)})
        if (i+1)%12==0:print(f'Scored {i+1}/{len(frames)}',flush=True)
    csv_write(out/'per_frame.csv',rows)
    videos=[]
    for scene in sorted({r['scene'] for r in rows}):
        for method in candidates:
            group=[r for r in rows if r['scene']==scene and r['method']==method]
            videos.append({'scene':scene,'split':group[0]['split'],'method':method,
                **{k:float(np.mean([r[k] for r in group])) for k in ['mae_mm','rmse_mm','abs_rel','delta1_fraction']}})
    csv_write(out/'per_video.csv',videos)
    summary=[]
    for split in ['calibration','validation','test']:
        for method in candidates:
            group=[r for r in videos if r['split']==split and r['method']==method]
            summary.append({'split':split,'method':method,**{k:float(np.mean([r[k] for r in group])) for k in ['mae_mm','rmse_mm','abs_rel','delta1_fraction']}})
    dump(out/'summary.json',summary)
    differences=[]
    for scene in sorted({r['scene'] for r in videos if r['split']=='test'}):
        m={r['method']:r['mae_mm'] for r in videos if r['scene']==scene}
        differences.append(m['selected_fusion']-m['neural_inverse_affine'])
    rng=np.random.default_rng(20260926)
    ci=np.quantile(rng.choice(differences,(10000,len(differences))).mean(axis=1),[.025,.975])
    dump(out/'paired_diagnostic.json',{'candidate_minus_neural_mm':float(np.mean(differences)),
        'sequence_bootstrap_95ci':ci.tolist(),'interpretation':'Descriptive only; one anatomy, old observed test'})
    dump(out/'manifest.json',{str(p.relative_to(out)):sha(p) for p in out.iterdir() if p.is_file()})
    print(json.dumps([r for r in summary if r['split']=='test'],indent=2))
if __name__=='__main__':main()
