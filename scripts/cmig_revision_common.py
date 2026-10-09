"""Shared, frozen protocol utilities for the CMIG revision (no test selection)."""
import os
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
from pathlib import Path
import sys, json, csv, hashlib, datetime, zipfile
import numpy as np
R = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(R), str(R/'scripts'), str(R/'tests')]
O = R/'outputs/cmig_revision'
OLD = R/'outputs/major_revision'
SEEDS = [11, 23, 47]
REGIMES = ['source_half', 'target_only', 'adapted_double']
FACTORS = {'identity':(1.,0.), 'dark':(1.5,0.), 'blur':(1.,1.5), 'dark_blur':(1.5,1.5)}
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4<<20),b''): h.update(b)
    return h.hexdigest()
def dump(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def read(p): return json.loads(p.read_text())
def writecsv(p,rows):
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def cohort(r):
    return 'drends_'+r['setup'].lower() if r['cohort']=='drends' else r['cohort']
def codepaths():
    return [R/'scripts'/s for s in ['cmig_revision_common.py','run_cmig_controls.py','run_cmig_calibration.py','score_cmig_revision.py']]
def check():
    p=read(O/'protocol.json')
    for name,h in p['code_sha256'].items():
        assert sha(R/name)==h, 'Frozen analysis code changed: '+name
    assert sha(OLD/'training_protocol.json')==p['parent_protocol_sha256']
    return p
def freeze():
    from evaluate_major_revision import frames
    from train_major_revision_baseline import data
    assert not (O/'protocol.json').exists(), 'Refusing protocol overwrite'
    train=data();test=frames()
    trainhash={r['id']:sha(R/r['rgb']) for r in train}
    testhash={r['id']:sha(R/r['rgb']) for r in test}
    assert not set(trainhash.values())&set(testhash.values())
    assert len(test)==101 and len(train)==72
    O.mkdir(exist_ok=True)
    with zipfile.ZipFile(O/'manuscript_before_cmig_revision.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in (R/'manuscript_cmig').rglob('*'):
            if p.is_file():z.write(p,p.relative_to(R))
    dump(O/'protocol.json',{
      'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
      'status':'Post-hoc extension; all cohorts have previously been examined. Settings frozen before these new comparisons, not a preregistered blind confirmation.',
      'parent_protocol_sha256':sha(OLD/'training_protocol.json'),
      'code_sha256':{str(p.relative_to(R)):sha(p) for p in codepaths()},
      'training_rgb_sha256':trainhash,'test_rgb_sha256':testhash,
      'train_rows':train,'test_rows':test,
      'neural_controls':{
        'seeds':SEEDS,'optimizer':'Unchanged parent AdamW1e-4 wd0.01; final checkpoint only; same generic frozen backbone and decoder initialization',
        'source_half':'600 updates, batch1 source:600 source draws, same source exposure and updates as adapted but different batch size',
        'target_only':'600 updates, batch2 target:1200 target draws, same update and batch budget as source-only and adapted',
        'adapted_double':'1200 updates, batch2 (1 source+1 target):1200 source and1200 target draws; matches original source exposure but doubles updates',
        'sampling':'Separate source and target RNGs seeded identically per seed; first600 source_half draws equal adapted_double source draws; original controls retain their original RNG. No exact historical sample-order match claimed.'},
      'calibration':{
        'backbones':['Generic DA V2 Small','Official EndoDAC+PCC checkpoint'],
        'families':['affine_ls','radial_ls'],'lambda':0,
        'source_only':16384,'source_target':{'source':8192,'target':8192},
        'sampling':'Deterministic equal group then frame allocations; same pixel indices across both backbones; standardization on calibration pixels only. Equal domain weights, approximately equal object weights within domains.',
        'neural_comparison':'Same training images, not equal dense-label pixel budget, capacity or original pretraining compute.',
        'pcc_test_exclusion':'SCARED8/9 omitted because public PCC training list includes all nine SCARED objects.',
        'small_resolution':[224,280],'pcc_resolution':[256,320],
        'range_mm':[1,100],'secondary_range_mm':[1,300],
        'primary':'All valid native reference pixels; no per-test alignment, no clipping reference depths'},
      'controlled_factors':{'models':'Existing source_only/adapted, all three seeds','factors':FACTORS,'implementation':'Resize RGB to224x280 using official processor with normalization/rescaling disabled; then gamma and Gaussian sigma1.5 INPUT pixels; then official normalization without resizing. Includes DRENDS. Same GT/mask across interventions. New pipeline identity is used for paired contrasts.', 'physical_causality':False},
      'depth_strata_mm':[0,30,60,100,200,300],
      'statistics':'Average frames within group, then groups, then seeds. Seed SD not population CI. Equal-bin mean uses only bins supported in all cohorts for a fixed method/bound; report retained counts; exploratory diagnostic, not primary ranking.',
      'limitations':'No new physical donors/device serials or clinical sizing/ICC collected; no SOTA or physical causal claim.'})
    print('FROZEN',sha(O/'protocol.json'),flush=True)
if __name__=='__main__': freeze()
