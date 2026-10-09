"""Fixed-protocol, multi-seed procedural evaluation; no training or selection.

Direct invocation is supported. Outputs refuse to overwrite an existing run.
All methods share cases, depth, masks and scale within each scale condition.
The legacy renderer remains intentionally unchanged, including its limitations.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))
# Explicitly fix the evaluated path; do not silently download or change backend.
os.environ['HF_HUB_ENABLE_DOWNLOAD'] = '0'
os.environ['HF_HUB_OFFLINE'] = '1'
import cv2
import numpy as np
import scipy
from scipy import stats
from scipy.spatial import ConvexHull, distance
from lesion_3d_measurement.src.camera_model import CameraIntrinsics, backproject_depth_to_pointcloud
from lesion_3d_measurement.src.depth_anything_backbone import DepthAnythingV2Backbone
from lesion_3d_measurement.src.ground_truth_benchmark import generate_benchmark_cohort
from lesion_3d_measurement.src.lie_algebra_obb import compute_robust_obb_lie
from lesion_3d_measurement.src.metric_scale import MetricScaleCalibrator, ToolDimensionPrior
from lesion_3d_measurement.src.robust_obb_methods import pca_obb, mad_filtered_pca_obb
from lesion_3d_measurement.tests.verified_reanalysis import holm_adjust

METHODS = ('pca_full', 'pca_q01_q99', 'mad_pca_q01_q99', 'so3_never', 'so3_always', 'so3_adaptive')
SCALES = ('oracle_depth', 'prior', 'tool', 'fusion')
COMPARISONS = (
    ('so3_never', 'pca_q01_q99', 'matched_no_filter'),
    ('so3_always', 'mad_pca_q01_q99', 'matched_radial_MAD'),
    ('so3_adaptive', 'mad_pca_q01_q99', 'different_filter_policy_exploratory'),
)

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')

def write_csv(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)

def size_bin(x):
    return int(x >= 5) + int(x >= 10) + int(x >= 20)

def seed_mean_ci(values, random_seed, draws):
    """Equal-size seed blocks: bootstrap whole seed means, never method rows."""
    x = np.asarray(values, dtype=float)
    rng = np.random.default_rng(random_seed)
    estimates = rng.choice(x, size=(draws, len(x)), replace=True).mean(axis=1)
    return np.quantile(estimates, [.025, .975]).tolist()

def metrics(rows):
    truth = np.asarray([r['true_major_mm'] for r in rows])
    pred = np.asarray([r['predicted_major_mm'] for r in rows])
    error = pred - truth
    cm = np.zeros((4, 4), dtype=int)
    for a, b in zip(truth, pred):
        cm[size_bin(a), size_bin(b)] += 1
    supports = cm.sum(axis=1)
    recall = np.divide(cm.diagonal(), supports, out=np.zeros(4, dtype=float), where=supports > 0)
    denom = np.var(truth) + np.var(pred) + (truth.mean() - pred.mean()) ** 2
    ccc = 2 * np.mean((truth - truth.mean()) * (pred - pred.mean())) / denom if denom else None
    total = float(np.sum((truth-truth.mean()) ** 2))
    sd = float(error.std(ddof=1))
    return {
        'n_cases': len(rows), 'mae_mm': float(np.abs(error).mean()),
        'rmse_mm': float(np.sqrt(np.mean(error**2))),
        'mre_pct': float(np.mean(np.abs(error)/truth)*100),
        'mean_bias_mm': float(error.mean()), 'error_sd_mm': sd,
        'bland_altman_loa_mm': [float(error.mean()-1.96*sd), float(error.mean()+1.96*sd)],
        'concordance_ccc': None if ccc is None else float(ccc),
        'predictive_r2': 1-float(np.sum(error**2))/total if total else None,
        'size_bin_agreement_pct': float(100*np.trace(cm)/cm.sum()),
        'size_bin_balanced_accuracy_pct': float(100*recall[supports > 0].mean()),
        'size_bin_recall': recall.tolist(), 'size_bin_support': supports.tolist(),
        'size_bin_confusion_true_rows_pred_columns': cm.tolist(),
    }

def dataset_audit():
    base = ROOT/'datasets/kvasir-instrument/kvasir-instrument'
    if not (base/'train.txt').exists() or not (base/'test.txt').exists():
        return {'status': 'missing_split_files'}
    splits = {k:(base/(k+'.txt')).read_text().split() for k in ('train', 'test')}
    files = [p for p in (base/'images').rglob('*') if p.is_file()]
    lookup = {key:p for p in files for key in (p.name, p.stem)}
    hashes = {}; missing = []
    for split, ids in splits.items():
        hashes[split] = defaultdict(list)
        for sid in ids:
            p = lookup.get(Path(sid).name, lookup.get(Path(sid).stem))
            if p is None:
                missing.append({'split':split, 'id':sid}); continue
            hashes[split][digest(p)].append(sid)
    overlap = sorted(set(hashes['train']) & set(hashes['test']))
    pairs = [{'sha256':h, 'train':hashes['train'][h], 'test':hashes['test'][h]} for h in overlap]
    excluded = {sid for row in pairs for sid in row['train']}
    # Suggested train-only exclusions; original data and splits are untouched.
    return {'status':'complete' if not missing else 'incomplete',
            'split_counts':{k:len(v) for k,v in splits.items()},
            'split_file_sha256':{k:digest(base/(k+'.txt')) for k in splits},
            'id_overlap':sorted(set(splits['train']) & set(splits['test'])),
            'byte_identical_cross_split_groups':pairs, 'missing_images':missing,
            'suggested_train_exclusions_preserving_test':sorted(excluded),
            'remaining_checks':'Near duplicates, patient/video grouping and mask provenance are not established.',
            'used_for_current_measurement_benchmark':False}

def summarize(rows, seeds, draws):
    groups = defaultdict(list)
    for row in rows:
        groups[(row['scale_mode'],row['method'])].append(row)
    summaries = []; per_seed = []; strata = []
    for (scale, method), group in groups.items():
        block_mae = []
        for seed in seeds:
            m = metrics([r for r in group if r['seed'] == seed])
            per_seed.append({'scale_mode':scale, 'method':method, 'seed':seed, **m})
            block_mae.append(m['mae_mm'])
        summaries.append({'scale_mode':scale, 'method':method, **metrics(group),
                          'seed_mae_sd_mm':float(np.std(block_mae,ddof=1)),
                          'mae_seed_block_bootstrap_95ci':seed_mean_ci(block_mae,20260926,draws)})
        for b in range(4):
            sub = [r for r in group if size_bin(r['true_major_mm']) == b]
            strata.append({'scale_mode':scale,'method':method,'size_bin':b,**metrics(sub)})
    paired = []
    for scale in SCALES:
        for candidate, baseline, design in COMPARISONS:
            a = {(r['seed'],r['sample_id']):r for r in groups[(scale,candidate)]}
            b = {(r['seed'],r['sample_id']):r for r in groups[(scale,baseline)]}
            assert a.keys() == b.keys()
            block_diffs = []
            for seed in seeds:
                keys = sorted(k for k in a if k[0] == seed)
                block_diffs.append(float(np.mean([a[k]['abs_error_mm']-b[k]['abs_error_mm'] for k in keys])))
            p = float(stats.wilcoxon(block_diffs).pvalue) if np.any(np.asarray(block_diffs)!=0) else 1.
            paired.append({'scale_mode':scale,'candidate':candidate,'baseline':baseline,'design':design,
                           'n_seed_blocks':len(seeds),'mean_abs_error_difference_mm':float(np.mean(block_diffs)),
                           'seed_block_bootstrap_95ci':seed_mean_ci(block_diffs,20260927,draws),
                           'seed_mean_differences_mm':block_diffs,'wilcoxon_seed_means_p':p})
    for row,p in zip(paired,holm_adjust([r['wilcoxon_seed_means_p'] for r in paired])):
        row['wilcoxon_holm_12_comparisons_p'] = float(p)
    return summaries, per_seed, strata, paired

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/paper_strengthening')
    parser.add_argument('--seeds',type=int,nargs='+',default=list(range(3100,3110)))
    parser.add_argument('--samples-per-seed',type=int,default=32)
    parser.add_argument('--bootstrap-draws',type=int,default=10000)
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds) or len(args.seeds)<2:
        parser.error('Use at least two distinct seeds.')
    if args.samples_per_seed < 4 or args.samples_per_seed % 4:
        parser.error('samples-per-seed must be positive and divisible by four.')
    if args.bootstrap_draws < 100:
        parser.error('Use at least 100 bootstrap draws.')
    out = args.output.resolve(); out.mkdir(parents=True,exist_ok=False)
    started = time.monotonic()
    paths = sorted((ROOT/'src').glob('*.py')) + [Path(__file__).resolve(),ROOT/'tests/verified_reanalysis.py']
    source_hashes = {str(p.relative_to(ROOT)):digest(p) for p in paths}
    protocol = {
        'created_utc':datetime.now(timezone.utc).isoformat(), 'seeds':args.seeds,
        'samples_per_seed':args.samples_per_seed, 'methods':METHODS,'scale_modes':SCALES,
        'backend':'DepthAnythingV2Backbone photometric fallback; CPU; remote weights disabled',
        'scale_definition':'30 mm lesion-ROI median prior; 2.4 mm tool; fixed 0.5/0.5 fusion',
        'primary_endpoint':'Absolute major-diameter error against legacy nominal generator dimensions',
        'comparisons':COMPARISONS,'quantiles':[.01,.99],
        'filter':'Matched pairs use never/never or radial MAD always/always; adaptive pair is exploratory.',
        'statistics':{'bootstrap_draws':args.bootstrap_draws,'unit':'whole seed blocks',
                      'paired_test':'Wilcoxon on seed-average paired errors; Holm across all 12 comparisons',
                      'ci':'unadjusted 95% percentile intervals; not simultaneous intervals'},
        'selection':'No training, hyperparameter search, data-driven fusion or seed exclusion.',
        'limitations':[
            'Seeds are new synthetic draws, not external clinical data or a prospectively untouched test set.',
            'Legacy size and morphology cycles are confounded; strata must not be interpreted independently.',
            'Nominal dimensions are not guaranteed to equal rendered surface diameter.',
            'Generator and photometric inverse share lighting assumptions; true masks are supplied.',
            'Oracle changes depth source as well as scale; it is a separate reference, not deployable calibration.',
            'Size bins are not clinical risk labels; no inter-rater ICC or physical patient reference.',
        ],
        'source_sha256_at_start':source_hashes,
        'command':[sys.executable,*sys.argv],
    }
    dump(out/'protocol.json',protocol)  # Persist before looking at outcomes.
    model = DepthAnythingV2Backbone(device='cpu')
    assert model.pipe is None, 'This protocol only evaluates the explicit fallback.'
    intr = CameraIntrinsics.from_fov(640,480,140.)
    calibrator = MetricScaleCalibrator(); rows=[]; truth_audit=[]
    for seed in args.seeds:
        cohort = generate_benchmark_cohort(args.samples_per_seed,seed=seed)
        for sample in cohort:
            rel = model.predict(sample.rgb_image,tool_mask=sample.tool_mask)
            prior = calibrator.calibrate_with_prior(rel,roi_mask=sample.lesion_mask,target_distance_mm=30.)
            tool,_ = calibrator.calibrate_with_tool(rel,sample.tool_mask,intr,ToolDimensionPrior(2.4))
            maps = {'oracle_depth':sample.ground_truth_depth_mm,'prior':rel*prior,
                    'tool':rel*tool,'fusion':rel*(.5*prior+.5*tool)}
            for scale,depth in maps.items():
                points,_ = backproject_depth_to_pointcloud(depth,intr,sample.lesion_mask)
                if len(points)<10 or not np.all(np.isfinite(points)):
                    raise ValueError(f'Invalid point support: {seed}/{sample.sample_id}/{scale}')
                if scale == 'oracle_depth':
                    hull = points[ConvexHull(points).vertices]
                    chord = float(distance.pdist(hull).max())
                    truth_audit.append({'seed':seed,'sample_id':sample.sample_id,
                                        'nominal_major_mm':sample.true_major_mm,
                                        'rendered_surface_max_chord_mm':chord,
                                        'chord_minus_nominal_mm':chord-sample.true_major_mm})
                estimates = {'pca_full':pca_obb(points)[0],
                             'pca_q01_q99':pca_obb(points,quantiles=(.01,.99))[0],
                             'mad_pca_q01_q99':mad_filtered_pca_obb(points,quantiles=(.01,.99))[0]}
                for mode in ('never','always','adaptive'):
                    estimates['so3_'+mode] = compute_robust_obb_lie(points,outlier_filter=mode).major_diameter_mm
                for method,pred in estimates.items():
                    error = float(pred-sample.true_major_mm)
                    if not np.isfinite(error): raise ValueError('Non-finite estimate')
                    rows.append({'seed':seed,'sample_id':sample.sample_id,'scale_mode':scale,
                                 'method':method,'true_major_mm':sample.true_major_mm,
                                 'predicted_major_mm':float(pred),'signed_error_mm':error,
                                 'abs_error_mm':abs(error),'size_bin':size_bin(sample.true_major_mm),
                                 'morphology':sample.true_paris_morphology,
                                 'working_distance_mm':sample.working_distance_mm})
        write_csv(out/'trials.csv',rows)
        print(f'Completed seed {seed}: {len(rows)} method rows, {time.monotonic()-started:.1f}s',flush=True)
    summary,per_seed,strata,paired=summarize(rows,args.seeds,args.bootstrap_draws)
    write_csv(out/'rendered_truth_audit.csv',truth_audit)
    dump(out/'dataset_audit.json',dataset_audit())
    dump(out/'per_seed.json',per_seed); dump(out/'size_strata.json',strata)
    environment = {'python':sys.version,'executable':sys.executable,'platform':platform.platform(),
                   'numpy':np.__version__,'scipy':scipy.__version__,'opencv':cv2.__version__}
    for name in ('torch','pandas','matplotlib'):
        environment[name]=importlib.metadata.version(name)
    dump(out/'summary.json',{'n_unique_cases':len(args.seeds)*args.samples_per_seed,
         'n_method_rows':len(rows),'summary':summary,'paired_comparisons':paired,
         'rendered_truth_audit':{'mean_absolute_chord_minus_nominal_mm':float(np.mean([abs(r['chord_minus_nominal_mm']) for r in truth_audit])),
                                 'interpretation':'Different estimands: rendered sampled-surface chord vs nominal design size; not a corrected ground truth.'},
         'environment':environment,'seconds':time.monotonic()-started})
    after={str(p.relative_to(ROOT)):digest(p) for p in paths}
    manifest={'source_sha256_at_start':source_hashes,'source_sha256_at_end':after,
              'source_changed_during_run':[p for p in source_hashes if after[p]!=source_hashes[p]],
              'outputs_sha256':{p.name:digest(p) for p in out.iterdir() if p.is_file() and p.name!='console.log'}}
    dump(out/'manifest.json',manifest)
    if manifest['source_changed_during_run']:
        raise RuntimeError('Source changed during run; inspect manifest before interpreting results.')
    print('Completed:',out,flush=True)

if __name__ == '__main__':
    main()
