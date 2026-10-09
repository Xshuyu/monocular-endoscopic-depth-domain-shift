"""
Strict Evaluation Script: Regularized Radial Affine vs. Standard Baselines
Evaluates depth calibration models on all 92 test frames across 3 C3VD sub-datasets.
No training on test sets, strictly reproducible.
"""

import sys, json, os
from pathlib import Path
import numpy as np
from scipy import stats

ROOT = Path('/ssd-1/temp/kimicode/multmodel/lesion_3d_measurement')
sys.path.insert(0, str(ROOT / 'tests'))
from c3vdv2_external_depth_validation import read_reference, metrics

def evaluate_all():
    old = ROOT / 'datasets/external/C3VDv2_native_subset'
    dev = ROOT / 'datasets/external/C3VDv2_robust_development'
    old_manifest = json.loads((old / 'download_manifest.json').read_text())
    manifest = json.loads((dev / 'download_manifest.json').read_text())

    old_frames = [{**r, 'root': old, 'cache': ROOT / 'outputs/c3vdv2_external_validation/raw_predictions' / f"{r['scene']}_{r['frame_id']}.npz"} for r in old_manifest['frames']]
    dev_frames = [{**r, 'root': dev, 'cache': ROOT / 'outputs/robust_depth_optimization/raw_predictions' / f"{r['scene']}_{r['frame_id']}.npz"} for r in manifest['frames']]

    # 1. 72 frames of reused C3VDv2 test set
    # 2. 8 frames of new C2 sigmoid video test set
    test_frames = [r for r in old_frames if r['split'] == 'test'] + [r for r in dev_frames if r['split'] == 'new_sequence_test']

    # 3. 12 frames of fresh C3VD transfer test set
    transfer_path = ROOT / 'outputs/c3vd_frozen_transfer'
    for p in sorted(transfer_path.glob('*_color.npz')):
        test_frames.append({
            'scene': 'trans_t1_a',
            'frame_id': p.stem,
            'evaluation_set': 'reused_c3vd_test',
            'transfer_cache': p,
            'root': ROOT / 'datasets/external/C3VD_fresh_subset',
            'depth_path': p.stem.replace('_color', '_depth') + '.tiff'
        })

    H, W = 1080, 1350
    cx, cy = W / 2.0, H / 2.0
    max_r = np.sqrt(cx**2 + cy**2)
    y_coords, x_coords = np.mgrid[0:H, 0:W]
    r_grid = np.sqrt((x_coords - cx)**2 + (y_coords - cy)**2) / max_r
    r2_grid = r_grid ** 2

    # Standard Affine Baseline (fitted on calibration split)
    params_std = [0.00904776, 0.01545231]
    def pred_std(n):
        d = params_std[0] * n + params_std[1]
        return np.clip(np.divide(1.0, d, out=np.full_like(n, 100.0), where=d > 0), 1.0, 100.0)

    # Proposed: Regularized Radial Affine (fitted on calibration split with L2=1e-4)
    params_opt = [0.00768324, 0.00426751, 0.01458983, 0.00127847]
    def pred_opt_rad(n):
        d = (params_opt[0] + params_opt[1] * r2_grid) * n + (params_opt[2] + params_opt[3] * r2_grid)
        return np.clip(np.divide(1.0, d, out=np.full_like(n, 100.0), where=d > 0), 1.0, 100.0)

    eval_groups = {}
    for row in test_frames:
        if 'evaluation_set' in row:
            eset = row['evaluation_set']
        else:
            eset = 'new_c2_sigmoid_video' if row['split'] == 'new_sequence_test' else 'reused_c3vdv2_test'
        eval_groups.setdefault(eset, []).append(row)

    print("=" * 80)
    print("EVALUATING REGULARIZED RADIAL AFFINE VS STANDARD BASELINE (92 FRAMES)")
    print("=" * 80)

    all_std_maes, all_opt_maes = [], []
    report_dict = {}

    for eset, rows in eval_groups.items():
        m_std, m_opt = [], []
        for row in rows:
            if 'transfer_cache' in row:
                with np.load(row['transfer_cache']) as c: n = c['Small'].copy()
                g, v = read_reference(row['root'] / row['depth_path'])
            else:
                with np.load(row['cache']) as c: n = c['relative_depth_anything'].copy()
                g, v = read_reference(row['root'] / row['depth_path'])
            
            s_res = metrics(pred_std(n), g, v)
            o_res = metrics(pred_opt_rad(n), g, v)
            m_std.append(s_res)
            m_opt.append(o_res)
            all_std_maes.append(s_res['mae_mm'])
            all_opt_maes.append(o_res['mae_mm'])
        
        mae_s = np.mean([x['mae_mm'] for x in m_std])
        mae_o = np.mean([x['mae_mm'] for x in m_opt])
        rel_s = np.mean([x['abs_rel'] for x in m_std])
        rel_o = np.mean([x['abs_rel'] for x in m_opt])
        d1_s  = np.mean([x['delta1_fraction'] for x in m_std]) * 100
        d1_o  = np.mean([x['delta1_fraction'] for x in m_opt]) * 100

        report_dict[eset] = {
            'n_frames': len(rows),
            'baseline_mae_mm': float(mae_s),
            'proposed_mae_mm': float(mae_o),
            'reduction_pct': float((mae_s - mae_o) / mae_s * 100),
            'baseline_absrel': float(rel_s),
            'proposed_absrel': float(rel_o),
            'baseline_delta1_pct': float(d1_s),
            'proposed_delta1_pct': float(d1_o)
        }
        print(f"\n[{eset}] ({len(rows)} frames):")
        print(f"  Baseline MAE: {mae_s:.3f} mm -> Proposed MAE: {mae_o:.3f} mm (Reduction: {(mae_s-mae_o)/mae_s*100:+.1f}%)")
        print(f"  Baseline AbsRel: {rel_s:.3f} -> Proposed AbsRel: {rel_o:.3f} (Reduction: {(rel_s-rel_o)/rel_s*100:+.1f}%)")
        print(f"  Baseline d1 (<1.25): {d1_s:.1f}% -> Proposed d1: {d1_o:.1f}% (Gain: {d1_o-d1_s:+.1f}%)")

    t_stat, p_val = stats.ttest_rel(all_std_maes, all_opt_maes)
    w_stat, w_pval = stats.wilcoxon(np.array(all_std_maes) - np.array(all_opt_maes))
    
    total_s = np.mean(all_std_maes)
    total_o = np.mean(all_opt_maes)

    print("\n" + "=" * 80)
    print("TOTAL AGGREGATE SUMMARY (N = 92 FRAMES):")
    print(f"  Baseline Standard Affine:   {total_s:.3f} mm")
    print(f"  Proposed Radial Affine:     {total_o:.3f} mm")
    print(f"  Global MAE Reduction:       {(total_s - total_o) / total_s * 100:+.1f}%")
    print(f"  Paired t-test:              t = {t_stat:.3f}, p = {p_val:.2e}")
    print(f"  Wilcoxon Signed-Rank Test:  W = {w_stat:.1f}, p = {w_pval:.2e}")
    print("=" * 80)

    out_file = ROOT / 'outputs/verified/radial_affine_benchmark_summary.json'
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, 'w') as f:
        json.dump(report_dict, f, indent=2)
    print(f"Results saved to: {out_file}")

if __name__ == '__main__':
    evaluate_all()
