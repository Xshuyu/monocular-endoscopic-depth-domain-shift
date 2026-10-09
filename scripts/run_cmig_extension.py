"""Extension calibration and factor interventions for CMIG revision.

Two additions on top of the frozen cmig_revision protocol:
  1. target_only calibration design: 16384 target-domain pixels only,
     fitting affine_ls and radial_ls heads for both backbones.
     Rationale: isolates recalibration-only baseline vs. neural adaptation.
  2. strong_blur factor: sigma=5.0 in model-input pixels (after resize to
     224x280), paired with identity and dark to give a stronger degradation
     contrast. Sigma 1.5 was too mild for a convincing sensitivity check;
     5.0 corresponds to ~2.2% of image height, approximating heavy lens fog.

All outputs are written to outputs/cmig_extension/ before any scoring.
No outputs are deleted or overwritten once written.  Scoring reads the
extension heads alongside the frozen cmig_revision heads.
"""
from cmig_revision_common import *
import argparse
import cv2, torch, tifffile
from transformers import AutoModelForDepthEstimation, AutoImageProcessor
from train_major_revision_baseline import M
from evaluate_major_revision import reference, frames
from c3vdv2_external_depth_validation import metrics
from src.radial_affine_calibration import fit, radius_squared, predict

EXT = R / 'outputs/cmig_extension'
STRONG_BLUR_SIGMA = 5.0   # model-input pixels; >3x stronger than original 1.5
EXTENSION_FACTORS = {
    'identity':    (1.0, 0.0),
    'dark':        (1.5, 0.0),
    'strong_blur': (1.0, STRONG_BLUR_SIGMA),
    'dark_strong_blur': (1.5, STRONG_BLUR_SIGMA),
}

# ── helpers ────────────────────────────────────────────────────────────────
def gt_for(r):
    """Return (gt_mm, valid_mask) for any row kind."""
    if 'kind' not in r:
        return reference(r)
    if r['kind'] == 'source':
        from c3vdv2_external_depth_validation import read_reference
        return read_reference(Path(r['gt']))
    xyz = tifffile.imread(r['gt'])
    g = xyz[..., 2].astype(float)
    return g, np.isfinite(xyz).all(-1) & (g > 0)


def indices_tgt(rows, total):
    """Deterministic equal-object, equal-frame pixel selection for target rows."""
    result = []
    groups = sorted({str(r['object']) for r in rows})
    for j, obj in enumerate(groups):
        rr = sorted([r for r in rows if str(r['object']) == obj],
                    key=lambda r: r['id'])
        n = total // len(groups) + (j < total % len(groups))
        for k, r in enumerate(rr):
            g, v = gt_for(r)
            valid = np.flatnonzero(v)
            count = n // len(rr) + (k < n % len(rr))
            assert len(valid) >= count, f'Not enough valid pixels in {r["id"]}'
            ix = valid[np.linspace(0, len(valid) - 1, count, dtype=int)]
            result.append((r, ix))
    assert sum(len(ix) for _, ix in result) == total
    return result


def extension_protocol_record():
    """Frozen protocol description for this extension."""
    return {
        'created_utc': __import__('datetime').datetime.now(
            __import__('datetime').timezone.utc).isoformat(),
        'parent_protocol_sha256': sha(O / 'protocol.json'),
        'purpose': (
            'Two additions to cmig_revision: (1) target_only calibration '
            'design using 16384 target pixels only, to separate recalibration '
            'from neural adaptation; (2) strong_blur factor at sigma=5.0 '
            'model-input pixels (resize-then-blur) for a more decisive '
            'sensitivity check. No new physical data collected.'
        ),
        'target_only_calibration': {
            'pixels': 16384,
            'domain': 'target only (SCARED subjects 1,2,3,6)',
            'families': ['affine_ls', 'radial_ls'],
            'lambda': 0,
            'backbones': ['small', 'pcc'],
            'rationale': (
                'Baseline: can the supervised-adaptation gain be matched by '
                'refitting the calibration head on target-domain frames only, '
                'without changing any model weights?'
            ),
        },
        'strong_blur_factor': {
            'sigma_model_input_pixels': STRONG_BLUR_SIGMA,
            'model_input_resolution': [224, 280],
            'sigma_as_fraction_of_height': STRONG_BLUR_SIGMA / 224,
            'applied_after': 'resize to model-input resolution',
            'factors': list(EXTENSION_FACTORS.keys()),
            'rationale': (
                'Original sigma=1.5 produced negligible MAE changes (<0.5 mm '
                'in most seeds). sigma=5.0 provides a stronger intervention '
                'comparable to lens contamination or image compression artifacts.'
            ),
            'models': ['source_only', 'adapted'],
            'seeds': SEEDS,
        },
        'scoring': 'Separate extension_comparison_summary.json; '
                   'combined with cmig_revision heads in combined_summary.json.',
    }


# ── phase 1: target_only calibration heads ─────────────────────────────────
def calibrate():
    """Fit target_only calibration heads for both backbones."""
    EXT.mkdir(exist_ok=True)
    out_path = EXT / 'extension_calibration_heads.json'
    if out_path.exists():
        raise ValueError('Refusing to overwrite frozen extension heads')
    # Write protocol first
    proto_path = EXT / 'extension_protocol.json'
    if not proto_path.exists():
        dump(proto_path, extension_protocol_record())
        print('WROTE extension_protocol.json', flush=True)

    p = check()
    tgt = [r for r in p['train_rows'] if r['kind'] == 'target']
    samples = indices_tgt(tgt, 16384)

    support = []
    for r, ix in samples:
        support.append({
            'frame': r['id'], 'kind': r['kind'],
            'object': str(r['object']), 'n': len(ix),
            'flat_indices_sha256':
                __import__('hashlib').sha256(
                    ix.astype('<i8').tobytes()).hexdigest(),
        })
    dump(EXT / 'extension_calibration_support.json', support)

    heads = {}
    for backbone in ['small', 'pcc']:
        dd, zz, rr, gg = [], [], [], []
        for row, ix in samples:
            g, v = gt_for(row)
            npz_path = O / 'raw' / backbone / (row['id'] + '.npz')
            with np.load(npz_path) as f:
                d = cv2.resize(f['prediction'], g.shape[::-1],
                               interpolation=cv2.INTER_LINEAR).astype(float)
            dd.extend(d.ravel()[ix])
            zz.extend(g.ravel()[ix])
            rr.extend(radius_squared(g.shape).ravel()[ix])
            gg.extend([row['kind']] * len(ix))
        for radial in [False, True]:
            name = f'{backbone}_target_only_' + (
                'radial_ls' if radial else 'affine_ls')
            heads[name] = fit(dd, rr, zz, gg, radial, 0.)
            assert heads[name]['n_samples'] == 16384
            print(f'FIT {name}  coeff={heads[name]["coefficients_a0_a1_b0_b1"]}',
                  flush=True)

    dump(out_path, {
        'parent_protocol_sha256': sha(O / 'protocol.json'),
        'extension_protocol_sha256': sha(EXT / 'extension_protocol.json'),
        'heads': heads,
        'test_reference_used': False,
        'sampling_support_sha256': sha(EXT / 'extension_calibration_support.json'),
    })
    print(f'FROZEN {len(heads)} extension calibration heads', flush=True)


# ── phase 2: strong_blur factor predictions ────────────────────────────────
def strong_blur_factors():
    """Run strong_blur and dark_strong_blur factor predictions."""
    EXT.mkdir(exist_ok=True)
    check()
    torch.set_num_threads(4)
    m = AutoModelForDepthEstimation.from_pretrained(
        M, local_files_only=True).cuda().eval()
    proc = AutoImageProcessor.from_pretrained(M, local_files_only=True,
                                              use_fast=False)
    major_protocol_sha = sha(OLD / 'training_protocol.json')
    states = {
        (regime, seed):
            torch.load(OLD / 'trained_models' / f'{regime}_{seed}.pt',
                       map_location='cpu', weights_only=True)
        for regime in ['source_only', 'adapted']
        for seed in SEEDS
    }

    for r in frames():
        rgb = cv2.cvtColor(cv2.imread(str(R / r['rgb'])), cv2.COLOR_BGR2RGB)
        # Resize to model-input resolution first, then apply perturbation
        inp = proc(
            images=rgb,
            return_tensors='np',
            size={'height': 224, 'width': 280},
            keep_aspect_ratio=False,
            do_rescale=False,
            do_normalize=False,
        ).pixel_values[0]
        resized = np.moveaxis(inp, 0, -1).astype(np.float32) / 255.

        for factor, (gamma, sigma) in EXTENSION_FACTORS.items():
            # Skip factors already covered by original cmig_revision factors
            if factor in ('identity', 'dark'):
                continue   # already run; don't duplicate
            paths = {
                (regime, seed):
                    EXT / 'strong_blur_predictions' / f'{regime}_{seed}' /
                    (r['id'] + '_' + factor + '.npz')
                for regime, seed in states
            }
            if all(p.exists() for p in paths.values()):
                continue

            im = np.clip(resized ** gamma, 0, 1)
            # Apply Gaussian blur in model-input pixel space
            im = cv2.GaussianBlur(im, (0, 0), sigma)

            pixels = proc(
                images=im,
                return_tensors='pt',
                do_resize=False,
                do_rescale=False,
            ).to('cuda')
            assert pixels.pixel_values.shape[-2:] == (224, 280)

            with torch.inference_mode():
                features = m.backbone.forward_with_filtered_kwargs(
                    pixels.pixel_values).feature_maps

            for key, state in states.items():
                dest = paths[key]
                if dest.exists():
                    continue
                assert state['protocol_sha256'] == major_protocol_sha
                m.neck.load_state_dict(state['neck'])
                m.head.load_state_dict(state['head'])
                with torch.inference_mode():
                    log = m.head(m.neck(features, 16, 20), 16, 20)
                pred = torch.exp(log.clamp(-10, 10)).squeeze().cpu().numpy()
                assert np.isfinite(pred).all()
                dest.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(dest, prediction=pred)

        print('STRONG_BLUR_FACTORS', r['id'], flush=True)


# ── phase 3: score extension ────────────────────────────────────────────────
def score():
    """Score extension calibration heads and strong_blur factors."""
    EXT.mkdir(exist_ok=True)
    check()
    # Load frozen extension heads
    ext_heads_path = EXT / 'extension_calibration_heads.json'
    if not ext_heads_path.exists():
        raise ValueError('Run calibrate phase first')
    ext_heads = read(ext_heads_path)['heads']
    # Load original frozen heads for combined output
    orig_heads = read(O / 'calibration_heads.json')['heads']

    KEYS = ['mae_mm', 'rmse_mm', 'abs_rel', 'bias_mm', 'delta1_fraction']
    edges = [0, 30, 60, 100, 200, 300]

    def agg(rows, name, groupkeys):
        buckets = {}
        for r in rows:
            buckets.setdefault(
                tuple(r[k] for k in groupkeys) + (r['object'],), []).append(r)
        objects = []
        for key, rr in sorted(buckets.items()):
            objects.append({
                **dict(zip(groupkeys + ['object'], key)),
                'frames': len(rr),
                'pixels': sum(r['pixels'] for r in rr),
                **{k: float(np.mean([r[k] for r in rr])) for k in KEYS},
            })
        buckets = {}
        for r in objects:
            buckets.setdefault(tuple(r[k] for k in groupkeys), []).append(r)
        summary = [{
            **dict(zip(groupkeys, key)),
            'groups': len(rr),
            'frames': sum(r['frames'] for r in rr),
            'pixels': sum(r['pixels'] for r in rr),
            **{k: float(np.mean([r[k] for r in rr])) for k in KEYS},
        } for key, rr in sorted(buckets.items())]
        writecsv(EXT / (name + '_per_frame.csv'), rows)
        dump(EXT / (name + '_per_object.json'), objects)
        dump(EXT / (name + '_summary.json'), summary)
        return summary

    # ── 3a: extension calibration heads scoring ──
    calib_rows = []
    for r in frames():
        gt, v = reference(r) if 'kind' not in r else (
            (lambda rr: reference(rr))(r))
        # use shared reference loader
        from evaluate_major_revision import reference as ref_fn
        gt, v = ref_fn(r)
        assert v.any()
        rad = radius_squared(gt.shape)
        all_heads = {**orig_heads, **ext_heads}
        for method, h in all_heads.items():
            backbone = method.split('_')[0]
            if backbone == 'pcc' and r.get('cohort') == 'scared':
                continue
            with np.load(O / 'raw' / backbone / (r['id'] + '.npz')) as f:
                d = cv2.resize(f['prediction'], gt.shape[::-1],
                               interpolation=cv2.INTER_LINEAR).astype(float)
            a0, a1, b0, b1 = h['coefficients_a0_a1_b0_b1']
            inv = (a0 + a1 * rad) * d + b0 + b1 * rad
            raw = np.divide(1., inv, out=np.full_like(inv, np.inf),
                            where=inv > 0)
            for bound in [100, 300]:
                pred = np.clip(raw, 1, bound)
                base = {
                    'cohort': cohort(r), 'object': str(r['object']),
                    'frame': r['id'], 'method': method, 'seed': 0,
                    'bound': bound,
                }
                calib_rows.append({
                    **base, 'pixels': int(v.sum()), **metrics(pred, gt, v)})
        print('SCORE_CALIB', r['id'], flush=True)
    agg(calib_rows, 'combined_calibration',
        ['cohort', 'method', 'seed', 'bound'])

    # ── 3b: strong_blur factor scoring ──
    sbf_rows = []
    major_protocol_sha = sha(OLD / 'training_protocol.json')
    for r in frames():
        from evaluate_major_revision import reference as ref_fn
        gt, v = ref_fn(r)
        assert v.any()
        for regime in ['source_only', 'adapted']:
            for seed in SEEDS:
                # identity and dark from original factor_predictions
                for factor in ['identity', 'dark']:
                    npz = (O / 'factor_predictions' /
                           f'{regime}_{seed}' /
                           (r['id'] + '_' + factor + '.npz'))
                    with np.load(npz) as f:
                        raw = cv2.resize(f['prediction'], gt.shape[::-1],
                                         interpolation=cv2.INTER_LINEAR)
                    pred = np.clip(raw, 1, 100)
                    sbf_rows.append({
                        'cohort': cohort(r), 'object': str(r['object']),
                        'frame': r['id'], 'method': regime, 'seed': seed,
                        'factor': factor, 'pixels': int(v.sum()),
                        **metrics(pred, gt, v),
                    })
                # strong_blur and dark_strong_blur from extension
                for factor in ['strong_blur', 'dark_strong_blur']:
                    npz = (EXT / 'strong_blur_predictions' /
                           f'{regime}_{seed}' /
                           (r['id'] + '_' + factor + '.npz'))
                    with np.load(npz) as f:
                        raw = cv2.resize(f['prediction'], gt.shape[::-1],
                                         interpolation=cv2.INTER_LINEAR)
                    pred = np.clip(raw, 1, 100)
                    sbf_rows.append({
                        'cohort': cohort(r), 'object': str(r['object']),
                        'frame': r['id'], 'method': regime, 'seed': seed,
                        'factor': factor, 'pixels': int(v.sum()),
                        **metrics(pred, gt, v),
                    })
        print('SCORE_SBF', r['id'], flush=True)

    ss = agg(sbf_rows, 'strong_blur_factors',
             ['cohort', 'method', 'seed', 'factor'])
    # Compute 2x2 factorial effects: identity/dark vs. blur_weak(blur)/blur_strong(strong_blur)
    # Re-use identity/dark from existing, compute main effects for strong_blur only
    effects = []
    for c_val, m_val, s_val in sorted({
            (r['cohort'], r['method'], r['seed']) for r in ss}):
        a = {r['factor']: r['mae_mm']
             for r in ss
             if (r['cohort'], r['method'], r['seed']) == (c_val, m_val, s_val)}
        if all(k in a for k in ['identity', 'dark', 'strong_blur', 'dark_strong_blur']):
            e00, e10, e01, e11 = (a['identity'], a['dark'],
                                  a['strong_blur'], a['dark_strong_blur'])
            effects.append({
                'cohort': c_val, 'method': m_val, 'seed': s_val,
                'dark_main_mm': ((e10 - e00) + (e11 - e01)) / 2,
                'strong_blur_main_mm': ((e01 - e00) + (e11 - e10)) / 2,
                'interaction_mm': e11 - e10 - e01 + e00,
                **a,
            })
    dump(EXT / 'strong_blur_effects.json', effects)
    print('STRONG_BLUR_EFFECTS', len(effects), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('phase', choices=['calibrate', 'strong_blur_factors', 'score'])
    a = p.parse_args()
    globals()[a.phase]()
