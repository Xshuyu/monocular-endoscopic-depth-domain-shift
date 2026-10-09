"""Depth-matched (same-depth-bin) factorial scoring for the CMIG revision.

Motivation: the pooled strong-blur factorial effects mix reference-depth
ranges across cohorts (C3VD is shallow, DRENDS is deep), so cohort-level
blur effects are not depth-matched and cross-cohort comparisons can be
misread.  Here every factorial contrast is computed inside a fixed
reference-depth bin, where all four conditions share identical pixels and
identical reference-depth support.  Bins follow the frozen cmig_revision
depth_strata_mm edges; predictions keep the primary [1,100] mm clip.

Settings are written to depth_matched_protocol.json BEFORE any scoring.
Existing output files are never overwritten; the script refuses to run if
any of its output files already exist.  No new physical data collected.
"""
from cmig_revision_common import *
import argparse, cv2
from evaluate_major_revision import frames, reference
from c3vdv2_external_depth_validation import metrics

EXT = R / 'outputs/cmig_extension'
EDGES = [0, 30, 60, 100, 200, 300]   # frozen cmig_revision depth_strata_mm
BOUND = 100                          # primary prediction clip, as frozen
FACTOR_SOURCES = {
    'identity': O / 'factor_predictions',
    'dark': O / 'factor_predictions',
    'strong_blur': EXT / 'strong_blur_predictions',
    'dark_strong_blur': EXT / 'strong_blur_predictions',
}
KEYS = ['mae_mm', 'rmse_mm', 'abs_rel', 'bias_mm', 'delta1_fraction']
OUTS = ['depth_matched_per_frame.csv', 'depth_matched_per_object.json',
        'depth_matched_summary.json', 'depth_matched_effects.json']


def protocol():
    return {
        'created_utc': __import__('datetime').datetime.now(
            __import__('datetime').timezone.utc).isoformat(),
        'parent_protocol_sha256': sha(O / 'protocol.json'),
        'extension_protocol_sha256': sha(EXT / 'extension_protocol.json'),
        'purpose': (
            'Depth-matched factorial failure analysis. All darkening/blur '
            'contrasts are computed inside fixed reference-depth bins so '
            'that every compared condition has identical pixels and '
            'identical reference-depth support. This removes the '
            'depth-distribution mismatch present in pooled cohort-level '
            'effects. No new physical data collected.'
        ),
        'depth_bins_mm': EDGES,
        'prediction_clip_mm': [1, BOUND],
        'models': ['source_only', 'adapted'],
        'seeds': SEEDS,
        'factors': list(FACTOR_SOURCES),
        'blur_definition': 'strong_blur: Gaussian sigma=5.0 model-input '
                           'pixels after resize to 224x280 (see '
                           'extension_protocol.json)',
        'aggregation': ('Mean per frame within bin, then equal weight per '
                        'object group, then equal weight over groups; seed '
                        'effects averaged with sample SD. Frames without '
                        'bin support excluded.'),
        'interpretation': ('Within-bin contrasts are pixel-paired across '
                           'conditions. Cross-cohort comparison is '
                           'meaningful only within the same bin and remains '
                           'descriptive, not a causal identification.'),
        'selection': 'All frames, models, seeds and factors scored; no '
                     'test-performance-based selection.',
    }


def score():
    check()   # frozen cmig_revision code and parent protocol intact
    proto = EXT / 'depth_matched_protocol.json'
    for name in OUTS:
        if (EXT / name).exists():
            raise ValueError('Refusing to overwrite ' + str(EXT / name))
    if not proto.exists():
        dump(proto, protocol())
        print('WROTE depth_matched_protocol.json', flush=True)

    rows = []
    for r in frames():
        gt, v = reference(r)
        assert v.any()
        for regime in ['source_only', 'adapted']:
            for seed in SEEDS:
                for factor, base in FACTOR_SOURCES.items():
                    npz = base / f'{regime}_{seed}' / (r['id'] + '_' + factor + '.npz')
                    with np.load(npz) as f:
                        raw = cv2.resize(f['prediction'], gt.shape[::-1],
                                         interpolation=cv2.INTER_LINEAR)
                    pred = np.clip(raw, 1, BOUND)
                    for lo, hi in zip(EDGES[:-1], EDGES[1:]):
                        mask = v & (gt > lo) & (gt <= hi)
                        if not mask.any():
                            continue
                        rows.append({
                            'cohort': cohort(r), 'object': str(r['object']),
                            'frame': r['id'], 'method': regime, 'seed': seed,
                            'factor': factor, 'bin': f'{lo}_{hi}',
                            'pixels': int(mask.sum()),
                            **metrics(pred, gt, mask)})
        print('DEPTH_BIN', r['id'], flush=True)

    # Same aggregation convention as score_cmig_revision.aggregate:
    # frames -> object groups -> summary, equal weight per level.
    groupkeys = ['cohort', 'method', 'seed', 'factor', 'bin']
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
            **{k: float(np.mean([r[k] for r in rr])) for k in KEYS}})
    buckets = {}
    for r in objects:
        buckets.setdefault(tuple(r[k] for k in groupkeys), []).append(r)
    summary = [{
        **dict(zip(groupkeys, key)),
        'groups': len(rr),
        'frames': sum(r['frames'] for r in rr),
        'pixels': sum(r['pixels'] for r in rr),
        **{k: float(np.mean([r[k] for r in rr])) for k in KEYS}}
        for key, rr in sorted(buckets.items())]
    writecsv(EXT / 'depth_matched_per_frame.csv', rows)
    dump(EXT / 'depth_matched_per_object.json', objects)
    dump(EXT / 'depth_matched_summary.json', summary)

    # Factorial effects inside each depth bin; all four conditions required.
    effects = []
    combos = sorted({(x['cohort'], x['method'], x['seed'], x['bin'])
                     for x in summary})
    for c, m, s, b in combos:
        a = {x['factor']: x['mae_mm'] for x in summary
             if (x['cohort'], x['method'], x['seed'], x['bin']) == (c, m, s, b)}
        if all(k in a for k in FACTOR_SOURCES):
            e00, e10 = a['identity'], a['dark']
            e01, e11 = a['strong_blur'], a['dark_strong_blur']
            pix = {x['factor']: x['pixels'] for x in summary
                   if (x['cohort'], x['method'], x['seed'], x['bin'])
                   == (c, m, s, b)}
            assert len(set(pix.values())) == 1, 'bin support must be identical across factors'
            effects.append({
                'cohort': c, 'method': m, 'seed': s, 'bin': b,
                'pixels': list(pix.values())[0],
                'dark_main_mm': ((e10 - e00) + (e11 - e01)) / 2,
                'strong_blur_main_mm': ((e01 - e00) + (e11 - e10)) / 2,
                'interaction_mm': e11 - e10 - e01 + e00,
                **a})
    dump(EXT / 'depth_matched_effects.json', effects)

    # Compact seed-averaged printout for manuscript drafting.
    print('\n=== depth-matched strong-blur effects (mean +/- SD over seeds) ===')
    for c in sorted({e['cohort'] for e in effects}):
        for m in ['source_only', 'adapted']:
            for b in [f'{lo}_{hi}' for lo, hi in zip(EDGES[:-1], EDGES[1:])]:
                ee = [e for e in effects
                      if (e['cohort'], e['method'], e['bin']) == (c, m, b)]
                if not ee:
                    continue
                blur = [e['strong_blur_main_mm'] for e in ee]
                dark = [e['dark_main_mm'] for e in ee]
                ident = [e['identity'] for e in ee]
                print(f"{c:16s} {m:12s} bin {b:8s} px={ee[0]['pixels']:>10d} "
                      f"identity={np.mean(ident):8.3f} "
                      f"dark={np.mean(dark):+7.3f}±{np.std(dark, ddof=1) if len(dark) > 1 else 0:5.3f} "
                      f"blur={np.mean(blur):+7.3f}±{np.std(blur, ddof=1) if len(blur) > 1 else 0:5.3f}",
                    flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('phase', choices=['score'])
    a = p.parse_args()
    globals()[a.phase]()
