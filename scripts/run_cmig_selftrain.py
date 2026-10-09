"""Self-training baselines for the CMIG revision (post-hoc extension).

Two pseudo-label regimes on top of the frozen major_revision pipeline:
  self_train:    student initialized identically to the `adapted` regime
                 (same torch seed, same generic init and conv3 reinit, same
                 numpy sampler and therefore identical frame draws), 600
                 updates, batch = 1 source (true label) + 1 target
                 (pseudo-label).  Only the provenance of the target label
                 differs from `adapted`.
  self_train_ft: fine-tune from the seed-matched source_only checkpoint
                 with the same 600-update budget and batch composition
                 (the literal pseudo-label-then-fine-tune design).

Pseudo-labels are seed-matched source_only teacher log-depth predictions
at 224x280 on the 20 SCARED training keyframes (subjects 1,2,3,6); every
pixel is supervised and no reference values are used.  Settings are
written to outputs/cmig_selftrain/protocol.json BEFORE any training.
Existing files are never overwritten.  All regimes and seeds are scored
on every cohort; no test-performance-based selection.
"""
from cmig_revision_common import *
import argparse, time
import torch, cv2
from torch import nn
from train_major_revision_baseline import model, M
from evaluate_major_revision import frames, reference
from c3vdv2_external_depth_validation import metrics

ST = R / 'outputs/cmig_selftrain'
SELF_REGIMES = ['self_train', 'self_train_ft']
KEYS = ['mae_mm', 'rmse_mm', 'abs_rel', 'bias_mm', 'delta1_fraction']


def protocol_record(train_rows):
    return {
        'created_utc': __import__('datetime').datetime.now(
            __import__('datetime').timezone.utc).isoformat(),
        'status': ('Post-hoc extension; all cohorts have previously been '
                   'examined. Settings frozen before these new comparisons, '
                   'not a preregistered blind confirmation.'),
        'parent_protocol_sha256': sha(O / 'protocol.json'),
        'major_training_protocol_sha256': sha(OLD / 'training_protocol.json'),
        'purpose': (
            'Self-training baselines: does the adapted regime gain come '
            'from true target-domain supervision, or do pseudo-labels from '
            'the source_only model on the same target images suffice? '
            'No new physical data collected; the same 20 target training '
            'keyframes are used, with labels replaced by teacher '
            'predictions.'
        ),
        'teacher': ('Seed-matched source_only final checkpoint from '
                    'outputs/major_revision (no validation/test selection); '
                    'log-depth at 224x280, clamp(-10,10); all pixels '
                    'supervised (valid=all), no reference values read.'),
        'regimes': {
            'self_train': ('600 updates, batch = 1 source(true) + 1 '
                           'target(pseudo). Identical torch/numpy seeding '
                           'to the adapted regime: same conv3 '
                           'reinitialization and identical frame draws; '
                           'only target-label provenance differs.'),
            'self_train_ft': ('Fine-tune from seed-matched source_only '
                              'checkpoint; same 600 updates, batch = 1 '
                              'source(true) + 1 target(pseudo). Total '
                              'source exposure therefore exceeds the '
                              'adapted regime; descriptive, not '
                              'exposure-matched.'),
        },
        'seeds': SEEDS,
        'optimizer': 'Unchanged parent AdamW lr=1e-4 wd=0.01, clip 1, '
                     'SmoothL1(log, beta=0.1), final checkpoint only',
        'training_rgb_sha256': {r['id']: sha(R / r['rgb']) for r in train_rows},
        'scoring': ('Primary unaligned MAE at [1,100] mm clip, secondary '
                    '[1,300]; same frames/objects/aggregation as '
                    'score_cmig_revision.'),
        'selection': 'All checkpoints scored on all cohorts; no '
                     'test-performance-based selection.',
    }


def ensure_protocol():
    check()   # frozen cmig_revision code and parent protocol intact
    proto = ST / 'protocol.json'
    if not proto.exists():
        ST.mkdir(exist_ok=True)
        dump(proto, protocol_record(read(O / 'protocol.json')['train_rows']))
        print('WROTE cmig_selftrain/protocol.json', flush=True)
    return read(proto)


# ── phase 1: pseudo-labels ──────────────────────────────────────────────────
def pseudo():
    p = ensure_protocol()
    torch.set_num_threads(4)
    m = model().cuda().eval()
    rows = read(O / 'protocol.json')['train_rows']
    tgt = [r for r in rows if r['kind'] == 'target']
    record = {}
    for seed in SEEDS:
        state = torch.load(OLD / 'trained_models' / f'source_only_{seed}.pt',
                           map_location='cpu', weights_only=True)
        assert state['protocol_sha256'] == sha(OLD / 'training_protocol.json')
        m.neck.load_state_dict(state['neck'])
        m.head.load_state_dict(state['head'])
        outdir = ST / 'pseudo_labels' / f'seed_{seed}'
        outdir.mkdir(parents=True, exist_ok=True)
        for r in tgt:
            dest = outdir / (r['id'] + '.pt')
            if dest.exists():
                continue
            v = torch.load(OLD / 'training_cache' / (r['id'] + '.pt'),
                           weights_only=True)
            with torch.inference_mode():
                log = m.head(m.neck([x.cuda() for x in v['features']],
                                    16, 20), 16, 20)
            log = log.clamp(-10, 10).squeeze(0).cpu()
            assert torch.isfinite(log).all()
            assert tuple(log.shape) == tuple(v['log_gt'].shape)
            torch.save({'log_pseudo': log}, dest)
        for r in tgt:
            record[f'{seed}/{r["id"]}'] = sha(outdir / (r['id'] + '.pt'))
        print('PSEUDO', seed, flush=True)
    out = ST / 'pseudo_labels.json'
    if not out.exists():
        dump(out, {'teacher': 'source_only seed-matched final checkpoint',
                   'protocol_sha256': sha(ST / 'protocol.json'),
                   'label_sha256': record})
    print('FROZEN pseudo-labels', flush=True)


# ── phase 2: training ───────────────────────────────────────────────────────
def train():
    p = ensure_protocol()
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    rows = read(O / 'protocol.json')['train_rows']
    cached = {r['id']: torch.load(OLD / 'training_cache' / (r['id'] + '.pt'),
                                  weights_only=True) for r in rows}
    for seed in SEEDS:
        for r in rows:
            if r['kind'] != 'target':
                continue
            pl = torch.load(ST / 'pseudo_labels' / f'seed_{seed}' /
                            (r['id'] + '.pt'), weights_only=True)
            e = cached[r['id']]
            cached[(seed, r['id'])] = {
                'features': e['features'], 'log_gt': pl['log_pseudo'],
                'valid': torch.ones_like(e['valid'])}
    src = [r['id'] for r in rows if r['kind'] == 'source']
    tgt = [r['id'] for r in rows if r['kind'] == 'target']
    out = ST / 'trained_models'
    out.mkdir(exist_ok=True)
    for seed in SEEDS:
        for regime in SELF_REGIMES:
            dest = out / f'{regime}_{seed}.pt'
            if dest.exists():
                continue
            # Identical seeding order to train_major_revision_baseline:
            # manual_seed -> sampler rng -> model -> conv3 reinit.
            torch.manual_seed(seed)
            rng = np.random.default_rng(seed)
            m = model().cuda()
            for x in m.backbone.parameters():
                x.requires_grad_(False)
            if regime == 'self_train':
                nn.init.normal_(m.head.conv3.weight, 0, .001)
                nn.init.constant_(m.head.conv3.bias, np.log(50))
            else:
                state = torch.load(
                    OLD / 'trained_models' / f'source_only_{seed}.pt',
                    map_location='cpu', weights_only=True)
                m.neck.load_state_dict(state['neck'])
                m.head.load_state_dict(state['head'])
            m.neck.train()
            m.head.train()
            pars = [x for x in m.parameters() if x.requires_grad]
            opt = torch.optim.AdamW(pars, lr=1e-4, weight_decay=.01)
            log, draws, start = [], [], time.monotonic()
            for step in range(600):
                ids = [src[rng.integers(len(src))],
                       tgt[rng.integers(len(tgt))]]
                draws.append(ids)
                batch = [cached[ids[0]], cached[(seed, ids[1])]]
                features = [torch.cat([x['features'][i] for x in batch]).cuda()
                            for i in range(len(batch[0]['features']))]
                pred = m.head(m.neck(features, 16, 20), 16, 20)
                gt = torch.stack([x['log_gt'] for x in batch]).cuda()
                valid = torch.stack([x['valid'] for x in batch]).cuda()
                loss = torch.stack([nn.functional.smooth_l1_loss(
                    pred[i][valid[i]], gt[i][valid[i]], beta=.1)
                    for i in range(2)]).mean()
                assert torch.isfinite(loss)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(pars, 1.)
                opt.step()
                if step % 100 == 0 or step == 599:
                    log.append({'step': step + 1,
                                'loss': float(loss.detach())})
                    print(regime, seed, step + 1, log[-1]['loss'], flush=True)
            torch.save({'neck': m.neck.state_dict(),
                        'head': m.head.state_dict(), 'seed': seed,
                        'regime': regime, 'steps': 600,
                        'trainable_parameters': sum(x.numel() for x in pars),
                        'protocol_sha256': sha(ST / 'protocol.json')}, dest)
            dump(out / f'{regime}_{seed}_log.json',
                 {'loss': log, 'draws': draws,
                  'elapsed_seconds': time.monotonic() - start})
            del m, opt
            torch.cuda.empty_cache()
    dump(ST / 'training_complete.json',
         {'checkpoints': {q.name: sha(q) for q in out.glob('*.pt')},
          'protocol_sha256': sha(ST / 'protocol.json')})


# ── phase 3: inference ──────────────────────────────────────────────────────
def inference():
    ensure_protocol()
    torch.set_num_threads(4)
    m = model().cuda().eval()
    for seed in SEEDS:
        for regime in SELF_REGIMES:
            state = torch.load(ST / 'trained_models' / f'{regime}_{seed}.pt',
                               map_location='cpu', weights_only=True)
            assert state['protocol_sha256'] == sha(ST / 'protocol.json')
            m.neck.load_state_dict(state['neck'])
            m.head.load_state_dict(state['head'])
            out = ST / 'predictions' / f'{regime}_{seed}'
            out.mkdir(parents=True, exist_ok=True)
            for r in frames():
                dest = out / (r['id'] + '.npz')
                if dest.exists():
                    continue
                v = torch.load(OLD / 'test_features' /
                               (r['id'] + '_identity.pt'), weights_only=True)
                with torch.inference_mode():
                    log = m.head(m.neck([x.cuda() for x in v['features']],
                                        16, 20), 16, 20)
                pred = torch.exp(log.clamp(-10, 10)).squeeze().cpu().numpy()
                assert np.isfinite(pred).all()
                np.savez_compressed(dest, prediction=pred)
            print('SELFTRAIN INFERENCE', regime, seed, flush=True)


# ── phase 4: scoring ────────────────────────────────────────────────────────
def score():
    ensure_protocol()
    outs = ['selftrain_per_frame.csv', 'selftrain_per_object.json',
            'selftrain_summary.json']
    for name in outs:
        if (ST / name).exists():
            raise ValueError('Refusing to overwrite ' + str(ST / name))
    rows = []
    for r in frames():
        gt, v = reference(r)
        assert v.any()
        for regime in SELF_REGIMES:
            for seed in SEEDS:
                with np.load(ST / 'predictions' / f'{regime}_{seed}' /
                             (r['id'] + '.npz')) as f:
                    raw = cv2.resize(f['prediction'], gt.shape[::-1],
                                     interpolation=cv2.INTER_LINEAR)
                for bound in [100, 300]:
                    pred = np.clip(raw, 1, bound)
                    rows.append({'cohort': cohort(r),
                                 'object': str(r['object']),
                                 'frame': r['id'], 'method': regime,
                                 'seed': seed, 'bound': bound,
                                 'pixels': int(v.sum()),
                                 **metrics(pred, gt, v)})
        print('SCORE', r['id'], flush=True)
    groupkeys = ['cohort', 'method', 'seed', 'bound']
    buckets = {}
    for r in rows:
        buckets.setdefault(
            tuple(r[k] for k in groupkeys) + (r['object'],), []).append(r)
    objects = []
    for key, rr in sorted(buckets.items()):
        objects.append({**dict(zip(groupkeys + ['object'], key)),
                        'frames': len(rr),
                        'pixels': sum(r['pixels'] for r in rr),
                        **{k: float(np.mean([r[k] for r in rr]))
                           for k in KEYS}})
    buckets = {}
    for r in objects:
        buckets.setdefault(tuple(r[k] for k in groupkeys), []).append(r)
    summary = [{**dict(zip(groupkeys, key)),
                'groups': len(rr),
                'frames': sum(r['frames'] for r in rr),
                'pixels': sum(r['pixels'] for r in rr),
                **{k: float(np.mean([r[k] for r in rr])) for k in KEYS}}
               for key, rr in sorted(buckets.items())]
    writecsv(ST / 'selftrain_per_frame.csv', rows)
    dump(ST / 'selftrain_per_object.json', objects)
    dump(ST / 'selftrain_summary.json', summary)

    # Context table against the frozen cmig_revision regimes.
    import statistics
    old = read(O / 'comparison_summary.json')
    print('\n=== self-training in context (bound=100, mean +/- SD seeds) ===')
    for c in ['c3vd', 'scared', 'servct', 'drends_phantom', 'drends_exvivo']:
        line = f'{c:16s}'
        for m in ['source_only', 'adapted', 'target_only']:
            vals = [x['mae_mm'] for x in old
                    if x['cohort'] == c and x['method'] == m
                    and x['bound'] == 100]
            line += f'  {m}={statistics.mean(vals):7.3f}'
        for m in SELF_REGIMES:
            vals = [x['mae_mm'] for x in summary
                    if x['cohort'] == c and x['method'] == m
                    and x['bound'] == 100]
            line += (f'  {m}={statistics.mean(vals):7.3f}'
                     f'±{statistics.stdev(vals):.3f}')
        print(line, flush=True)


# ── phase 5: verification ───────────────────────────────────────────────────
def check_selftrain():
    p = ensure_protocol()
    complete = read(ST / 'training_complete.json')
    assert len(complete['checkpoints']) == 2 * len(SEEDS)
    counts = []
    for name, h in complete['checkpoints'].items():
        path = ST / 'trained_models' / name
        assert sha(path) == h
        s = torch.load(path, map_location='cpu', weights_only=True)
        assert s['protocol_sha256'] == sha(ST / 'protocol.json')
        assert s['trainable_parameters'] == 2728513
        log = read(path.with_name(path.stem + '_log.json'))
        draws = log['draws']
        src = sum(x.startswith('source_') for batch in draws for x in batch)
        tgt = sum(x.startswith('target_') for batch in draws for x in batch)
        assert (src, tgt, len(draws)) == (600, 600, 600)
        assert all(np.isfinite(x['loss']) for x in log['loss'])
        counts.append({'checkpoint': name, 'source_draws': src,
                       'target_draws': tgt, 'steps': len(draws)})
    # The parent major_revision run did not record frame draws, so equality
    # with the adapted regime's draws cannot be checked post-hoc; it holds
    # by identical construction (same manual_seed/rng sequence and draw
    # order). Only the draw-count contract is verified here.
    for seed in SEEDS:
        own = read(ST / 'trained_models' / f'self_train_{seed}_log.json')
        assert len(own['draws']) == 600
    labels = read(ST / 'pseudo_labels.json')
    for key, h in labels['label_sha256'].items():
        seed, frame = key.split('/', 1)
        assert sha(ST / 'pseudo_labels' / f'seed_{seed}' / (frame + '.pt')) == h
    rr = list(csv.DictReader((ST / 'selftrain_per_frame.csv').open()))
    ss = read(ST / 'selftrain_summary.json')
    gkeys = ['cohort', 'method', 'seed', 'bound']
    keys = [tuple(r[k] for k in gkeys + ['frame']) for r in rr]
    assert len(set(keys)) == len(keys)
    for s in ss:
        a = [r for r in rr if all(r[k] == str(s[k]) for k in gkeys)]
        objects = sorted({r['object'] for r in a})
        val = np.mean([np.mean([float(r['mae_mm']) for r in a
                                if r['object'] == obj]) for obj in objects])
        assert np.isclose(val, s['mae_mm'], rtol=1e-12)
        assert len(objects) == s['groups']
    assert all(np.isfinite(float(r['mae_mm'])) for r in rr)
    checks = {
        'protocol_hashes_verified': True,
        'six_checkpoints_verified': True,
        'pseudo_label_hashes_verified': True,
        'training_draw_counts_verified': True,
        'summaries_reaggregated': True,
        'per_frame_rows': len(rr),
        'training_counts': counts,
        'no_reference_used_for_pseudo_labels': True,
        'no_test_performance_selection': True,
    }
    dump(ST / 'validation_checks.json', checks)
    print(json.dumps(checks, indent=2), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('phase', choices=['pseudo', 'train', 'inference',
                                      'score', 'check_selftrain'])
    a = ap.parse_args()
    globals()[a.phase]()
