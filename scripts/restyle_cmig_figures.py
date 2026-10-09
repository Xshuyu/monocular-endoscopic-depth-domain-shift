"""Restyle the six manuscript figures in place (style only; data unchanged).

Reads existing outputs/, never modifies them. Overwrites
manuscript_cmig/Figure_{1..6}.pdf and manuscript_mbec/Figure_{1..6}.pdf,
and writes PNG previews to /tmp/restyle_preview/.

Mapping: 1=radial_architecture, 2=test_per_video, 3=test_radius_depth,
4=scared_subject_results, 5=major_revision_results, 6=factorial_effects.
"""
from pathlib import Path
import json, csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
PREVIEW = Path('/tmp/restyle_preview'); PREVIEW.mkdir(exist_ok=True)

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Liberation Serif', 'Nimbus Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'pdf.fonttype': 42,
    'svg.fonttype': 'none',
    'font.size': 9,
    'axes.linewidth': 0.8,
})

def save(fig, name):
    # This generator is scoped to the active CMIG manuscript. Do not overwrite
    # figures belonging to the separate MBEC submission.
    fig.savefig(ROOT / 'manuscript_cmig' / f'{name}.pdf', facecolor='white')
    if name == 'Figure_2':
        # Flat submission package uses the same six figure files as its source.
        fig.savefig(ROOT / 'manuscript_cmig' / 'submission' / f'{name}.pdf',
                    facecolor='white')
    fig.savefig(PREVIEW / f'{name}.png', dpi=100, facecolor='white')
    plt.close(fig)
    print(f'saved {name}')

def read_csv(p):
    with p.open() as f:
        return list(csv.DictReader(f))

def tidy(ax):
    ax.spines[['top', 'right']].set_visible(False)

# ---------------------------------------------------------------- Figure 1
def figure_1():
    # Compact the three workflow stages vertically and use the available
    # figure area for larger, readable labels.
    W, H = 1200, 700
    lanes = [
        ('A  Development and calibration', 20, 15, 1160, 180, '#edf4fb', '#245b85', '-'),
        ('B  Frozen new-sequence evaluation', 20, 225, 1160, 255, '#eef6f2', '#22624d', '-'),
        ('C  Optional size measurement', 20, 510, 1160, 165, '#faf5eb', '#8a612a', '--'),
    ]
    boxes = [
        dict(xy=(40, 58, 340, 112), color='#245b85', title='Development inputs',
             lines=['C3VDv2: 52 frames / 6 scenes', 'Frozen DA V2 Small; reference depth']),
        dict(xy=(430, 58, 340, 112), color='#245b85', title='Matched calibration heads',
             lines=['Affine LS / ridge; radial LS / ridge', '16,384 pixels; train-only scaling']),
        dict(xy=(820, 58, 340, 112), color='#245b85', title='Grouped selection and freeze',
             lines=['Nested leave-scene-out selection', 'Refit on development; freeze coefficients']),
        dict(xy=(40, 280, 340, 145), color='#22624d', title='New RGB input',
             lines=['C3VD: 4 videos / 48 frames', 'Frozen DA V2 Small', 'No test reference at inference']),
        dict(xy=(430, 280, 340, 145), color='#22624d', title='Frozen depth calibration',
             lines=['Image-center radius $r^2$',
                    '$q=(a_0+a_1r^2)d+b_0+b_1r^2$',
                    'Invert and clip to 1–100 mm']),
        dict(xy=(820, 280, 340, 145), color='#22624d', title='Native-grid evaluation',
             lines=['MAE / RMSE / AbsRel / $\\delta_1$',
                    'Equal-video aggregation', 'Radius × depth diagnostics']),
        dict(xy=(820, 438, 340, 30), color='#22624d',
             title='Reference depth: scoring only', lines=[], compact=True),
        dict(xy=(40, 550, 340, 95), color='#8a612a', title='Additional inputs',
             lines=['Lesion mask; calibrated rays']),
        dict(xy=(430, 550, 340, 95), color='#8a612a', title='Research size endpoint',
             lines=['Ray back-projection; robust PCA extent']),
        dict(xy=(820, 550, 340, 95), color='#8a612a', title='Evidence boundary',
             lines=['Not a clinical caliper; not clinically validated']),
    ]
    edges = [
        ([(380, 114), (430, 114)], False),
        ([(770, 114), (820, 114)], False),
        ([(990, 170), (990, 225), (210, 225), (210, 280)], False),
        ([(380, 352), (430, 352)], False),
        ([(770, 352), (820, 352)], False),
        ([(990, 438), (990, 425)], False),
        ([(600, 425), (600, 550)], True),
        ([(380, 598), (430, 598)], True),
        ([(770, 598), (820, 598)], True),
    ]
    fig, ax = plt.subplots(figsize=(12, 7))
    fig.subplots_adjust(0, 0, 1, 1)
    ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis('off')
    for title, x, y, w, h, fill, color, ls in lanes:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=12',
                                    facecolor=fill, edgecolor=color, linewidth=1.2, linestyle=ls))
        ax.text(x + 20, y + 32, title, fontsize=15, fontweight='bold', color=color)
    for b in boxes:
        x, y, w, h = b['xy']
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=8',
                                    facecolor='white', edgecolor=b['color'], linewidth=1.35))
        title_offset = h / 2 if b.get('compact', False) else 30
        ax.text(x + w / 2, y + title_offset, b['title'], ha='center', va='center',
                fontsize=13.5, fontweight='bold', color=b['color'])
        y0, step = y + 61, 24
        for j, line in enumerate(b['lines']):
            ax.text(x + w / 2, y0 + j * step, line, ha='center', va='center',
                    fontsize=12.2, color='#243442')
    for points, dashed in edges:
        color = '#8a612a' if dashed else '#425666'
        for a, b in zip(points[:-2], points[1:-1]):
            ax.plot([a[0], b[0]], [a[1], b[1]], color=color, linewidth=1.7,
                    linestyle='--' if dashed else '-')
        ax.add_patch(FancyArrowPatch(points[-2], points[-1], arrowstyle='-|>',
                                     mutation_scale=15, color=color, linewidth=1.9,
                                     linestyle='--' if dashed else '-'))
    ax.text(790, 216, 'Frozen heads', ha='center', va='bottom', fontsize=12,
            color='#425666', bbox={'facecolor': 'white', 'edgecolor': 'none', 'pad': 1.5})
    # The scoring-only reference label belongs to the lower edge of the
    # evaluation lane and is placed just inside that lane, aligned to the
    # reference-depth box rather than floating at its former box center.
    ax.text(990, 463, 'Reference depth: scoring only', ha='center', va='center',
            fontsize=11.5, color='#22624d',
            bbox={'facecolor': 'white', 'edgecolor': 'none', 'pad': 1.5})
    ax.text(620, 486, 'Predicted axial depth $Z$', va='center', fontsize=12, color='#8a612a')
    save(fig, 'Figure_1')

# ---------------------------------------------------------------- Figure 2
METHODS = ['affine_ls', 'affine_ridge', 'radial_ls', 'radial_ridge']
MLABELS = ['Affine LS', 'Affine ridge', 'Radial LS', 'Radial ridge']
MCOLORS = ['#4477aa', '#ee7733', '#228833', '#cc3311']

def figure_2():
    rows = read_csv(ROOT / 'outputs/radial_confirmatory/test_per_video.csv')
    scenes = sorted({r['scene'] for r in rows})
    lookup = {(r['scene'], r['method']): float(r['mae_mm']) for r in rows}
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    x = np.arange(len(scenes))
    width = 0.19
    offsets = (np.arange(len(METHODS)) - (len(METHODS) - 1) / 2) * width
    for j, (m, label, color) in enumerate(zip(METHODS, MLABELS, MCOLORS)):
        vals = [lookup[s, m] for s in scenes]
        ax.bar(x + offsets[j], vals, width=width * 0.94, label=label,
               color=color, edgecolor='white', linewidth=0.35, zorder=3)
    ax.set_xticks(x, ['Cecum T2', 'Cecum T3', 'Sigmoid T2', 'Transverse T2'])
    ax.set_ylim(bottom=0)
    ax.set_ylabel('MAE (mm)')
    ax.legend(ncol=2, fontsize=8.5, frameon=False, loc='upper right',
              columnspacing=1.2, handletextpad=0.6)
    ax.grid(axis='y', color='#d9d9d9', linewidth=0.55, zorder=0)
    tidy(ax)
    fig.tight_layout()
    save(fig, 'Figure_2')
    print({(s, m): lookup[s, m] for s in scenes for m in METHODS})

# ---------------------------------------------------------------- Figure 3
def figure_3():
    rows = read_csv(ROOT / 'outputs/radial_confirmatory/test_strata_summary.csv')
    val = {(r['method'], int(r['radius_bin']), int(r['depth_bin'])): float(r['mae_mm'])
           for r in rows}
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.4), sharex=True, sharey=True)
    axes = axes.ravel()
    styles = [dict(color=MCOLORS[0], linestyle='-', linewidth=3, marker='o', markersize=8,
                   markerfacecolor='white', markeredgewidth=1.6, zorder=2),
              dict(color=MCOLORS[1], linestyle='--', linewidth=1.6, marker='x', markersize=6,
                   markeredgewidth=1.5, zorder=4),
              dict(color=MCOLORS[2], linestyle='-', linewidth=1.6, marker='^', markersize=6,
                   zorder=3),
              dict(color=MCOLORS[3], linestyle='-.', linewidth=1.6, marker='D', markersize=5,
                   zorder=3)]
    for ax, db, title in zip(axes, [-1, 0, 1, 2],
                             ['All depths', '0–10 mm', '10–30 mm', '30–100 mm']):
        for m, label, st in zip(METHODS, MLABELS, styles):
            vals = [val.get((m, rb, db), np.nan) for rb in range(3)]
            ax.plot(range(3), vals, label=label, **st)
        ax.set_xticks(range(3), ['Center', 'Middle', 'Edge'])
        ax.set_title(title, fontsize=10)
        tidy(ax)
    for ax in axes[:2]:
        ax.tick_params(labelbottom=False)
    axes[0].set_ylabel('Region MAE (mm)')
    axes[2].set_ylabel('Region MAE (mm)')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.045), ncol=4,
               fontsize=9, frameon=False)
    fig.text(0.5, 0.012,
             'Affine LS and affine ridge overlap (maximum difference < 0.000001 mm).',
             ha='center', fontsize=8.5)
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    save(fig, 'Figure_3')
    print({k: v for k, v in sorted(val.items()) if k[0] in METHODS})

# ---------------------------------------------------------------- Figure 4
def figure_4():
    rows = read_csv(ROOT / 'outputs/scared_expansion/per_subject.csv')
    subjects = [1, 2, 3, 6, 7, 8, 9]
    panels = [
        [('small_affine_ls', 'Small + affine LS', '#4477aa', 'o', '-'),
         ('small_radial_ls', 'Small + radial LS', '#228833', '^', '-')],
        [('small_radial_ridge', 'Small + radial ridge', '#cc3311', 'D', '-'),
         ('sgbm', 'SGBM (stereo)', '#ee7733', 's', '--'),
         ('raft_sceneflow', 'RAFT-Stereo (SceneFlow)', '#882255', 'v', '-.')],
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
    out = {}
    for ax, methods in zip(axes, panels):
        for m, label, color, marker, ls in methods:
            vals = [float(next(x for x in rows if int(x['subject']) == n and x['method'] == m
                               and x['support'] == 'all_reference')['mae_mm'])
                    for n in subjects]
            out[m] = vals
            ax.plot(range(7), vals, marker=marker, linestyle=ls, color=color, label=label,
                    linewidth=1.5, markersize=5)
        ax.set_xticks(range(7), [str(n) + ('*' if n in [8, 9] else '') for n in subjects])
        ax.set_xlabel('SCARED subject (* previously observed)')
        ax.legend(fontsize=8, frameon=False)
        tidy(ax)
    axes[0].set_ylabel('MAE (mm)')
    fig.tight_layout()
    save(fig, 'Figure_4')
    print(out)

# ---------------------------------------------------------------- Figure 5
def figure_5():
    O = ROOT / 'outputs/major_revision'
    objs = json.loads((O / 'metric_per_object.json').read_text())
    raw = read_csv(O / 'original_transfer_diagnostics.csv')

    def match(row, c):
        return row['cohort'] == c if c not in ['Phantom', 'ExVivo'] \
            else row['cohort'] == 'drends' and row['object'].startswith(c)

    def stats(c, regime, bound=100):
        vals = [np.mean([r['mae_mm'] for r in objs
                         if match(r, c) and r['regime'] == regime and r['seed'] == seed
                         and r['factor'] == 'identity' and r['bound'] == bound
                         and r['support'] == 'all_reference'])
                for seed in [11, 23, 47]]
        return float(np.mean(vals)), float(np.std(vals, ddof=1))

    def orig(c, bound=100, metric='mae_mm'):
        a = [r for r in raw if match(r, c) and r['method'] == 'small_radial_ridge'
             and int(r['bound']) == bound and r['support'] == 'all_reference']
        return float(np.mean([np.mean([float(r[metric]) for r in a if r['object'] == obj])
                              for obj in sorted({r['object'] for r in a})]))

    cohorts = ['c3vd', 'scared', 'servct', 'Phantom', 'ExVivo']
    summary = [{'cohort': c, 'orig': orig(c),
                'src': stats(c, 'source_only'), 'adp': stats(c, 'adapted')} for c in cohorts]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2),
                           gridspec_kw={'width_ratios': [1.4, 1]})
    x = np.arange(5); offsets = [-0.20, 0.0, 0.20]
    series = [
        ('Frozen radial ridge', '#777777', lambda r: (r['orig'], 0.0), 'o'),
        ('Source-only decoder', '#4477aa', lambda r: r['src'], 's'),
        ('Adapted decoder', '#cc3311', lambda r: r['adp'], '^'),
    ]
    for off, (label, color, get, marker) in zip(offsets, series):
        vals = [get(r) for r in summary]
        ax[0].errorbar(x + off, [v[0] for v in vals], yerr=[v[1] for v in vals],
                       fmt=marker, color=color, label=label, markersize=6,
                       capsize=2.5, linewidth=1.0, linestyle='none')
    ax[0].set_xticks(x, ['C3VD', 'SCARED\nheld out', 'SERV-CT', 'DRENDS\nphantom',
                         'DRENDS\nex vivo'])
    ax[0].set_ylabel('MAE (mm)')
    ax[0].set_title('(a) Transfer and adaptation', fontsize=10)
    ax[0].legend(fontsize=8, frameon=False)
    xs = np.arange(3)
    for metric, label, color, marker in [
            ('mae_mm', 'Unaligned', '#777777', 'o'),
            ('oracle_offset_mae_mm', 'Offset diagnostic', '#228833', 's'),
            ('oracle_scale_mae_mm', 'Scale diagnostic', '#aa8800', '^')]:
        ax[1].plot(xs, [orig(c, metric=metric) for c in ['c3vd', 'scared', 'servct']],
                   marker=marker, color=color, linewidth=1.5, markersize=5.5,
                   label=label)
    ax[1].set_xticks(xs, ['C3VD', 'SCARED', 'SERV-CT'])
    ax[1].set_ylabel('MAE (mm)')
    ax[1].set_title('(b) Error decomposition', fontsize=10)
    ax[1].legend(fontsize=8, frameon=False)
    for a in ax:
        tidy(a)
    fig.tight_layout()
    save(fig, 'Figure_5')
    print(summary)
    print({m: [orig(c, metric=m) for c in ['c3vd', 'scared', 'servct']]
           for m in ['mae_mm', 'oracle_offset_mae_mm', 'oracle_scale_mae_mm']})

# ---------------------------------------------------------------- Figure 6
def figure_6():
    effects = json.loads((ROOT / 'outputs/major_revision/factorial_effects.json').read_text())
    cohorts = ['c3vd', 'scared', 'servct']
    fig, ax = plt.subplots(figsize=(6.8, 3.4))
    out = {}
    for j, (key, label, color, marker) in enumerate([
            ('dark_main', 'Gamma darkening', '#4477aa', 'o'),
            ('blur_main', 'Gaussian blur', '#ee7733', 's'),
            ('interaction', 'Interaction', '#228833', '^')]):
        vals = [next(r for r in effects if r['cohort'] == c and r['regime'] == 'adapted')[key]
                for c in cohorts]
        out[key] = vals
        ax.errorbar(np.arange(3) + (j - 1) * 0.18, [v['mean'] for v in vals],
                    yerr=[v['sd_seeds'] for v in vals], fmt=marker, color=color, label=label,
                    markersize=6, capsize=3, linewidth=1.2, linestyle='none')
    ax.axhline(0, color='k', lw=0.8)
    ax.set_xticks(range(3), ['C3VD', 'SCARED held out', 'SERV-CT'])
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylabel('Paired change in MAE (mm)')
    ax.legend(ncol=3, fontsize=8, frameon=False, loc='upper center',
              bbox_to_anchor=(0.5, 1.14), columnspacing=1.2)
    tidy(ax)
    fig.tight_layout()
    save(fig, 'Figure_6')
    print(out)

if __name__ == '__main__':
    figure_1()
    figure_2()
    figure_3()
    figure_4()
    figure_5()
    figure_6()
