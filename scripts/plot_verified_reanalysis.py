"""Render publication figures only from the independently rerun result manifest."""

from pathlib import Path
import json
import sys
import csv

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "outputs" / "verified_reanalysis"
OUT = ROOT / "outputs" / "verified_figures"
SEED = 20260924
result = json.loads((DATA / "verified_reanalysis.json").read_text())
manifest = json.loads((DATA / "manifest.json").read_text())

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
    "font.size": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": .8,
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
    "legend.frameon": False,
})

sys.path.insert(0, "/home/suyuxiong/.config/opencode/skills/nature-skills/skills/nature-figure/scripts")
from audit_panel_alignment import require_matplotlib_panel_alignment

COLORS = {
    "pca_full": "#4D4D4D",
    "pca_q01_q99": "#3775BA",
    "mad_pca_q01_q99": "#42949E",
    "so3_quantile_volume": "#B64342",
}
METHOD_LABELS = {
    "pca_full": "PCA full support",
    "pca_q01_q99": "PCA 1–99% extent",
    "mad_pca_q01_q99": "MAD-PCA 1–99%",
    "so3_quantile_volume": "SO(3) quantile-volume",
}


def bootstrap_ci(values, seed, draws=10000):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    draw = rng.choice(values, size=(draws, len(values)), replace=True).mean(axis=1)
    return [float(np.quantile(draw, .025)), float(np.quantile(draw, .975))]


def save(fig, basename, multi=False):
    OUT.mkdir(parents=True, exist_ok=True)
    if multi:
        require_matplotlib_panel_alignment(
            fig,
            json_out=str(OUT / f"{basename}.alignment.json"),
            overlay_svg=str(OUT / f"{basename}.alignment.svg"),
            tolerance_pt=1.5,
            gutter_tolerance_pt=1.5,
            require_panel_labels=True,
            strict=True,
        )
    fig.savefig(OUT / f"{basename}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{basename}.svg", bbox_inches="tight")
    fig.savefig(OUT / f"{basename}.png", dpi=600, bbox_inches="tight")
    fig.savefig(OUT / f"{basename}.tiff", dpi=600, bbox_inches="tight")
    plt.close(fig)


def fig1_factorial():
    summary = result["procedural"]["summary"]
    scales = ["oracle", "prior", "tool", "fusion"]
    methods = list(METHOD_LABELS)
    matrix = np.array([[next(r["mae_mm"] for r in summary if r["scale_mode"] == s and r["method"] == m) for m in methods] for s in scales])
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(np.arange(len(methods)), ["PCA\nfull", "PCA\nq01–99", "MAD-PCA\nq01–99", "SO(3)\nq-volume"])
    ax.set_yticks(np.arange(len(scales)), [s.capitalize() for s in scales])
    ax.set_xlabel("Geometry estimator")
    ax.set_ylabel("Depth/scale condition")
    ax.set_title("Procedural MAE: scale and geometry factors separated", loc="left", fontweight="bold")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            color = "white" if matrix[i, j] > np.median(matrix) else "black"
            ax.text(j, i, f"{matrix[i,j]:.2f}", ha="center", va="center", color=color, fontsize=8)
    cbar = fig.colorbar(im, ax=ax, pad=.02)
    cbar.set_label("MAE (mm)")
    ax.text(0, -0.35, "50 procedural cases; bootstrap intervals in source data", transform=ax.transAxes, fontsize=7)
    save(fig, "Fig1_factorial_scale_geometry")


def fig2_paired():
    paired = result["procedural"]["paired_tests"]
    scales = ["oracle", "prior", "tool", "fusion"]
    methods = ["pca_q01_q99", "mad_pca_q01_q99", "so3_quantile_volume"]
    fig, axes = plt.subplots(1, 4, figsize=(9.2, 3.2), sharey=True)
    for ax, scale in zip(axes, scales):
        rows = [r for r in paired if r["scale_mode"] == scale]
        values, intervals = [], []
        for m in methods:
            row = next(r for r in rows if r["comparison"].startswith(m + " vs"))
            values.append(row["mean_paired_abs_error_difference_mm"])
            intervals.append(row["paired_difference_bootstrap_95ci"])
        vals = values
        pvals = [next(r["wilcoxon_p_holm"] for r in rows if r["comparison"].startswith(m + " vs")) for m in methods]
        ax.axhline(0, color="#222222", lw=.8)
        ax.bar(np.arange(3), vals, color=[COLORS[m] for m in methods], width=.48)
        lower = np.array(vals) - np.array([x[0] for x in intervals])
        upper = np.array([x[1] for x in intervals]) - np.array(vals)
        ax.errorbar(np.arange(3), vals, yerr=[lower, upper], fmt="none", ecolor="#222222", capsize=2, lw=.7)
        ax.set_xticks(np.arange(3), ["PCA q", "MAD-PCA", "SO(3)"], rotation=25, ha="right", rotation_mode="anchor")
        ax.set_title(scale.capitalize(), fontweight="bold")
    for ax, label in zip(axes, "abcd"):
        ax.text(-0.15, 1.08, label, transform=ax.transAxes, fontsize=10, fontweight="bold", va="top", ha="left")
    # Exact adjusted p-values are available in the source-data JSON; keeping
    # them out of the dense panel prevents text from colliding with intervals.
    axes[0].set_ylabel("Paired Δ absolute error vs full PCA (mm)")
    fig.suptitle("Estimator comparisons are scale-condition dependent", x=.03, ha="left", fontsize=10, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, .94])
    save(fig, "Fig2_paired_geometry_comparisons", multi=True)


def fig3_tool_scale():
    with (DATA / "procedural_factorial_trials.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    distances = [20.0, 30.0, 45.0]
    modes = ["prior", "tool", "fusion"]
    methods = ["pca_full", "so3_quantile_volume"]
    fig, axes = plt.subplots(1, 3, figsize=(8.8, 3.3), sharey=True)
    for ax, distance in zip(axes, distances):
        for offset, (scale, method) in enumerate([(s, m) for s in modes for m in methods]):
            values = np.array([
                float(r["abs_error_mm"]) for r in rows
                if abs(float(r["working_distance_mm"]) - distance) < 5
                and r["scale_mode"] == scale and r["method"] == method
            ])
            if values.size == 0:
                continue
            mean = values.mean()
            ci = bootstrap_ci(values, SEED + int(distance) + offset)
            x = offset
            color = COLORS[method]
            ax.scatter(x, mean, color=color, s=25, zorder=3)
            ax.vlines(x, ci[0], ci[1], color=color, lw=1.2)
            ax.hlines([ci[0], ci[1]], x - .08, x + .08, color=color, lw=.8)
        ax.set_xticks(np.arange(6), ["Prior\nPCA", "Prior\nSO3", "Tool\nPCA", "Tool\nSO3", "Fuse\nPCA", "Fuse\nSO3"], fontsize=6)
        ax.set_title(f"{distance:.0f} mm working distance", fontweight="bold")
        ax.grid(axis="y", color="#DDDDDD", lw=.5)
    for ax, label in zip(axes, "abc"):
        ax.text(-0.15, 1.08, label, transform=ax.transAxes, fontsize=10, fontweight="bold", va="top", ha="left")
    axes[0].set_ylabel("Absolute major-axis error (mm)")
    fig.suptitle("Scale-mode error by working distance", x=.03, ha="left", fontsize=10, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, .92])
    save(fig, "Fig3_scale_error_stratification", multi=True)


def main():
    print("source result manifest seed:", manifest["seed"])
    fig1_factorial()
    fig2_paired()
    fig3_tool_scale()
    print("Wrote verified figures to", OUT)


if __name__ == "__main__":
    main()
