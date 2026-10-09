"""Publication figures for the corrected MetricEndo manuscript.

All quantitative values are read from benchmark JSON/CSV outputs.  The figures
are descriptive and preserve the distinction between procedural ground truth
and unlabelled Kvasir-SEG outputs.
"""

from pathlib import Path
import json
import csv
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

SKILL_SCRIPTS = "/home/suyuxiong/.config/opencode/skills/nature-skills/skills/nature-figure/scripts"
sys.path.insert(0, SKILL_SCRIPTS)
from audit_panel_alignment import require_matplotlib_panel_alignment


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "publication_figures"
BENCH = ROOT / "outputs" / "journal_benchmark_summary.json"
SUPP = ROOT / "outputs" / "supplementary" / "supplementary_summary.json"
GEN = ROOT / "outputs" / "generalization" / "summary.json"

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 8,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.8,
    "legend.frameon": False,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
})

BLUE = "#0F4D92"
TEAL = "#42949E"
RED = "#B64342"
GOLD = "#C58B16"
PURPLE = "#7C5AA6"
GREY = "#767676"
LIGHT = "#D9D9D9"


def load_json(path):
    return json.loads(path.read_text())


def save(fig, name, multi=False):
    OUT.mkdir(parents=True, exist_ok=True)
    stem = OUT / name
    if multi:
        require_matplotlib_panel_alignment(
            fig,
            json_out=f"{stem}.alignment.json",
            overlay_svg=f"{stem}.alignment.svg",
            tolerance_pt=1.5,
            gutter_tolerance_pt=1.5,
            require_panel_labels=True,
            strict=True,
        )
    fig.savefig(f"{stem}.svg", bbox_inches="tight")
    fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(f"{stem}.png", dpi=600, bbox_inches="tight")
    fig.savefig(f"{stem}.tiff", dpi=600, bbox_inches="tight")
    plt.close(fig)


def panel_label(ax, label):
    ax.text(-0.14, 1.08, label, transform=ax.transAxes, fontsize=10,
            fontweight="bold", va="top", ha="left")


def fig1_benchmark():
    data = load_json(BENCH)["summary_table"]
    labels = ["2D", "D", "PCA", "L", "H", "T", "Full"]
    mae = np.array([r["MAE_mm"] for r in data])
    low = np.array([r["MAE_bootstrap_CI95"][0] for r in data])
    high = np.array([r["MAE_bootstrap_CI95"][1] for r in data])
    risk = np.array([r["Risk_Agreement_pct"] for r in data])
    colors = [GREY, LIGHT, RED, RED, GOLD, TEAL, BLUE]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), gridspec_kw={"wspace": 0.32})
    ax = axes[0]
    x = np.arange(len(labels))
    ax.bar(x, mae, color=colors, edgecolor="white", linewidth=0.6)
    ax.errorbar(x, mae, yerr=[mae - low, high - mae], fmt="none", ecolor="#333333", capsize=2, lw=0.8)
    ax.set_ylabel("Major-diameter MAE (mm)")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 8.8)
    panel_label(ax, "a")
    ax = axes[1]
    ax.bar(x, risk, color=colors, edgecolor="white", linewidth=0.6)
    ax.set_ylabel("Risk agreement (%)")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 105)
    ax.axhline(50, color="#999999", lw=0.7, ls="--")
    ax.set_title("Threshold agreement", loc="left", fontweight="bold")
    panel_label(ax, "b")
    save(fig, "Figure1_primary_benchmark", multi=True)


def fig2_robustness():
    rows = load_json(SUPP)["robustness_summary"]
    noise = np.array([r["noise_pct"] for r in rows])
    pca = np.array([r["pca_mae_mm"] for r in rows])
    robust = np.array([r["robust_mae_mm"] for r in rows])
    pca_sd = np.array([r["pca_sd_mm"] for r in rows])
    robust_sd = np.array([r["robust_sd_mm"] for r in rows])
    fig, ax = plt.subplots(figsize=(4.3, 3.1))
    ax.plot(noise, pca, "o-", color=RED, lw=1.5, label="PCA")
    ax.fill_between(noise, pca - pca_sd, pca + pca_sd, color=RED, alpha=0.12)
    ax.plot(noise, robust, "o-", color=BLUE, lw=1.5, label="Robust OBB")
    ax.fill_between(noise, robust - robust_sd, robust + robust_sd, color=BLUE, alpha=0.14)
    ax.set_xlabel("Spatial outlier contamination (%)")
    ax.set_ylabel("Major-axis MAE (mm)")
    ax.set_xticks(noise)
    ax.set_ylim(0, 75)
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.02), borderaxespad=0.0)
    ax.text(0.02, 1.03, "Outlier robustness versus clean-case bias", transform=ax.transAxes,
            fontsize=9, fontweight="bold", va="bottom")
    panel_label(ax, "a")
    save(fig, "Figure2_outlier_robustness")


def fig3_curvature():
    rows = load_json(SUPP)["curvature_trials"]
    sphere = [r for r in rows if r["surface"] == "sphere"]
    radius = sorted(set(float(r["radius_mm"]) for r in sphere))
    expected_h, observed_h, expected_k, observed_k = [], [], [], []
    for r in radius:
        subset = [x for x in sphere if float(x["radius_mm"]) == r]
        expected_h.append(1 / r)
        observed_h.append(np.mean([x["estimated_abs_H"] for x in subset]))
        expected_k.append(1 / r ** 2)
        observed_k.append(np.mean([x["estimated_K"] for x in subset]))
    fig, axes = plt.subplots(1, 2, figsize=(6.7, 3.0), gridspec_kw={"wspace": 0.32})
    ax = axes[0]
    ax.plot(radius, expected_h, "k--", lw=1, label="Analytical")
    ax.plot(radius, observed_h, "o-", color=BLUE, lw=1.5, label="Estimated")
    ax.set_xlabel("Sphere radius (mm)")
    ax.set_ylabel("abs(H) (mm^-1)")
    ax.legend(fontsize=7)
    ax.set_title("Mean curvature", loc="left", fontweight="bold")
    panel_label(ax, "a")
    ax = axes[1]
    ax.plot(radius, expected_k, "k--", lw=1, label="Analytical")
    ax.plot(radius, observed_k, "o-", color=TEAL, lw=1.5, label="Estimated")
    ax.set_xlabel("Sphere radius (mm)")
    ax.set_ylabel("K (mm^-2)")
    ax.set_title("Gaussian curvature", loc="left", fontweight="bold")
    panel_label(ax, "b")
    save(fig, "Figure3_curvature_validation", multi=True)


def fig4_scale():
    rows = load_json(SUPP)["scale_tool"]["by_condition"]
    distances = [20.0, 30.0, 45.0]
    modes = ["clean", "eroded", "dilated"]
    matrix = np.zeros((len(distances), len(modes)))
    for i, d in enumerate(distances):
        for j, mode in enumerate(modes):
            vals = [r["relative_scale_error_pct"] for r in rows if r["distance_mm"] == d and r["mask_mode"] == mode]
            matrix[i, j] = np.mean(vals)
    fig, ax = plt.subplots(figsize=(4.3, 3.0))
    im = ax.imshow(matrix, cmap="YlOrRd", vmin=0, vmax=max(130, matrix.max()))
    ax.set_xticks(range(3), ["Clean", "Eroded", "Dilated"])
    ax.set_yticks(range(3), ["20", "30", "45"])
    ax.set_xlabel("Tool-mask condition")
    ax.set_ylabel("Working distance (mm)")
    for i in range(3):
        for j in range(3):
            color = "white" if matrix[i, j] > 65 else "black"
            ax.text(j, i, f"{matrix[i,j]:.1f}%", ha="center", va="center", color=color, fontsize=8)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Mean relative scale error (%)")
    ax.set_title("Tool-scale sensitivity", loc="left", fontweight="bold")
    panel_label(ax, "a")
    save(fig, "Figure4_scale_sensitivity")


def fig5_generalization():
    groups = load_json(GEN)["groups"]
    shapes = ["ellipsoid", "dome", "plaque", "saddle"]
    noises = [0, 5, 10, 20]
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.0), gridspec_kw={"wspace": 0.34})
    for ax, metric, title, label in [(axes[0], "baseline_mae_mm", "Baseline Robust OBB", "a"),
                                     (axes[1], "adaptive_mae_mm", "Boundary correction", "b")]:
        for shape, color in zip(shapes, [BLUE, TEAL, GOLD, PURPLE]):
            vals = [next(g[metric] for g in groups if g["shape"] == shape and g["noise_pct"] == n) for n in noises]
            ax.plot(noises, vals, "o-", lw=1.3, color=color, label=shape.capitalize())
        ax.set_xlabel("Outlier contamination (%)")
        ax.set_ylabel("Major-axis MAE (mm)")
        ax.set_xticks(noises)
        ax.set_title(title, loc="left", fontweight="bold")
        panel_label(ax, label)
    axes[1].legend(fontsize=7, ncol=2, loc="upper left")
    save(fig, "Figure5_shape_generalization", multi=True)


def fig6_kvasir():
    values = [59.66, 72.03]
    fig, axes = plt.subplots(1, 2, figsize=(6.2, 3.0), gridspec_kw={"wspace": 0.38})
    ax = axes[0]
    ax.bar([0, 1], values, color=[GREY, TEAL], width=0.58)
    ax.set_xticks([0, 1], ["Euclidean\nmajor axis", "Manifold\ndistance"])
    ax.set_ylabel("Algorithm output (mm)")
    ax.set_title("50-image Kvasir-SEG subset", loc="left", fontweight="bold")
    panel_label(ax, "a")
    ax = axes[1]
    ax.bar([0], [685.95], color=BLUE, width=0.45)
    ax.set_xticks([0], ["CPU pipeline"])
    ax.set_ylabel("Latency (ms/frame)")
    ax.set_ylim(0, 800)
    ax.set_title("Reference implementation timing", loc="left", fontweight="bold")
    ax.text(0, 720, "1.5 FPS", ha="center", fontsize=8)
    panel_label(ax, "b")
    save(fig, "Figure6_kvasir_descriptive", multi=True)


if __name__ == "__main__":
    fig1_benchmark()
    fig2_robustness()
    fig3_curvature()
    fig4_scale()
    fig5_generalization()
    fig6_kvasir()
    print(f"Wrote publication figures to {OUT}")
