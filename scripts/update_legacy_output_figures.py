"""Replace legacy root-output figures with corrected benchmark visualizations."""

from pathlib import Path
import json

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
DATA = json.loads((OUT / "journal_benchmark_summary.json").read_text())

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
    "savefig.dpi": 600,
})

BLUE = "#0F4D92"
RED = "#B64342"
TEAL = "#42949E"
GREY = "#767676"
GOLD = "#C58B16"


def save(fig, name):
    fig.savefig(OUT / name, dpi=300)
    plt.close(fig)


def figure1_bland_altman():
    rows = DATA["per_sample"]
    true = np.array([r["true_major_mm"] for r in rows])
    pred = np.array([r["MetricEndo (Full pipeline)"] for r in rows])
    means = (true + pred) / 2
    diff = pred - true
    bias = float(np.mean(diff))
    sd = float(np.std(diff, ddof=1))
    loa_low, loa_high = bias - 1.96 * sd, bias + 1.96 * sd
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    ax.scatter(means, diff, s=42, c=BLUE, edgecolors="white", linewidths=0.5, alpha=0.9)
    ax.axhline(bias, color=RED, ls="--", lw=1.5, label=f"Bias = {bias:+.2f} mm")
    ax.axhline(loa_low, color=GREY, ls=":", lw=1.3, label=f"95% LoA = [{loa_low:+.2f}, {loa_high:+.2f}] mm")
    ax.axhline(loa_high, color=GREY, ls=":", lw=1.3)
    ax.set_xlabel("Mean of estimate and procedural reference (mm)")
    ax.set_ylabel("Estimate - procedural reference (mm)")
    ax.set_title("Procedural benchmark agreement (N=50)", fontweight="bold")
    ax.text(0.02, 0.03, "Procedural data only; not in-vivo clinical agreement", transform=ax.transAxes, fontsize=8)
    ax.legend(loc="upper left", fontsize=8)
    save(fig, "Figure1_Bland_Altman.png")


def figure2_boxplot():
    methods = [r["Method"] for r in DATA["summary_table"]]
    errors = DATA["per_sample"]
    key_map = {
        "Baseline 1: 2D Pixel Equivalent": "2D pixel",
        "Baseline 2: Uncalibrated Depth + Euclidean": "Uncalib. depth",
        "Baseline 3: Standard PCA + Dijkstra Graph": "PCA",
        "Ablation 1: Ours w/o Lie OBB (PCA)": "No Lie",
        "Ablation 2: Ours w/o Heat Method (Euclidean)": "No heat",
        "Ablation 3: Ours w/o Tool Scale (Prior)": "No tool",
        "MetricEndo (Full pipeline)": "Full",
    }
    vals = []
    labels = []
    for method in methods:
        vals.append([abs(r[method] - r["true_major_mm"]) for r in errors])
        labels.append(key_map.get(method, method))
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    bp = ax.boxplot(vals, patch_artist=True, tick_labels=labels, showmeans=True,
                    meanprops={"marker": "o", "markerfacecolor": RED, "markeredgecolor": RED, "markersize": 5})
    colors = [GREY, "#C9C9C9", RED, RED, GOLD, TEAL, BLUE]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
    ax.set_ylabel("Absolute major-diameter error (mm)")
    ax.set_title("Corrected procedural benchmark error distributions (N=50)", fontweight="bold")
    ax.tick_params(axis="x", rotation=25, rotation_mode="anchor")
    ax.text(0.01, 0.98, "Bootstrap intervals are reported in the manuscript table", transform=ax.transAxes, va="top", fontsize=8)
    save(fig, "Figure2_Ablation_Boxplot.png")


def figure3_scatter():
    rows = DATA["per_sample"]
    true = np.array([r["true_major_mm"] for r in rows])
    series = [
        ("Baseline 1 (2D pixel)", "Baseline 1: 2D Pixel Equivalent", RED, "^"),
        ("Baseline 3 (PCA)", "Baseline 3: Standard PCA + Dijkstra Graph", GOLD, "s"),
        ("Full pipeline", "MetricEndo (Full pipeline)", BLUE, "o"),
    ]
    fig, ax = plt.subplots(figsize=(7, 7), dpi=300)
    for label, key, color, marker in series:
        pred = np.array([r[key] for r in rows])
        ax.scatter(true, pred, color=color, marker=marker, alpha=0.75, s=42, label=label, edgecolors="white", linewidths=0.4)
    lo, hi = float(true.min()) - 1, float(true.max()) + 1
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="Identity")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Procedural reference major diameter (mm)")
    ax.set_ylabel("Estimated major diameter (mm)")
    ax.set_title("Corrected procedural size comparison (N=50)", fontweight="bold")
    ax.legend(loc="upper left", fontsize=8)
    ax.text(0.02, 0.03, "No clinical ground truth is implied", transform=ax.transAxes, fontsize=8)
    save(fig, "Figure3_Correlation_Scatter.png")


def main():
    figure1_bland_altman()
    figure2_boxplot()
    figure3_scatter()
    print("Updated legacy root-output figures from journal_benchmark_summary.json")


if __name__ == "__main__":
    main()
