"""Create reviewer-response figures strictly from frozen radial-confirmatory CSV/JSON.

No fitting, prediction, parameter selection, or test-label alignment is performed.
The plots are descriptive because the new test contains four videos from three
anatomical segment families, not independent patients.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "outputs/radial_confirmatory"
OUT = DATA / "reviewer_response_supplement/figures"
OUT.mkdir(parents=True, exist_ok=True)

COLORS = {
    "affine_ls": "#1f4e79",
    "affine_ridge": "#75aadb",
    "radial_ls": "#2b8c5a",
    "radial_ridge": "#d55e00",
    "historical_affine": "#777777",
    "historical_radial": "#aa77aa",
}
LABELS = {
    "affine_ls": "Global affine (LS)",
    "affine_ridge": "Global affine (ridge)",
    "radial_ls": "Radial affine (LS)",
    "radial_ridge": "Radial affine (ridge)",
    "historical_affine": "Historical fixed affine*",
    "historical_radial": "Historical fixed radial*",
}


def read_csv(name: str) -> list[dict]:
    with (DATA / name).open(newline="") as f:
        return list(csv.DictReader(f))


def save(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUT / f"{stem}.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def fig1_per_video() -> None:
    rows = read_csv("test_per_video.csv")
    scenes = sorted({r["scene"] for r in rows})
    methods = ["affine_ls", "radial_ls", "radial_ridge", "historical_affine", "historical_radial"]
    lookup = {(r["scene"], r["method"]): float(r["mae_mm"]) for r in rows}

    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    x = np.arange(len(scenes))
    width = 0.15
    offsets = np.linspace(-2, 2, len(methods)) * width
    for method, dx in zip(methods, offsets):
        vals = [lookup[(s, method)] for s in scenes]
        ax.bar(x + dx, vals, width=width * 0.95, label=LABELS[method],
               color=COLORS[method], edgecolor="white", linewidth=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("_", "\n") for s in scenes], fontsize=9)
    ax.set_ylabel("Axial-depth MAE (mm)")
    ax.set_title("New-sequence test: per-video MAE (4 videos; descriptive)", weight="bold")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(ncol=2, frameon=False, fontsize=8.5)
    ax.text(0.01, -0.23,
            "*Historical fixed coefficients have unresolved fitting provenance; shown only as exploratory controls.",
            transform=ax.transAxes, fontsize=8.5, color="#555555")
    fig.tight_layout()
    save(fig, "audit_fig1_test_per_video")


def fig2_dev_vs_test() -> None:
    dev = json.loads((DATA / "development_summary.json").read_text())
    test = json.loads((DATA / "test_summary.json").read_text())
    d = {x["method"]: x for x in dev}
    t = {x["method"]: x for x in test}
    methods = ["affine_ls", "affine_ridge", "radial_ls", "radial_ridge"]
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    x = np.arange(len(methods))
    w = 0.36
    ax.bar(x - w/2, [d[m]["mae_mm"] for m in methods], w,
           label="Development (6 previously explored scenes)", color="#9bbad3")
    ax.bar(x + w/2, [t[m]["mae_mm"] for m in methods], w,
           label="New-sequence test (4 videos)", color="#315d7d")
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[m] for m in methods], rotation=12, ha="right")
    ax.set_ylabel("Equal-scene/video mean axial-depth MAE (mm)")
    ax.set_title("Development-to-test transfer and regularization effects", weight="bold")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False, fontsize=8.5)
    ax.text(0.01, -0.22,
            "Development scenes were used in prior exploration. New test is phantom video data, not patients.",
            transform=ax.transAxes, fontsize=8.5, color="#555555")
    fig.tight_layout()
    save(fig, "audit_fig2_dev_test_transfer")


def fig3_joint_strata() -> None:
    rows = read_csv("test_strata_summary.csv")
    methods = ["affine_ls", "radial_ridge"]
    rlabels = ["Center", "Middle", "Edge"]
    dlabels = ["0–10", "10–30", "30–100"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6), sharey=True)
    for ax, method in zip(axes, methods):
        mat = np.full((3, 3), np.nan)
        support = np.zeros((3, 3), dtype=int)
        for row in rows:
            if row["method"] != method:
                continue
            rb, db = int(row["radius_bin"]), int(row["depth_bin"])
            if db >= 0:
                mat[rb, db] = float(row["mae_mm"])
                support[rb, db] = int(row["videos"])
        im = ax.imshow(mat, cmap="magma", vmin=0, vmax=20, aspect="auto")
        ax.set_xticks(range(3), dlabels)
        ax.set_yticks(range(3), rlabels)
        ax.set_xlabel("Reference axial depth (mm)")
        ax.set_title(LABELS[method], weight="bold")
        for i in range(3):
            for j in range(3):
                if np.isfinite(mat[i, j]):
                    ax.text(j, i, f"{mat[i,j]:.1f}\n{support[i,j]} videos",
                            ha="center", va="center", color="white" if mat[i,j] > 10 else "black", fontsize=8)
                else:
                    ax.text(j, i, "No support", ha="center", va="center", color="#555555", fontsize=7.5)
    axes[0].set_ylabel("Normalized image-radius region")
    fig.colorbar(im, ax=axes, shrink=0.85, label="Equal-video MAE (mm)")
    fig.suptitle("Joint radius × reference-depth error strata on the new test", weight="bold")
    fig.text(0.5, -0.01,
             "Cell labels include the number of test videos contributing. Empty cells are not zero-error cells. Descriptive only.",
             ha="center", fontsize=8.5, color="#555555")
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    save(fig, "audit_fig3_radius_depth_support")


def fig4_lambdas() -> None:
    outer = json.loads((DATA / "outer_folds.json").read_text())
    scenes = [x["held"] for x in outer]
    families = ["affine_ridge", "radial_ridge"]
    vals = {fam: [] for fam in families}
    for fold in outer:
        for fam in families:
            vals[fam].append(float(fold["choices"][fam]["lambda"]))
    fig, ax = plt.subplots(figsize=(9, 4.7))
    x = np.arange(len(scenes))
    ax.plot(x, vals["affine_ridge"], "o-", lw=2, color=COLORS["affine_ridge"], label="Affine ridge")
    ax.plot(x, vals["radial_ridge"], "s-", lw=2, color=COLORS["radial_ridge"], label="Radial ridge")
    ax.set_yscale("log")
    ax.set_xticks(x, [s.replace("c1_", "") for s in scenes], rotation=15, ha="right")
    ax.set_ylabel("Inner-selected $\\lambda$ (log scale)")
    ax.set_title("Scene-level regularization selection is heterogeneous", weight="bold")
    ax.grid(axis="y", alpha=0.25, which="both")
    ax.legend(frameon=False)
    ax.text(0.01, -0.32,
            "Ridge choices frequently lie at search-grid boundaries; the four-video test is too small for population claims.",
            transform=ax.transAxes, fontsize=8.5, color="#555555")
    fig.tight_layout()
    save(fig, "audit_fig4_lambda_selection")


if __name__ == "__main__":
    fig1_per_video()
    fig2_dev_vs_test()
    fig3_joint_strata()
    fig4_lambdas()
    print(f"Wrote reviewer-audit figures to {OUT}")
