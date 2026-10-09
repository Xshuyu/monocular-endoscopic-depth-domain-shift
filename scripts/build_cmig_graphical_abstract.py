from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "manuscript_cmig" / "submission"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 14,
    "pdf.fonttype": 42,
})

fig, ax = plt.subplots(figsize=(12, 7))
fig.subplots_adjust(0, 0, 1, 1)
ax.set_xlim(0, 1200)
ax.set_ylim(700, 0)
ax.axis("off")

navy = "#245b85"
green = "#28745a"
amber = "#a86d16"
ink = "#243442"
muted = "#5b6872"

ax.text(600, 55, "Metric depth calibration under domain shift", ha="center",
        va="center", fontsize=28, fontweight="bold", color=ink)
ax.text(600, 96, "Matched calibration • supervised adaptation • bounded transfer evidence",
        ha="center", va="center", fontsize=15, color=muted)

def box(x, y, w, h, title, lines, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                               boxstyle="round,pad=0.012,rounding_size=18",
                               facecolor="white", edgecolor=color, linewidth=2.2))
    ax.text(x + w / 2, y + 38, title, ha="center", va="center",
            fontsize=19, fontweight="bold", color=color)
    for i, line in enumerate(lines):
        ax.text(x + w / 2, y + 82 + i * 35, line, ha="center", va="center",
                fontsize=15, color=ink)

box(45, 160, 325, 205, "1  Calibrate", [
    "Frozen monocular backbone",
    "Affine / radial × LS / ridge",
    "Matched 16,384-pixel budget",
], navy)
box(437, 160, 325, 205, "2  Adapt", [
    "20 measured target frames",
    "Object-disjoint training",
    "Three random seeds",
], green)
box(830, 160, 325, 205, "3  Evaluate", [
    "Held-out cohorts",
    "Native-grid depth scoring",
    "No test-reference fitting",
], navy)

for x0, x1 in [(370, 437), (762, 830)]:
    ax.add_patch(FancyArrowPatch((x0, 262), (x1, 262), arrowstyle="-|>",
                                mutation_scale=22, linewidth=2.2, color=muted))

ax.text(600, 420, "Adaptation changed error in a cohort-dependent way", ha="center",
        va="center", fontsize=20, fontweight="bold", color=ink)

results = [
    ("SCARED", "36.72 → 13.67 mm", "held-out improvement", green),
    ("SERV-CT", "55.37 → 22.21 mm", "held-out improvement", green),
    ("C3VD", "8.02 → 9.09 mm", "source-domain increase", amber),
    ("DRENDS", "93.78 / 57.22 mm", "phantom / ex vivo; residual error", amber),
]
xs = [45, 337, 629, 921]
for x, (name, value, note, color) in zip(xs, results):
    ax.add_patch(FancyBboxPatch((x, 465), 235, 125,
                               boxstyle="round,pad=0.01,rounding_size=14",
                               facecolor="#f7f9fa", edgecolor="#d6dde2", linewidth=1.2))
    ax.text(x + 117.5, 493, name, ha="center", va="center",
            fontsize=16, fontweight="bold", color=ink)
    ax.text(x + 117.5, 531, value, ha="center", va="center",
            fontsize=15, fontweight="bold", color=color)
    ax.text(x + 117.5, 565, note, ha="center", va="center",
            fontsize=11.5, color=muted)

ax.text(600, 638,
        "Conditional gains with source-domain trade-offs; no device-invariant depth or clinical sizing claim",
        ha="center", va="center", fontsize=15, fontweight="bold", color=ink)
ax.text(600, 673, "Values are cohort MAE (mm); DRENDS groups are not verified independent donors.",
        ha="center", va="center", fontsize=11.5, color=muted)

OUT.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT / "graphical_abstract.pdf", facecolor="white")
fig.savefig(OUT / "graphical_abstract.png", dpi=300, facecolor="white")
plt.close(fig)
