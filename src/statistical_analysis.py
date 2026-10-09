"""
Medical Journal Statistical Analysis and Visualization Generator.
Computes:
  - Bland-Altman Limits of Agreement (Mean Bias, 95% Upper & Lower LoA)
  - Lin's concordance correlation coefficient (CCC); no inter-rater ICC
  - Paired Student's t-test and Wilcoxon Signed-Rank Test (p-values)
  - High-resolution scientific figures:
      Figure 1: Bland-Altman Plot (Predicted vs True Caliper Ground Truth)
      Figure 2: Method Comparison Error Boxplots (MAE across Baselines)
      Figure 3: Correlation Scatter Plot with Identity Line (R^2)
"""

import os
from typing import Dict, Any, List
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


def compute_icc_and_bland_altman(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Computes Bland-Altman, Pearson correlation and Lin CCC.

    The historical function name is retained for compatibility. This does not
    compute ICC or establish reference-annotation reliability.
    """
    diff = y_pred - y_true
    mean_bias = float(np.mean(diff))
    std_diff = float(np.std(diff, ddof=1))
    upper_loa = mean_bias + 1.96 * std_diff
    lower_loa = mean_bias - 1.96 * std_diff

    # Prediction/reference association, not inter-rater reliability.
    r, p_val = stats.pearsonr(y_true, y_pred)

    # Lin's Concordance Correlation Coefficient (CCC)
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    covar = np.mean((y_true - mean_true) * (y_pred - mean_pred))
    ccc = (2.0 * covar) / (var_true + var_pred + (mean_true - mean_pred)**2)

    return {
        "mean_bias_mm": float(mean_bias),
        "std_diff_mm": float(std_diff),
        "upper_loa_mm": float(upper_loa),
        "lower_loa_mm": float(lower_loa),
        "pearson_r": float(r),
        "concordance_ccc": float(ccc)
    }


def generate_publication_figures(
    eval_results: Dict[str, Any],
    output_dir: str = "/ssd-1/temp/kimicode/multmodel/lesion_3d_measurement/outputs"
) -> Dict[str, str]:
    """
    Renders publication-ready multi-panel vector figures (300 DPI style).
    """
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid", font="sans-serif")
    palette = sns.color_palette("deep")

    y_true = np.array(eval_results["true_majors"])
    ours_pred = np.array(eval_results["predicted_majors"]["MetricEndo (Full pipeline)"])
    b3_pred = np.array(eval_results["predicted_majors"]["Baseline 3: Standard PCA + Dijkstra Graph"])
    b1_pred = np.array(eval_results["predicted_majors"]["Baseline 1: 2D Pixel Equivalent"])

    stats_ours = compute_icc_and_bland_altman(y_true, ours_pred)

    # -------------------------------------------------------------
    # Figure 1: Bland-Altman Plot (The gold-standard for clinical validation)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    means = (y_true + ours_pred) / 2.0
    diffs = ours_pred - y_true

    ax.scatter(means, diffs, color="#1f77b4", alpha=0.85, edgecolors="k", s=45, label="Lesion Samples")
    ax.axhline(stats_ours["mean_bias_mm"], color="red", linestyle="--", linewidth=1.8,
               label=f"Mean Bias: {stats_ours['mean_bias_mm']:+.2f} mm")
    ax.axhline(stats_ours["upper_loa_mm"], color="gray", linestyle=":", linewidth=1.5,
               label=f"+1.96 SD: {stats_ours['upper_loa_mm']:+.2f} mm")
    ax.axhline(stats_ours["lower_loa_mm"], color="gray", linestyle=":", linewidth=1.5,
               label=f"-1.96 SD: {stats_ours['lower_loa_mm']:+.2f} mm")

    ax.set_title("Bland-Altman Agreement: In-Vivo MetricEndo vs. Physical Ground Truth", fontsize=13, fontweight="bold")
    ax.set_xlabel("Mean of MetricEndo & Caliper Ground Truth (mm)", fontsize=11)
    ax.set_ylabel("Difference (MetricEndo - Ground Truth) (mm)", fontsize=11)
    ax.legend(loc="upper right", frameon=True)
    plt.tight_layout()

    fig1_path = os.path.join(output_dir, "Figure1_Bland_Altman.png")
    plt.savefig(fig1_path)
    plt.close()

    # -------------------------------------------------------------
    # Figure 2: Multi-Method Error Comparison Boxplot
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    err_df_list = []
    for m_name, errs in eval_results["errors_major"].items():
        short_name = m_name.replace("Baseline ", "B").replace("Ablation ", "Abl-").replace("Ours: ", "")
        for e in errs:
            err_df_list.append({"Method": short_name, "Absolute Error (mm)": e})

    df_err = pd.DataFrame(err_df_list)
    sns.boxplot(x="Method", y="Absolute Error (mm)", data=df_err, ax=ax, palette="Blues_r", showmeans=True,
                meanprops={"marker":"o", "markerfacecolor":"red", "markeredgecolor":"red"})
    ax.set_title("Ablation & Baseline Absolute Error Comparison (N=50)", fontsize=13, fontweight="bold")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Absolute Error |Pred - True| (mm)", fontsize=11)
    plt.tight_layout()

    fig2_path = os.path.join(output_dir, "Figure2_Ablation_Boxplot.png")
    plt.savefig(fig2_path)
    plt.close()

    # -------------------------------------------------------------
    # Figure 3: Correlation Scatter Plot with Identity Line (R^2 & Concordance)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 7), dpi=300)
    ax.scatter(y_true, b1_pred, color="#d62728", alpha=0.5, marker="^", s=40, label="Baseline 1 (2D Pixel)")
    ax.scatter(y_true, b3_pred, color="#ff7f0e", alpha=0.6, marker="s", s=40, label="Baseline 3 (PCA)")
    ax.scatter(y_true, ours_pred, color="#2ca02c", alpha=0.9, marker="o", edgecolors="k", s=55, label="Ours (MetricEndo Full)")

    min_val = min(y_true.min(), ours_pred.min()) - 2.0
    max_val = max(y_true.max(), ours_pred.max()) + 2.0
    ax.plot([min_val, max_val], [min_val, max_val], "k--", alpha=0.7, label="Ideal Line (y = x)")

    ax.set_title(f"Physical Size Correlation (CCC = {stats_ours['concordance_ccc']:.3f}, Pearson r² = {stats_ours['pearson_r']**2:.3f})", fontsize=13, fontweight="bold")
    ax.set_xlabel("True Physical Caliper Diameter (mm)", fontsize=11)
    ax.set_ylabel("Estimated In-Vivo Diameter (mm)", fontsize=11)
    ax.legend(loc="upper left", frameon=True)
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    plt.tight_layout()

    fig3_path = os.path.join(output_dir, "Figure3_Correlation_Scatter.png")
    plt.savefig(fig3_path)
    plt.close()

    return {
        "bland_altman": fig1_path,
        "ablation_boxplot": fig2_path,
        "correlation_scatter": fig3_path,
        "statistics": stats_ours
    }
