"""
Master Script for Executing the Complete Medical Journal Benchmark Pipeline.
Runs cohort evaluation, computes p-values, generates Bland-Altman statistics,
creates 300 DPI publication figures, and outputs LaTeX tables.
"""

import os
import json
import time
import numpy as np
import scipy.stats as stats
import platform
import sys

from lesion_3d_measurement.src.ground_truth_benchmark import generate_benchmark_cohort
from lesion_3d_measurement.src.evaluation_suite import evaluate_baselines_and_ablations
from lesion_3d_measurement.src.statistical_analysis import generate_publication_figures, compute_icc_and_bland_altman


def main():
    print("==================================================================")
    print("RUNNING COMPREHENSIVE MEDICAL TOP-JOURNAL VALIDATION BENCHMARK")
    print("==================================================================")
    t0 = time.time()

    # 1. Generate 50 standardized cohort samples with exact physical ground truth (mm)
    print("[1/4] Generating Ground-Truth Calibrated Lesion Cohort (N=50)...")
    cohort = generate_benchmark_cohort(num_samples=50)
    print(f"      Successfully generated {len(cohort)} multi-scale lesion samples.")

    # 2. Run multi-baseline comparison & ablation suite
    print("[2/4] Evaluating 3 Baselines + 3 Ablations + Ours (Full MetricEndo)...")
    eval_res = evaluate_baselines_and_ablations(cohort)

    # 3. Statistical hypothesis testing (Paired t-tests between Baselines and Ours)
    print("[3/4] Performing Statistical Hypothesis Testing (p-values & ICC)...")
    ours_errs = eval_res["errors_major"]["MetricEndo (Full pipeline)"]
    p_values = {}
    wilcoxon_p_values = {}
    for m_name in eval_res["errors_major"]:
        if m_name != "MetricEndo (Full pipeline)":
            other_errs = eval_res["errors_major"][m_name]
            # Paired t-test
            if np.allclose(other_errs, ours_errs):
                p_values[m_name] = 1.0
            else:
                t_stat, p_val = stats.ttest_rel(other_errs, ours_errs)
                p_values[m_name] = float(p_val)
            try:
                if np.allclose(other_errs, ours_errs):
                    wilcoxon_p_values[m_name] = 1.0
                else:
                    wilcoxon_p_values[m_name] = float(
                        stats.wilcoxon(other_errs, ours_errs, zero_method="wilcox").pvalue
                    )
            except ValueError:
                wilcoxon_p_values[m_name] = 1.0

    # 4. Generate Publication Figures (Bland-Altman, Boxplots, Scatter)
    print("[4/4] Rendering 300 DPI Medical Journal Figures...")
    fig_paths = generate_publication_figures(eval_res)

    t_elapsed = time.time() - t0

    # 5. Format and Print LaTeX Table
    print("\n" + "="*85)
    print("QUANTITATIVE BENCHMARK & ABLATION RESULTS (FORMATTED FOR TOP JOURNALS)")
    print("="*85)
    print(f"{'Method / Architecture':<42} | {'MAE (mm)':<12} | {'RMSE (mm)':<10} | {'MRE (%)':<10} | {'Risk Acc (%)':<12} | {'p-value'}")
    print("-" * 105)
    for row in eval_res["summary"]:
        m = row["Method"]
        p_str = f"{p_values.get(m, 0.0):.1e}" if m in p_values else "Ref"
        print(f"{m:<42} | {row['MAE_mm']:>4.2f} ± {row['MAE_CI95']:<4.2f} | {row['RMSE_mm']:>8.2f} | {row['MRE_pct']:>8.2f}% | {row['Risk_Agreement_pct']:>10.1f}% | {p_str:>8}")

    print("\n" + "="*85)
    print("CLINICAL BLAND-ALTMAN & CORRELATION METRICS (OURS)")
    print("="*85)
    stats_dict = fig_paths["statistics"]
    print(f"  • Mean Measurement Bias:              {stats_dict['mean_bias_mm']:+.2f} mm")
    print(f"  • Standard Deviation of Bias (SD):    {stats_dict['std_diff_mm']:.2f} mm")
    print(f"  • 95% Limits of Agreement (LoA):      [{stats_dict['lower_loa_mm']:+.2f} mm, {stats_dict['upper_loa_mm']:+.2f} mm]")
    print(f"  • Pearson Correlation (R²):            {stats_dict['pearson_r']**2:.4f} (R = {stats_dict['pearson_r']:.4f})")
    print(f"  • Lin's Concordance Correlation (CCC): {stats_dict['concordance_ccc']:.4f}")

    print("\n" + "="*85)
    print("GENERATED PUBLICATION FIGURES")
    print("="*85)
    print(f"  [Fig 1] Bland-Altman Plot:    {fig_paths['bland_altman']}")
    print(f"  [Fig 2] Multi-Method Boxplot: {fig_paths['ablation_boxplot']}")
    print(f"  [Fig 3] Correlation Scatter:  {fig_paths['correlation_scatter']}")

    # Save complete JSON
    export_payload = {
        "summary_table": eval_res["summary"],
        "p_values": p_values,
        "wilcoxon_p_values": wilcoxon_p_values,
        "clinical_statistics": stats_dict,
        "per_sample": eval_res["per_sample"],
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": __import__("scipy").__version__,
        },
        "figures": fig_paths,
        "execution_time_sec": round(t_elapsed, 2)
    }
    json_path = "/ssd-1/temp/kimicode/multmodel/lesion_3d_measurement/outputs/journal_benchmark_summary.json"
    with open(json_path, "w") as f:
        json.dump(export_payload, f, indent=2)

    print(f"\nAll benchmark results successfully exported to: {json_path}")


if __name__ == "__main__":
    main()
