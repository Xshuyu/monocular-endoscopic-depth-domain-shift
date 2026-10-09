"""Leakage-safe evaluation of disagreement-triggered robust PCA fallback."""

import csv
import json
import platform
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.stats import wilcoxon

from lesion_3d_measurement.src.robust_obb_methods import (
    adaptive_filter_obb,
    confidence_aware_obb,
    mad_rejection_fraction,
    mad_filtered_pca_obb,
    pca_obb,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "adaptive_obb"
SEED = 20260925


def bootstrap_ci(x, seed, draws=10000):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(draws, len(x)), replace=True).mean(axis=1)
    return [float(np.quantile(means, .025)), float(np.quantile(means, .975))]


def make_surface(seed, axes, n=1500):
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(n, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    axes = np.asarray(axes, float)
    v /= np.sqrt(np.sum((v / axes[None, :]) ** 2, axis=1, keepdims=True))
    p = Rotation.random(random_state=rng).apply(v) + rng.uniform([-3, -3, 25], [3, 3, 40])
    return p, rng


def main():
    train = range(0, 100)
    validation = range(100, 200)
    test = range(200, 500)
    thresholds = [0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10]
    noise_levels = [0, 1, 2, 5, 10, 20]
    axes_cases = [[12, 7, 3], [18, 10, 5], [30, 16, 8]]

    def make_rows(seeds, include_best=None):
        all_rows = []
        for seed in seeds:
            axes = axes_cases[seed % len(axes_cases)]
            truth = 2 * np.asarray(axes)
            clean, rng = make_surface(SEED + seed, axes)
            for noise_pct in noise_levels:
                n_out = int(len(clean) * noise_pct / 100)
                out = rng.uniform([-45, -45, 3], [45, 45, 75], (n_out, 3)) if n_out else np.empty((0, 3))
                points = np.vstack([clean, out])
                pca_full = pca_obb(points)
                mad = mad_filtered_pca_obb(points, quantiles=None)
                rejection_fraction = mad_rejection_fraction(points)
                methods = {
                    "pca_full": (pca_full, {"mode": "pca_full", "rejection_fraction": rejection_fraction}),
                    "mad_pca": (mad, {"mode": "mad_pca", "rejection_fraction": rejection_fraction}),
                }
                for threshold in thresholds if include_best is None else [include_best]:
                    pred, diag = adaptive_filter_obb(points, threshold)
                    methods[f"adaptive_{threshold}"] = (pred, diag)
                for method, (extent, diag) in methods.items():
                    extent = np.sort(extent)[::-1]
                    err = float(np.mean(np.abs(extent - truth)))
                    all_rows.append({
                        "split": "train" if seed in train else "validation" if seed in validation else "test",
                        "seed": seed, "noise_pct": noise_pct, "true_major_mm": truth[0],
                        "true_minor_mm": truth[1], "true_thickness_mm": truth[2],
                        "method": method, "rejection_fraction": diag["rejection_fraction"],
                        "rejection_fraction": diag["rejection_fraction"],
                        "estimated_major_mm": float(extent[0]), "mean_abs_extent_error_mm": err,
                    })
        return all_rows

    def cache_cases(seeds):
        cache = []
        for seed in seeds:
            axes = axes_cases[seed % len(axes_cases)]
            truth = 2 * np.asarray(axes)
            clean, rng = make_surface(SEED + seed, axes)
            for noise_pct in noise_levels:
                n_out = int(len(clean) * noise_pct / 100.0)
                out = rng.uniform([-45, -45, 3], [45, 45, 75], (n_out, 3)) if n_out else np.empty((0, 3))
                points = np.vstack([clean, out])
                pca = pca_obb(points)
                robust = mad_filtered_pca_obb(points, quantiles=None)
                rejection = mad_rejection_fraction(points)
                cache.append({
                    "seed": seed, "noise_pct": noise_pct, "truth": truth,
                    "pca": np.sort(pca)[::-1], "robust": np.sort(robust)[::-1],
                    "rejection_fraction": rejection,
                })
        return cache

    validation_rows = make_rows(validation)
    threshold_scores = []
    for threshold in thresholds:
        clean_err = [r["mean_abs_extent_error_mm"] for r in validation_rows if r["noise_pct"] == 0 and r["method"] == f"adaptive_{threshold}"]
        noisy_err = [r["mean_abs_extent_error_mm"] for r in validation_rows if r["noise_pct"] in (1, 2, 5, 10, 20) and r["method"] == f"adaptive_{threshold}"]
        pca_clean = [r["mean_abs_extent_error_mm"] for r in validation_rows if r["noise_pct"] == 0 and r["method"] == "pca_full"]
        threshold_scores.append({
            "threshold_deg": threshold,
            "clean_mae_mm": float(np.mean(clean_err)),
            "noisy_mae_mm": float(np.mean(noisy_err)),
            "clean_noninferiority_delta_mm": float(np.mean(clean_err) - np.mean(pca_clean)),
        })
    eligible = [r for r in threshold_scores if r["clean_noninferiority_delta_mm"] <= .25]
    selected = min(eligible, key=lambda r: r["noisy_mae_mm"])["threshold_deg"] if eligible else None
    test_rows = make_rows(test, include_best=selected) if selected is not None else []

    # Select triage thresholds on validation only. Candidate intervals are
    # fixed before test evaluation; accepted outputs must maintain clean
    # non-inferiority while reducing contaminated error.
    val_cache = cache_cases(validation)
    rejection_grid = [r / 1000 for r in (20, 30, 40, 50, 60, 80, 100, 120)]
    triage_candidates = []
    for low in rejection_grid:
        for high in rejection_grid:
            if high <= low:
                continue
            per_noise = {}
            for noise in noise_levels:
                decisions = [r for r in val_cache if r["noise_pct"] == noise]
                estimates = []
                abstain = 0
                for r in decisions:
                    pca_err = float(np.mean(np.abs(r["pca"] - r["truth"])))
                    mad_err = float(np.mean(np.abs(r["robust"] - r["truth"])))
                    rejection = r["rejection_fraction"]
                    if rejection < low:
                        estimates.append(pca_err)
                    elif rejection > high:
                        estimates.append(mad_err)
                    else:
                        abstain += 1
                per_noise[noise] = {"accepted_mae": float(np.mean(estimates)) if estimates else None, "abstention_rate": abstain / max(len(decisions), 1)}
            pca_clean = [float(np.mean(np.abs(r["pca"] - r["truth"]))) for r in val_cache if r["noise_pct"] == 0]
            clean_coverage = 1.0 - per_noise[0]["abstention_rate"]
            clean_delta = (per_noise[0]["accepted_mae"] - float(np.mean(pca_clean))) if per_noise[0]["accepted_mae"] is not None else float("inf")
            noisy_mae = np.mean([v["accepted_mae"] for n, v in per_noise.items() if n >= 5 and v["accepted_mae"] is not None])
            triage_candidates.append({"low": low, "high": high, "clean_coverage": float(clean_coverage), "clean_delta": float(clean_delta), "noisy_accepted_mae": float(noisy_mae), "validation": per_noise})
    triage_eligible = [r for r in triage_candidates if r["clean_delta"] <= .25 and r["clean_coverage"] >= .95]
    triage_selected = min(triage_eligible, key=lambda r: r["noisy_accepted_mae"]) if triage_eligible else None

    test_triage_rows = []
    if triage_selected:
        low, high = triage_selected["low"], triage_selected["high"]
        test_cache = cache_cases(test)
        for r in test_cache:
            if r["rejection_fraction"] < low:
                status, extent = "quantify_pca", r["pca"]
            elif r["rejection_fraction"] > high:
                status, extent = "quantify_mad", r["robust"]
            else:
                status, extent = "abstain_interval", None
            err = None if extent is None else float(np.mean(np.abs(extent - r["truth"])))
            test_triage_rows.append({"seed": r["seed"], "noise_pct": r["noise_pct"], "status": status, "rejection_fraction": r["rejection_fraction"], "mean_abs_extent_error_mm": err})

    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "validation_trials.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(validation_rows[0]))
        writer.writeheader(); writer.writerows(validation_rows)
    with (OUT / "test_trials.csv").open("w", newline="") as f:
        if test_rows:
            writer = csv.DictWriter(f, fieldnames=list(test_rows[0]))
            writer.writeheader(); writer.writerows(test_rows)

    test_summary = []
    if selected is not None:
        for noise in noise_levels:
            subset = [r for r in test_rows if r["noise_pct"] == noise]
            by_method = {m: [r["mean_abs_extent_error_mm"] for r in subset if r["method"] == m] for m in ("pca_full", "mad_pca", f"adaptive_{selected}")}
            row = {"noise_pct": noise, "n_trials": len(by_method["pca_full"])}
            for method, errors in by_method.items():
                row[f"{method}_mae_mm"] = float(np.mean(errors))
                row[f"{method}_bootstrap_95ci"] = bootstrap_ci(errors, SEED + noise + len(method))
            d = np.asarray(by_method[f"adaptive_{selected}"]) - np.asarray(by_method["pca_full"])
            row["adaptive_vs_pca_mean_paired_delta_mm"] = float(d.mean())
            row["adaptive_vs_pca_wilcoxon_p"] = float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
            test_summary.append(row)
    result = {
        "protocol": {"seed": SEED, "train_seeds": [0, 99], "validation_seeds": [100, 199], "test_seeds": [200, 499], "axes_cases_mm": axes_cases, "noise_levels_pct": noise_levels, "rejection_fraction_threshold_grid": thresholds, "clean_noninferiority_margin_mm": .25, "minimum_clean_coverage": .95},
        "validation_threshold_scores": threshold_scores,
        "selected_threshold_deg": selected,
        "triage_validation_candidates": triage_candidates,
        "selected_triage_thresholds": triage_selected,
        "test_summary": test_summary,
        "triage_test_rows": test_triage_rows,
        "environment": {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__},
    }
    (OUT / "adaptive_obb_summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
