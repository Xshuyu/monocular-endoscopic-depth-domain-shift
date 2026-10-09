"""Leakage-safe benchmark for adaptive calibration and boundary-aware OBB."""

import csv
import json
import platform
import sys
from pathlib import Path

import numpy as np

from lesion_3d_measurement.src.lie_algebra_obb import compute_robust_obb_lie


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "controlled_improvement"


def make_ellipsoid(seed, major, minor, height, n=600):
    rng = np.random.default_rng(seed)
    phi = rng.uniform(0, np.pi, n)
    theta = rng.uniform(0, 2 * np.pi, n)
    rad = rng.uniform(0, 1, n) ** (1 / 3)
    return np.column_stack([
        major / 2 * rad * np.sin(phi) * np.cos(theta),
        minor / 2 * rad * np.sin(phi) * np.sin(theta),
        height / 2 * rad * np.cos(phi) + 30,
    ])


def predict(points, correction):
    return compute_robust_obb_lie(points, boundary_correction=correction).major_diameter_mm


def main():
    # Train/validation/test are disjoint by seed and dimensions.
    train = list(range(0, 20))
    validation = list(range(20, 30))
    test = list(range(30, 50))
    dimensions = [(8, 5, 3), (12, 8, 4), (20, 12, 5), (28, 18, 7), (35, 22, 9)]

    # Select correction only on validation data.  The grid is intentionally
    # small and fixed before test evaluation.
    grid = np.linspace(0.0, 1.0, 11)
    validation_scores = []
    for correction in grid:
        errors = []
        for i, seed in enumerate(validation):
            major, minor, height = dimensions[i % len(dimensions)]
            points = make_ellipsoid(seed, major, minor, height)
            errors.append(abs(predict(points, correction) - major))
        validation_scores.append((float(np.mean(errors)), float(correction)))
    best_correction = min(validation_scores)[1]

    rows = []
    for split, seeds in (("train", train), ("validation", validation), ("test", test)):
        for seed in seeds:
            major, minor, height = dimensions[seed % len(dimensions)]
            clean = make_ellipsoid(seed, major, minor, height)
            rng = np.random.default_rng(seed + 1000)
            n_out = int(len(clean) * 0.10)
            noisy = np.vstack([clean, rng.uniform([-40, -40, 10], [40, 40, 50], (n_out, 3))])
            for noise_name, points in (("clean", clean), ("10pct_outlier", noisy)):
                baseline = predict(points, 0.0)
                improved = predict(points, best_correction)
                rows.append({
                    "split": split,
                    "seed": seed,
                    "noise": noise_name,
                    "true_major_mm": major,
                    "baseline_major_mm": baseline,
                    "improved_major_mm": improved,
                    "baseline_abs_error_mm": abs(baseline - major),
                    "improved_abs_error_mm": abs(improved - major),
                    "selected_correction": best_correction,
                })

    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "trials.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "best_validation_correction": best_correction,
        "validation_grid": [{"correction": c, "mae_mm": e} for e, c in validation_scores],
        "test": {
            noise: {
                "baseline_mae_mm": float(np.mean([r["baseline_abs_error_mm"] for r in rows if r["split"] == "test" and r["noise"] == noise])),
                "improved_mae_mm": float(np.mean([r["improved_abs_error_mm"] for r in rows if r["split"] == "test" and r["noise"] == noise])),
            }
            for noise in ("clean", "10pct_outlier")
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
