"""Generalization benchmark for the fixed boundary-aware OBB correction."""

import csv
import json
import platform
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from lesion_3d_measurement.src.lie_algebra_obb import compute_robust_obb_lie


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "generalization"
FIXED_CORRECTION = 1.0


def _sample_shape(rng, shape, major, minor, height, n):
    if shape == "ellipsoid":
        phi = rng.uniform(0, np.pi, n)
        theta = rng.uniform(0, 2 * np.pi, n)
        rad = rng.uniform(0, 1, n) ** (1 / 3)
        return np.column_stack([
            major / 2 * rad * np.sin(phi) * np.cos(theta),
            minor / 2 * rad * np.sin(phi) * np.sin(theta),
            height / 2 * rad * np.cos(phi),
        ])
    if shape == "dome":
        phi = rng.uniform(0, np.pi / 2, n)
        theta = rng.uniform(0, 2 * np.pi, n)
        radius = major / 2
        return np.column_stack([
            radius * np.sin(phi) * np.cos(theta),
            radius * minor / major * np.sin(phi) * np.sin(theta),
            height * np.cos(phi),
        ])
    if shape == "plaque":
        x = rng.uniform(-major / 2, major / 2, n)
        y = rng.uniform(-minor / 2, minor / 2, n)
        z = height * np.exp(-2.0 * ((x / major) ** 2 + (y / minor) ** 2))
        return np.column_stack([x, y, z - height / 2])
    if shape == "saddle":
        x = rng.uniform(-major / 2, major / 2, n)
        y = rng.uniform(-minor / 2, minor / 2, n)
        z = height * ((x / major) ** 2 - (y / minor) ** 2)
        return np.column_stack([x, y, z])
    raise ValueError(shape)


def make_case(seed, shape, density, noise_pct, rotation_deg):
    rng = np.random.default_rng(seed)
    major = float(rng.uniform(8.0, 35.0))
    minor = float(major * rng.uniform(0.55, 0.9))
    height = float(rng.uniform(2.0, min(10.0, major * 0.45)))
    n = {"low": 250, "medium": 600, "high": 1200}[density]
    points = _sample_shape(rng, shape, major, minor, height, n)
    rotation = Rotation.from_euler("xyz", rng.uniform(-rotation_deg, rotation_deg, 3), degrees=True)
    points = rotation.apply(points)
    points[:, 2] += 30.0
    if noise_pct:
        count = int(n * noise_pct / 100.0)
        outliers = rng.uniform([-40, -40, 10], [40, 40, 50], (count, 3))
        points = np.vstack([points, outliers])
    return points, major


def main():
    rows = []
    case_id = 0
    for shape in ("ellipsoid", "dome", "plaque", "saddle"):
        for density in ("low", "medium", "high"):
            for noise_pct in (0, 5, 10, 20):
                for rotation_deg in (0, 30, 60):
                    for repeat in range(5):
                        seed = 10000 + case_id
                        points, truth = make_case(seed, shape, density, noise_pct, rotation_deg)
                        baseline = compute_robust_obb_lie(points, boundary_correction=0.0)
                        improved = compute_robust_obb_lie(points, boundary_correction=FIXED_CORRECTION)
                        adaptive = compute_robust_obb_lie(
                            points,
                            boundary_correction=FIXED_CORRECTION,
                            adaptive_boundary=True,
                        )
                        rows.append({
                            "case_id": case_id,
                            "seed": seed,
                            "shape": shape,
                            "density": density,
                            "noise_pct": noise_pct,
                            "rotation_deg": rotation_deg,
                            "true_major_mm": truth,
                            "baseline_major_mm": baseline.major_diameter_mm,
                            "improved_major_mm": improved.major_diameter_mm,
                            "adaptive_major_mm": adaptive.major_diameter_mm,
                            "baseline_abs_error_mm": abs(baseline.major_diameter_mm - truth),
                            "improved_abs_error_mm": abs(improved.major_diameter_mm - truth),
                            "adaptive_abs_error_mm": abs(adaptive.major_diameter_mm - truth),
                        })
                        case_id += 1

    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "trials.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    groups = []
    for shape in ("ellipsoid", "dome", "plaque", "saddle"):
        for noise in (0, 5, 10, 20):
            subset = [r for r in rows if r["shape"] == shape and r["noise_pct"] == noise]
            groups.append({
                "shape": shape,
                "noise_pct": noise,
                "n": len(subset),
                "baseline_mae_mm": float(np.mean([r["baseline_abs_error_mm"] for r in subset])),
                "improved_mae_mm": float(np.mean([r["improved_abs_error_mm"] for r in subset])),
                "adaptive_mae_mm": float(np.mean([r["adaptive_abs_error_mm"] for r in subset])),
                "improved_better_fraction": float(np.mean([r["improved_abs_error_mm"] < r["baseline_abs_error_mm"] for r in subset])),
                "adaptive_better_fraction": float(np.mean([r["adaptive_abs_error_mm"] < r["baseline_abs_error_mm"] for r in subset])),
            })
    summary = {
        "fixed_boundary_correction": FIXED_CORRECTION,
        "n_cases": len(rows),
        "groups": groups,
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
