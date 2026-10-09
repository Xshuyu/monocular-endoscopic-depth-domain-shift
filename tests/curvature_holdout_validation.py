"""Train/validation/test audit for local Monge curvature estimation."""

import csv
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from lesion_3d_measurement.src.differential_geometry import compute_manifold_curvature

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "curvature_holdout"
SEED = 872026
RADIUS = 10.0
K_GRID = (12, 20, 30, 40, 60)
NOISE_GRID = (0.0, 0.001, 0.005, 0.01)


def sample_cap(seed, n):
    rng = np.random.default_rng(seed)
    cos_theta = rng.uniform(np.cos(np.pi / 3), 1.0, n)
    theta = np.arccos(cos_theta)
    azimuth = rng.uniform(0, 2 * np.pi, n)
    xyz = np.column_stack([
        RADIUS * np.sin(theta) * np.cos(azimuth),
        RADIUS * np.sin(theta) * np.sin(azimuth),
        RADIUS * (1 - np.cos(theta)),
    ])
    normal = np.column_stack([
        np.sin(theta) * np.cos(azimuth),
        np.sin(theta) * np.sin(azimuth),
        -np.cos(theta),
    ])
    return xyz, normal


def evaluate(seed, n, noise, k):
    xyz, normal = sample_cap(seed, n)
    if noise:
        xyz = xyz + normal * np.random.default_rng(seed + 1_000_000).normal(0, noise, (n, 1))
    profile = compute_manifold_curvature(
        xyz, RADIUS * (1 - np.cos(np.pi / 3)), k_neighbors=k,
        normal_orientation="outward", random_seed=seed,
    )
    return {
        "seed": seed, "n_points": n, "noise_sd_mm": noise, "k": k,
        "H_est": profile.mean_mean_curvature_H,
        "K_est": profile.mean_gaussian_curvature_K,
        "abs_H_error": abs(abs(profile.mean_mean_curvature_H) - 1 / RADIUS),
        "abs_K_error": abs(profile.mean_gaussian_curvature_K - 1 / RADIUS**2),
        "shape_index": profile.mean_shape_index,
    }


def main():
    # Disjoint seeds and densities. Tune k only on validation; report holdout.
    train_seeds, val_seeds, test_seeds = range(0, 20), range(100, 120), range(200, 240)
    densities = {"train": 1200, "validation": 1800, "test": 2400}
    all_rows = []
    for split, seeds in (("train", train_seeds), ("validation", val_seeds), ("test", test_seeds)):
        for seed in seeds:
            for noise in NOISE_GRID:
                for k in K_GRID:
                    row = evaluate(SEED + seed, densities[split], noise, k)
                    row["split"] = split
                    all_rows.append(row)
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "trials.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0]))
        w.writeheader(); w.writerows(all_rows)
    candidates = []
    for k in K_GRID:
        h = [r["abs_H_error"] for r in all_rows if r["split"] == "validation" and r["noise_sd_mm"] == 0 and r["k"] == k]
        g = [r["abs_K_error"] for r in all_rows if r["split"] == "validation" and r["noise_sd_mm"] == 0 and r["k"] == k]
        candidates.append({"k": k, "validation_mean_H_error": float(np.mean(h)), "validation_mean_K_error": float(np.mean(g))})
    selected = min(candidates, key=lambda r: r["validation_mean_H_error"] + RADIUS * r["validation_mean_K_error"])["k"]
    holdout = []
    for noise in NOISE_GRID:
        rows = [r for r in all_rows if r["split"] == "test" and r["noise_sd_mm"] == noise and r["k"] == selected]
        holdout.append({
            "noise_sd_mm": noise, "k_selected_on_validation": selected,
            "n_trials": len(rows),
            "mean_abs_H_error": float(np.mean([r["abs_H_error"] for r in rows])),
            "mean_abs_K_error": float(np.mean([r["abs_K_error"] for r in rows])),
            "p95_abs_H_error": float(np.quantile([r["abs_H_error"] for r in rows], .95)),
            "p95_abs_K_error": float(np.quantile([r["abs_K_error"] for r in rows], .95)),
        })
    summary = {
        "truth": {"radius_mm": RADIUS, "abs_H_mm_inv": 1 / RADIUS, "K_mm_inv2": 1 / RADIUS**2},
        "split": {"train_seeds": [0, 19], "validation_seeds": [100, 119], "test_seeds": [200, 239], "points_per_split": densities},
        "candidate_scores": candidates,
        "selected_k": selected,
        "holdout_noise_results": holdout,
        "rotation_invariance_test": rotation_invariance_test(),
        "analytic_plane_saddle_test": analytic_plane_saddle_test(),
        "claim_limit": "analytic visible spherical cap only; does not validate Paris classification or patient surfaces",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def rotation_invariance_test():
    rows = []
    base, _ = sample_cap_points(seed=9876, n=2400, radius=RADIUS)
    sphere_center = np.array([0.0, 0.0, RADIUS])
    for seed in range(20):
        rot = Rotation.random(random_state=np.random.default_rng(6000 + seed))
        points = rot.apply(base)
        rotated_center = rot.apply(sphere_center)
        prof = compute_manifold_curvature(
            points, RADIUS * (1 - np.cos(np.pi / 3)), k_neighbors=30,
            normal_orientation="outward", random_seed=SEED + seed,
            normal_reference_point=rotated_center,
        )
        rows.append({"seed": seed, "H": prof.mean_mean_curvature_H, "K": prof.mean_gaussian_curvature_K})
    return {
        "n_rotations": len(rows),
        "H_range": [float(min(r["H"] for r in rows)), float(max(r["H"] for r in rows))],
        "K_range": [float(min(r["K"] for r in rows)), float(max(r["K"] for r in rows))],
        "H_sd": float(np.std([r["H"] for r in rows], ddof=1)),
        "K_sd": float(np.std([r["K"] for r in rows], ddof=1)),
    }


def sample_cap_points(seed, n, radius):
    rng = np.random.default_rng(seed)
    cost = rng.uniform(np.cos(np.pi / 3), 1, n)
    theta = np.arccos(cost)
    az = rng.uniform(0, 2 * np.pi, n)
    points = np.column_stack([
        radius * np.sin(theta) * np.cos(az),
        radius * np.sin(theta) * np.sin(az),
        radius * (1 - np.cos(theta)),
    ])
    return points, None


def analytic_plane_saddle_test():
    # Structured plane, plus a graph-sampled hyperbolic paraboloid. The saddle
    # test checks curvature sign, not a clinical morphology label.
    axis = np.linspace(-6, 6, 41)
    x, y = np.meshgrid(axis, axis)
    plane = np.column_stack([x.ravel(), y.ravel(), np.zeros(x.size)])
    z = (x**2 - y**2) / 100.0
    saddle = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
    results = {}
    for name, points in (("plane", plane), ("saddle", saddle)):
        profile = compute_manifold_curvature(
            points, 0.0, k_neighbors=30,
            normal_orientation="none", random_seed=0,
        )
        results[name] = {
            "mean_H": profile.mean_mean_curvature_H,
            "mean_K": profile.mean_gaussian_curvature_K,
            "curvedness": profile.mean_curvedness,
        }
    return results


if __name__ == "__main__":
    main()
